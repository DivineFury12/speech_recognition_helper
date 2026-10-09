import logging
import re
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import dates
from .llm import LLMProvider
from .nlu import CREATE_RE, Parsed, parse
from .search import search_entries
from .storage import Storage

log = logging.getLogger(__name__)

KIND_LABEL = {"task": "задачу", "reminder": "напоминание", "meeting": "встречу"}
KIND_SHORT = {"task": "задача", "reminder": "напоминание", "meeting": "встреча"}
CANCEL_RE = re.compile(r"^\s*(отмена|отмени(?:ть)?|не надо|забудь|стоп)\b", re.I)
HELP_TEXT = (
    "Я умею создавать задачи, напоминания и встречи, показывать записи за период и искать старые. "
    "Например: «Напомни завтра в 10 позвонить Ивану», «Что у меня на неделе?», «Найди задачу про презентацию»."
)


class InvalidTimezone(ValueError):
    pass


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise InvalidTimezone(f"Неизвестный часовой пояс: {name}") from e


class Assistant:
    def __init__(self, storage: Storage, llm: LLMProvider | None) -> None:
        self.storage = storage
        self.llm = llm

    def handle(self, text: str, tz_name: str, conversation_id: str | None = None,
               now: datetime | None = None) -> dict[str, Any]:
        tz = resolve_timezone(tz_name)
        now = (now or datetime.now(tz)).astimezone(tz)
        conversation_id = self.storage.ensure_conversation(conversation_id)
        self.storage.add_message(conversation_id, "user", text)

        draft = self.storage.get_draft(conversation_id)
        if draft and CANCEL_RE.search(text):
            self.storage.delete_draft(conversation_id)
            result = self._turn("Хорошо, отменил.", "no_action", "create_entry", [])
        else:
            parsed = parse(text, now.date(), self.llm, draft)
            result = self._dispatch(parsed, text, tz_name, now, conversation_id, draft)

        self.storage.add_message(conversation_id, "assistant", result["assistant_text"])
        return {
            "conversation_id": conversation_id,
            "turn_id": str(uuid.uuid4()),
            "user_text": text,
            **result,
        }

    @staticmethod
    def _turn(text: str, status: str, intent: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
        return {"assistant_text": text, "status": status, "intent": intent, "entries": entries}

    def _dispatch(self, parsed: Parsed, text: str, tz_name: str, now: datetime,
                  conversation_id: str, draft: dict | None) -> dict[str, Any]:
        if draft and parsed.intent in ("create_entry", "unknown") and not self._is_new_command(parsed, text, draft):
            return self._create(parsed, text, tz_name, now, conversation_id, draft)
        if draft:
            self.storage.delete_draft(conversation_id)
        if parsed.intent == "create_entry":
            return self._create(parsed, text, tz_name, now, conversation_id, None)
        if parsed.intent == "list_entries":
            return self._list(parsed, now.date())
        if parsed.intent == "search_entries":
            return self._search(parsed, text, now.date())
        return self._turn(HELP_TEXT, "no_action", "unknown", [])

    def _resolve(self, parsed: Parsed, now: datetime) -> dict[str, Any]:
        today = now.date()
        rel = dates.find_relative_datetime(" ".join(filter(None, [parsed.date_phrase, parsed.time_phrase])), now)
        d = dates.parse_date(parsed.date_phrase, today)
        t = dates.parse_time(parsed.time_phrase)
        if rel:
            d, t = rel.value.date(), rel.value.time().replace(second=0, microsecond=0)
        if d is None and parsed.date_phrase and parsed.date_iso:
            try:
                d = date.fromisoformat(parsed.date_iso)
            except ValueError:
                pass
        if t is None and parsed.time_phrase and parsed.time_hhmm:
            try:
                t = time.fromisoformat(parsed.time_hhmm)
            except ValueError:
                pass
        return {
            "date": d.isoformat() if d else None,
            "time": t.strftime("%H:%M") if t else None,
        }

    @staticmethod
    def _is_new_command(parsed: Parsed, text: str, draft: dict) -> bool:
        if draft.get("missing") == "title" or parsed.intent != "create_entry" or not parsed.title:
            return False
        return bool(CREATE_RE.search(dates.normalize(text)))

    def _create(self, parsed: Parsed, text: str, tz_name: str, now: datetime, conversation_id: str,
                draft: dict | None) -> dict[str, Any]:
        resolved = self._resolve(parsed, now)
        fields = dict(draft or {})
        fields.pop("missing", None)
        fields.setdefault("kind", parsed.kind or "task")
        if not fields.get("title"):
            if draft and draft.get("missing") == "title":
                fields["title"] = parsed.title or _as_title(text)
            elif parsed.title:
                fields["title"] = parsed.title
        for key in ("date", "time"):
            if resolved[key]:
                fields[key] = resolved[key]
        if parsed.duration_minutes:
            fields["duration_minutes"] = parsed.duration_minutes

        kind = fields["kind"]
        today = now.date()
        if kind in ("meeting", "reminder") and fields.get("time") and not fields.get("date"):
            slot = datetime.combine(today, time.fromisoformat(fields["time"]), now.tzinfo)
            fields["date"] = (today if slot > now else today + timedelta(days=1)).isoformat()

        missing = self._missing(fields)
        if missing:
            fields["missing"] = missing
            self.storage.save_draft(conversation_id, fields)
            return self._turn(self._question(fields, missing), "needs_clarification", "create_entry", [])

        self.storage.delete_draft(conversation_id)
        entry = self.storage.create_entry(
            kind=kind,
            title=fields["title"],
            date=fields.get("date"),
            time=fields.get("time"),
            timezone=tz_name,
            duration_minutes=fields.get("duration_minutes") or (60 if kind == "meeting" else None),
        )
        when = dates.format_when(
            date.fromisoformat(entry["date"]) if entry["date"] else None,
            time.fromisoformat(entry["time"]) if entry["time"] else None,
            today,
        )
        return self._turn(f"Готово, создал {KIND_LABEL[kind]}: «{entry['title']}», {when}.", "completed", "create_entry", [entry])

    @staticmethod
    def _missing(fields: dict) -> str | None:
        if not fields.get("title"):
            return "title"
        if fields["kind"] in ("meeting", "reminder"):
            if not fields.get("date") and not fields.get("time"):
                return "datetime"
            if not fields.get("time"):
                return "time"
            if not fields.get("date"):
                return "date"
        return None

    @staticmethod
    def _question(fields: dict, missing: str) -> str:
        kind = fields["kind"]
        if missing == "title":
            return {"meeting": "С кем или о чём встреча?", "reminder": "О чём напомнить?"}.get(kind, "Как назвать задачу?")
        if missing == "time":
            return "На какое время поставить встречу?" if kind == "meeting" else "Во сколько напомнить?"
        if missing == "date":
            return "На какой день?"
        return "Когда поставить встречу?" if kind == "meeting" else "Когда напомнить?"

    def _list(self, parsed: Parsed, today: date) -> dict[str, Any]:
        start, end = dates.parse_period(parsed.period_phrase, today)
        entries = self.storage.list_entries(start.isoformat(), end.isoformat())
        label = self._period_label(start, end, today)
        if not entries:
            return self._turn(f"{label.capitalize()} ничего нет.", "completed", "list_entries", [])
        lines = [self._line(e, today, show_date=start != end) for e in entries]
        return self._turn(f"{label.capitalize()} {self._count(len(entries))}:\n" + "\n".join(lines),
                          "completed", "list_entries", entries)

    def _search(self, parsed: Parsed, text: str, today: date) -> dict[str, Any]:
        query = parsed.query or text
        entries = search_entries(query, self.storage.all_entries(parsed.kind))
        if not entries:
            return self._turn(f"Не нашёл ничего похожего на «{query}».", "completed", "search_entries", [])
        lines = [self._line(e, today, show_date=True) for e in entries]
        return self._turn("Нашёл:\n" + "\n".join(lines), "completed", "search_entries", entries)

    @staticmethod
    def _line(entry: dict, today: date, show_date: bool) -> str:
        d = date.fromisoformat(entry["date"]) if entry["date"] else None
        t = time.fromisoformat(entry["time"]) if entry["time"] else None
        when = dates.format_when(d if show_date else None, t, today) if (show_date or t) else ""
        parts = [p for p in (when, entry["title"]) if p]
        return f"• {' — '.join(parts)} ({KIND_SHORT[entry['kind']]})"

    @staticmethod
    def _period_label(start: date, end: date, today: date) -> str:
        if start == end:
            if start == today:
                return "на сегодня"
            if start == today + timedelta(days=1):
                return "на завтра"
            return f"на {dates.format_date(start)}"
        return f"с {dates.format_date(start)} по {dates.format_date(end)}"

    @staticmethod
    def _count(n: int) -> str:
        if n % 10 == 1 and n % 100 != 11:
            word = "запись"
        elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            word = "записи"
        else:
            word = "записей"
        return f"{n} {word}"


def _as_title(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text).strip(" .,!?")
    return text[:1].upper() + text[1:] if text else None
