import logging
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import dates
from .llm import LLMProvider
from .nlu import ACTION_INTENTS, CREATE_RE, Parsed, parse, parse_rules
from .search import lexical, search_entries, semantic, stems
from .storage import DEFAULT_USER, Storage

log = logging.getLogger(__name__)

KIND_ACC = {"task": "задачу", "reminder": "напоминание", "meeting": "встречу"}
KIND_SHORT = {"task": "задача", "reminder": "напоминание", "meeting": "встреча"}
KIND_PLURAL = {
    "task": ("задача", "задачи", "задач"),
    "reminder": ("напоминание", "напоминания", "напоминаний"),
    "meeting": ("встреча", "встречи", "встреч"),
    None: ("запись", "записи", "записей"),
}
CANCEL_RE = re.compile(r"^\s*(?:нет,?\s*)?(?:отмена|отмени(?:те)?|не надо|не нужно|забудь|стоп|хватит)\s*[.!]*\s*$", re.I)
YES_RE = re.compile(
    r"^\s*(?:да|ага|угу|ок|окей|давай|конечно|верно|подтверждаю|ставь|удаляй|удали|всё равно|все равно|пусть)\b", re.I
)
NO_RE = re.compile(r"^\s*(?:нет|не надо|не нужно|не ставь|не удаляй|не)\b", re.I)
ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "последн": -1}
HELP_TEXT = (
    "Я умею создавать задачи, напоминания и встречи (в том числе повторяющиеся), показывать записи за период, "
    "искать старые, переносить, переименовывать, удалять и отмечать выполненными. "
    "Например: «Напомни завтра в 10 позвонить Ивану», «Что у меня на неделе?», «Найди задачу про презентацию», "
    "«Перенеси её на пятницу»."
)
MAX_CHOICES = 5


class InvalidTimezone(ValueError):
    pass


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise InvalidTimezone(f"Неизвестный часовой пояс: {name}") from e


@dataclass
class Ctx:
    user_id: str
    tz_name: str
    now: datetime
    cid: str

    @property
    def today(self) -> date:
        return self.now.date()


def _plural(n: int, kind: str | None) -> str:
    one, few, many = KIND_PLURAL[kind]
    if n % 10 == 1 and n % 100 != 11:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def _when(entry: dict, today: date, show_date: bool = True) -> str:
    d = date.fromisoformat(entry["date"]) if entry.get("date") else None
    t = time.fromisoformat(entry["time"]) if entry.get("time") else None
    rule = entry.get("recurrence")
    if rule:
        base = dates.RECURRENCE_LABEL[rule]
        start = dates.format_when(d, None, today) if d and d > today else ""
        at = f"в {t.strftime('%H:%M')}" if t else ""
        return " ".join(p for p in (base, f"с {start}" if start else "", at) if p)
    if not show_date and t:
        return f"в {t.strftime('%H:%M')}"
    return dates.format_when(d if show_date else None, t, today)


def _describe(entry: dict, today: date, accusative: bool = False) -> str:
    when = _when(entry, today)
    span = ""
    if entry["kind"] == "meeting" and entry.get("time") and entry.get("duration_minutes"):
        start = datetime.combine(date.today(), time.fromisoformat(entry["time"]))
        span = f"–{(start + timedelta(minutes=entry['duration_minutes'])).strftime('%H:%M')}"
    kind = (KIND_ACC if accusative else KIND_SHORT)[entry["kind"]]
    return f"{kind} «{entry['title']}» ({when}{span})"


def _as_title(text: str) -> str | None:
    text = re.sub(r"\s+", " ", text).strip(" .,!?")
    return text[:1].upper() + text[1:] if text else None


class Assistant:
    def __init__(self, storage: Storage, llm: LLMProvider | None, index=None) -> None:
        self.storage = storage
        self.llm = llm
        self.index = index

    def handle(self, text: str, tz_name: str, conversation_id: str | None = None,
               now: datetime | None = None, user_id: str = DEFAULT_USER) -> dict[str, Any]:
        tz = resolve_timezone(tz_name)
        now = (now or datetime.now(tz)).astimezone(tz)
        cid = self.storage.ensure_conversation(conversation_id, user_id)
        self.storage.add_message(cid, "user", text)
        ctx = Ctx(user_id, tz_name, now, cid)

        draft = self._load_draft(cid)
        if draft and CANCEL_RE.match(text):
            self.storage.delete_draft(cid)
            result = self._turn("Хорошо, отменил.", "no_action", draft.get("intent", "unknown"), [])
        elif draft and draft["type"] == "confirm":
            result = self._on_confirm(text, draft, ctx)
        elif draft and draft["type"] == "choose":
            result = self._on_choose(text, draft, ctx)
        else:
            parsed = parse(text, ctx.today, self.llm, draft["fields"] if draft else None)
            result = self._dispatch(parsed, text, ctx, draft)

        focus = result.pop("focus", None)
        if focus is not None:
            self.storage.set_context(cid, {"focus": focus})
        self.storage.add_message(cid, "assistant", result["assistant_text"])
        return {"conversation_id": cid, "turn_id": str(uuid.uuid4()), "user_text": text, **result}

    @staticmethod
    def _turn(text: str, status: str, intent: str, entries: list[dict[str, Any]],
              focus: list[str] | None = None) -> dict[str, Any]:
        result = {"assistant_text": text, "status": status, "intent": intent, "entries": entries}
        if focus is not None:
            result["focus"] = focus
        return result

    def _load_draft(self, cid: str) -> dict | None:
        draft = self.storage.get_draft(cid)
        if draft is None or "type" in draft:
            return draft
        fields = {k: v for k, v in draft.items() if k != "missing"}
        return {"type": "create", "intent": "create_entry", "fields": fields, "missing": draft.get("missing")}

    def _ask(self, ctx: Ctx, draft: dict, question: str, entries: list[dict] | None = None,
             focus: list[str] | None = None) -> dict[str, Any]:
        self.storage.save_draft(ctx.cid, {**draft, "question": question})
        return self._turn(question, "needs_clarification", draft["intent"], entries or [], focus)

    def _dispatch(self, parsed: Parsed, text: str, ctx: Ctx, draft: dict | None) -> dict[str, Any]:
        if draft and parsed.intent in ("create_entry", "unknown") and not self._is_new_command(parsed, text, draft):
            return self._create(parsed, text, ctx, draft)
        if draft:
            self.storage.delete_draft(ctx.cid)
        if parsed.intent == "create_entry":
            return self._create(parsed, text, ctx, None)
        if parsed.intent == "list_entries":
            return self._list(parsed, ctx)
        if parsed.intent == "search_entries":
            return self._search(parsed, text, ctx)
        if parsed.intent in ACTION_INTENTS:
            return self._action(parsed, ctx)
        return self._turn(HELP_TEXT, "no_action", "unknown", [])

    @staticmethod
    def _is_new_command(parsed: Parsed, text: str, draft: dict) -> bool:
        if draft.get("missing") == "title" or parsed.intent != "create_entry" or not parsed.title:
            return False
        return bool(CREATE_RE.search(dates.normalize(text)))

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
        return {"date": d.isoformat() if d else None, "time": t.strftime("%H:%M") if t else None}

    @staticmethod
    def _fill_date(fields: dict, now: datetime) -> None:
        if fields.get("time") and not fields.get("date") and (fields["kind"] != "task" or fields.get("recurrence")):
            slot = datetime.combine(now.date(), time.fromisoformat(fields["time"]), now.tzinfo)
            fields["date"] = (now.date() if slot > now else now.date() + timedelta(days=1)).isoformat()
        if fields.get("recurrence") and not fields.get("date"):
            fields["date"] = now.date().isoformat()

    def _create(self, parsed: Parsed, text: str, ctx: Ctx, draft: dict | None) -> dict[str, Any]:
        resolved = self._resolve(parsed, ctx.now)
        fields = dict(draft["fields"]) if draft else {}
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
        if parsed.recurrence:
            fields["recurrence"] = parsed.recurrence
        self._fill_date(fields, ctx.now)

        missing = self._missing(fields)
        if missing:
            new_draft = {"type": "create", "intent": "create_entry", "fields": fields, "missing": missing}
            return self._ask(ctx, new_draft, self._question(fields, missing))
        return self._commit_create(fields, ctx)

    def _commit_create(self, fields: dict, ctx: Ctx, force: bool = False) -> dict[str, Any]:
        kind = fields["kind"]
        if kind == "meeting":
            fields.setdefault("duration_minutes", 60)
            fields["duration_minutes"] = fields["duration_minutes"] or 60
        duplicate = self.storage.find_duplicate(
            kind=kind, title=fields["title"], date=fields.get("date"), time=fields.get("time"),
            recurrence=fields.get("recurrence"), user_id=ctx.user_id,
        )
        if duplicate:
            self.storage.delete_draft(ctx.cid)
            return self._turn(f"Такая запись уже есть: {_describe(duplicate, ctx.today)}.",
                              "completed", "create_entry", [duplicate], [duplicate["id"]])
        conflict = None if force else self._conflict(fields, ctx.user_id)
        if conflict:
            draft = {"type": "confirm", "action": "create", "intent": "create_entry", "fields": fields}
            question = (f"В это время уже есть {_describe(conflict, ctx.today)}. "
                        "Всё равно поставить? Или назовите другое время.")
            return self._ask(ctx, draft, question, [conflict])
        self.storage.delete_draft(ctx.cid)
        entry = self.storage.create_entry(
            kind=kind,
            title=fields["title"],
            date=fields.get("date"),
            time=fields.get("time"),
            timezone=ctx.tz_name,
            duration_minutes=fields.get("duration_minutes") if kind == "meeting" else None,
            recurrence=fields.get("recurrence"),
            user_id=ctx.user_id,
        )
        self._index([entry])
        text = f"Готово, создал {KIND_ACC[kind]}: «{entry['title']}», {_when(entry, ctx.today)}."
        return self._turn(text, "completed", "create_entry", [entry], [entry["id"]])

    def _index(self, entries: list[dict]) -> None:
        if self.index is None:
            return
        try:
            self.index.index(entries)
        except Exception as e:
            log.warning("Не удалось проиндексировать запись: %s", e)

    def _conflict(self, fields: dict, user_id: str, exclude_id: str | None = None) -> dict | None:
        if fields.get("kind") != "meeting" or not fields.get("date") or not fields.get("time"):
            return None
        start = datetime.combine(date.fromisoformat(fields["date"]), time.fromisoformat(fields["time"]))
        end = start + timedelta(minutes=fields.get("duration_minutes") or 60)
        for other in self.storage.list_entries(fields["date"], fields["date"], kind="meeting", user_id=user_id):
            if other["id"] == exclude_id or not other["time"] or other.get("status") == "cancelled":
                continue
            o_start = datetime.combine(date.fromisoformat(other["date"]), time.fromisoformat(other["time"]))
            o_end = o_start + timedelta(minutes=other["duration_minutes"] or 60)
            if start < o_end and o_start < end:
                return other
        return None

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

    def _on_confirm(self, text: str, draft: dict, ctx: Ctx) -> dict[str, Any]:
        action = draft["action"]
        if action in ("create", "update"):
            resolved = self._resolve(parse_rules(text, ctx.today), ctx.now)
            new = {k: v for k, v in resolved.items() if v}
            if new:
                if action == "create":
                    return self._commit_create({**draft["fields"], **new}, ctx)
                return self._commit_update(draft["entry_id"], {**draft["changes"], **new}, ctx)
        if YES_RE.match(text):
            if action == "create":
                return self._commit_create(draft["fields"], ctx, force=True)
            if action == "update":
                return self._commit_update(draft["entry_id"], draft["changes"], ctx, force=True)
            return self._commit_delete(draft["entry_id"], ctx)
        if NO_RE.match(text):
            self.storage.delete_draft(ctx.cid)
            return self._turn("Хорошо, не буду.", "no_action", draft["intent"], [])
        self.storage.delete_draft(ctx.cid)
        return self._dispatch(parse(text, ctx.today, self.llm), text, ctx, None)

    def _targets(self, parsed: Parsed, ctx: Ctx) -> list[dict]:
        focus_ids = self.storage.get_context(ctx.cid).get("focus", [])
        focus = [e for e in (self.storage.get_entry(i, ctx.user_id) for i in focus_ids) if e]
        if parsed.kind:
            focus = [e for e in focus if e["kind"] == parsed.kind]
        if not parsed.target or not stems(parsed.target):
            if focus or not parsed.target:
                return focus
            return self.storage.all_entries(parsed.kind, ctx.user_id)[:MAX_CHOICES + 1]
        pool = self.storage.all_entries(parsed.kind, ctx.user_id)
        if parsed.intent == "complete_entry":
            pool.sort(key=lambda e: e["status"] == "completed")
        ranked = lexical(parsed.target, pool)
        if ranked:
            best = ranked[0][0]
            return [e for s, e in ranked if s == best][: MAX_CHOICES + 1]
        if self.index is not None:
            hits = [e for _, e in self.index.rank(parsed.target, pool, ctx.user_id)]
            if hits:
                return hits[: MAX_CHOICES + 1]
        return semantic(parsed.target, pool, self.llm)[: MAX_CHOICES + 1] if self.llm else []

    def _changes(self, parsed: Parsed, ctx: Ctx) -> dict[str, Any]:
        changes = {k: v for k, v in self._resolve(parsed, ctx.now).items() if v}
        if parsed.new_title:
            changes["title"] = parsed.new_title[:1].upper() + parsed.new_title[1:]
        if parsed.duration_minutes:
            changes["duration_minutes"] = parsed.duration_minutes
        if parsed.recurrence:
            changes["recurrence"] = parsed.recurrence
        return changes

    def _action(self, parsed: Parsed, ctx: Ctx) -> dict[str, Any]:
        changes = self._changes(parsed, ctx) if parsed.intent == "update_entry" else {}
        if parsed.intent == "update_entry" and not changes:
            return self._turn("Что изменить? Например: «перенеси на завтра в 17:00» или «переименуй в …».",
                              "no_action", parsed.intent, [])
        targets = self._targets(parsed, ctx)
        if not targets:
            text = (f"Не нашёл запись «{parsed.target}»." if parsed.target
                    else "Не понял, о какой записи речь. Назовите её, например: «перенеси встречу с Иваном на завтра».")
            return self._turn(text, "no_action", parsed.intent, [])
        if len(targets) > 1:
            choices = targets[:MAX_CHOICES]
            lines = [f"{i}. {_describe(e, ctx.today)}" for i, e in enumerate(choices, 1)]
            draft = {"type": "choose", "action": parsed.intent, "intent": parsed.intent, "changes": changes,
                     "candidates": [e["id"] for e in choices]}
            question = "Нашёл несколько записей:\n" + "\n".join(lines) + "\nКакую вы имеете в виду?"
            return self._ask(ctx, draft, question, choices, [e["id"] for e in choices])
        return self._perform(parsed.intent, targets[0], changes, ctx)

    def _perform(self, intent: str, entry: dict, changes: dict, ctx: Ctx, confirmed: bool = False) -> dict[str, Any]:
        if intent == "update_entry":
            return self._commit_update(entry["id"], changes, ctx)
        if intent == "delete_entry":
            if confirmed:
                return self._commit_delete(entry["id"], ctx)
            draft = {"type": "confirm", "action": "delete", "intent": "delete_entry", "entry_id": entry["id"]}
            return self._ask(ctx, draft, f"Удалить {_describe(entry, ctx.today, accusative=True)}?", [entry], [entry["id"]])
        self.storage.delete_draft(ctx.cid)
        done = self.storage.update_entry(entry["id"], {"status": "completed"}, ctx.user_id)
        return self._turn(f"Отметил выполненной: {KIND_SHORT[done['kind']]} «{done['title']}».",
                          "completed", "complete_entry", [done], [done["id"]])

    def _on_choose(self, text: str, draft: dict, ctx: Ctx) -> dict[str, Any]:
        candidates = [e for e in (self.storage.get_entry(i, ctx.user_id) for i in draft["candidates"]) if e]
        pick = None
        t = dates.normalize(text)
        number = re.search(r"\b(\d)\b", t)
        if number and 1 <= int(number.group(1)) <= len(candidates):
            pick = candidates[int(number.group(1)) - 1]
        if pick is None:
            for stem, index in ORDINALS.items():
                if re.search(rf"\b{stem}", t) and len(candidates) >= abs(index):
                    pick = candidates[index - 1 if index > 0 else index]
                    break
        if pick is None:
            ranked = lexical(text, candidates)
            if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
                pick = ranked[0][1]
        if pick is None:
            if draft.get("action") in ACTION_INTENTS and parse_rules(text, ctx.today).intent in ("unknown", draft["action"]):
                return self._ask(ctx, draft, draft["question"], candidates, [e["id"] for e in candidates])
            self.storage.delete_draft(ctx.cid)
            return self._dispatch(parse(text, ctx.today, self.llm), text, ctx, None)
        self.storage.delete_draft(ctx.cid)
        return self._perform(draft["action"], pick, draft.get("changes", {}), ctx, confirmed=True)

    def _commit_update(self, entry_id: str, changes: dict, ctx: Ctx, force: bool = False) -> dict[str, Any]:
        entry = self.storage.get_entry(entry_id, ctx.user_id)
        if entry is None:
            self.storage.delete_draft(ctx.cid)
            return self._turn("Этой записи уже нет.", "no_action", "update_entry", [])
        merged = {**entry, **changes}
        self._fill_date(merged, ctx.now)
        changes = {k: merged[k] for k in set(changes) | {"date"} if merged.get(k) != entry.get(k)}
        conflict = None if force else self._conflict(merged, ctx.user_id, exclude_id=entry_id)
        if conflict:
            draft = {"type": "confirm", "action": "update", "intent": "update_entry",
                     "entry_id": entry_id, "changes": changes}
            question = (f"В это время уже есть {_describe(conflict, ctx.today)}. "
                        "Всё равно перенести? Или назовите другое время.")
            return self._ask(ctx, draft, question, [conflict])
        self.storage.delete_draft(ctx.cid)
        updated = self.storage.update_entry(entry_id, changes, ctx.user_id)
        parts = []
        if {"date", "time", "recurrence"} & set(changes):
            parts.append(f"теперь {_when(updated, ctx.today)}")
        if "duration_minutes" in changes:
            parts.append(f"длительность {updated['duration_minutes']} мин")
        if "title" in changes:
            head = f"Готово, переименовал {KIND_ACC[updated['kind']]} в «{updated['title']}»"
        else:
            head = f"Готово: {KIND_SHORT[updated['kind']]} «{updated['title']}»"
        text = head + (" — " + ", ".join(parts) if parts else "") + "."
        return self._turn(text, "completed", "update_entry", [updated], [updated["id"]])

    def _commit_delete(self, entry_id: str, ctx: Ctx) -> dict[str, Any]:
        self.storage.delete_draft(ctx.cid)
        entry = self.storage.get_entry(entry_id, ctx.user_id)
        if entry is None or not self.storage.delete_entry(entry_id, ctx.user_id):
            return self._turn("Этой записи уже нет.", "no_action", "delete_entry", [])
        return self._turn(f"Удалил {KIND_ACC[entry['kind']]} «{entry['title']}».", "completed", "delete_entry", [], [])

    def _list(self, parsed: Parsed, ctx: Ctx) -> dict[str, Any]:
        start, end = dates.parse_period(parsed.period_phrase, ctx.today)
        entries = self.storage.list_entries(start.isoformat(), end.isoformat(), kind=parsed.kind, user_id=ctx.user_id)
        label = self._period_label(start, end, ctx.today).capitalize()
        focus = list(dict.fromkeys(e["id"] for e in entries))
        if not entries:
            what = "ничего" if parsed.kind is None else KIND_PLURAL[parsed.kind][2]
            return self._turn(f"{label} {what} нет.", "completed", "list_entries", [], focus)
        lines = [self._line(e, ctx.today, show_date=start != end) for e in entries]
        text = f"{label} {len(entries)} {_plural(len(entries), parsed.kind)}:\n" + "\n".join(lines)
        return self._turn(text, "completed", "list_entries", entries, focus)

    def _search(self, parsed: Parsed, text: str, ctx: Ctx) -> dict[str, Any]:
        query = parsed.query or text
        pool = self.storage.all_entries(parsed.kind, ctx.user_id)
        entries = search_entries(query, pool, llm=self.llm, index=self.index, user_id=ctx.user_id)
        if not entries and parsed.kind:
            entries = search_entries(query, self.storage.all_entries(None, ctx.user_id), llm=self.llm,
                                     index=self.index, user_id=ctx.user_id)
        if not entries:
            return self._turn(f"Не нашёл ничего похожего на «{query}».", "completed", "search_entries", [], [])
        lines = [self._line(e, ctx.today, show_date=True) for e in entries]
        return self._turn("Нашёл:\n" + "\n".join(lines), "completed", "search_entries", entries,
                          [e["id"] for e in entries])

    @staticmethod
    def _line(entry: dict, today: date, show_date: bool) -> str:
        once = {**entry, "recurrence": None}
        when = _when(once, today, show_date) if (show_date or entry.get("time")) else ""
        done = " ✓" if entry.get("status") == "completed" else ""
        repeat = f", {dates.RECURRENCE_LABEL[entry['recurrence']]}" if entry.get("recurrence") else ""
        parts = [p for p in (when, entry["title"]) if p]
        return f"• {' — '.join(parts)} ({KIND_SHORT[entry['kind']]}{repeat}){done}"

    @staticmethod
    def _period_label(start: date, end: date, today: date) -> str:
        if start == end:
            if start == today:
                return "на сегодня"
            if start == today + timedelta(days=1):
                return "на завтра"
            return f"на {dates.format_date(start)}"
        return f"с {dates.format_date(start)} по {dates.format_date(end)}"
