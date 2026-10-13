import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date

from . import dates
from .llm import LLMError, LLMProvider

log = logging.getLogger(__name__)

INTENTS = {
    "create_entry", "list_entries", "search_entries",
    "update_entry", "delete_entry", "complete_entry", "unknown",
}
ACTION_INTENTS = {"update_entry", "delete_entry", "complete_entry"}
KINDS = {"task", "reminder", "meeting"}


@dataclass
class Parsed:
    intent: str = "unknown"
    kind: str | None = None
    title: str | None = None
    date_phrase: str | None = None
    time_phrase: str | None = None
    date_iso: str | None = None
    time_hhmm: str | None = None
    duration_minutes: int | None = None
    recurrence: str | None = None
    period_phrase: str | None = None
    query: str | None = None
    target: str | None = None
    new_title: str | None = None
    shift_minutes: int | None = None
    source: str = "rules"
    notes: list[str] = field(default_factory=list)


LIST_RE = re.compile(
    r"(что у меня|что (?:есть|запланировано|намечено|на )|какие (?:у меня )?(?:дела|задачи|встречи|напоминания|планы)"
    r"|есть ли у меня|покажи|список|расписани|мои (?:дела|задачи|встречи|планы)|чем я занят|планы на)"
)
SEARCH_RE = re.compile(
    r"(найди|найти|поищи|ищи|где (?:была|был|запись)|была (?:задача|встреча|запись)|какая (?:была )?задача про"
    r"|вспомни|когда (?:у меня )?(?:была|был|будет))"
)
CREATE_RE = re.compile(
    r"(напомни|напомнить|напомина|поставь|создай|добавь|запланируй|запиши|назначь|встреч|созвон|совещани|планерк|задач|нужно|надо|не забыть|каждw* (?:день|недел|месяц))"
)
DELETE_RE = re.compile(r"^(?:пожалуйста\s+)?(?:удали(?:ть|те)?|убери|сотри|отмени(?:ть|те)?)\b")
COMPLETE_RE = re.compile(
    r"(?:^|\b)(?:отмет\w*|помет\w*)\b.*\b(?:выполн\w*|сделан\w*|готов\w*|законч\w*)"
    r"|^(?:я\s+)?(?:уже\s+)?(?:сделал\w*|выполнил\w*|закончил\w*)\b|^готово\b"
)
UPDATE_RE = re.compile(
    r"^(?:пожалуйста\s+)?(?:а\s+)?(?:перенеси(?:те)?|передвинь|сдвинь|измени|поменяй|переименуй|перепиши|"
    r"поставь\s+(?:е[её]|его)\s+на|сделай\s+(?:е[её]|его|эту|этот))\b"
)
RENAME_RE = re.compile(r"\bпереименуй\w*\s+(.*?)\s*\bв\s+[«\"]?(.+?)[»\"]?\s*$", re.I)
PRONOUN_RE = re.compile(r"\b(?:е[её]|его|это|эту|этот|эта|их|него|нее|неё)\b")
MEETING_RE = re.compile(r"(встреч|созвон|совещани|планерк|собрани|митинг|переговор)")
REMINDER_RE = re.compile(r"(напомни|напомнить|напоминай|напоминать|напоминани|не забыть)")
TASK_RE = re.compile(r"\bзадач")

TRIGGER_RE = re.compile(
    r"^(?:пожалуйста\s+)?(?:напомина(?:й|йте)(?:\s+мне)?|напомни(?:те)?(?:\s+мне)?|поставь(?:те)?|создай(?:те)?|добавь(?:те)?|запланируй(?:те)?|запиши(?:те)?|назначь(?:те)?)"
    r"(?:\s+(?:мне|пожалуйста))*\s*:?\s*",
    re.I,
)
ACTION_TRIGGER_RE = re.compile(
    r"^(?:пожалуйста\s+)?(?:а\s+)?(?:удали(?:ть|те)?|убери|сотри|отмени(?:ть|те)?|перенеси(?:те)?|передвинь|сдвинь|"
    r"измени|поменяй|перепиши|отметь|пометь|сделай|я\s+сделал\w*|сделал\w*|выполнил\w*|закончил\w*)\s*",
    re.I,
)
ACTION_TAIL_RE = re.compile(r"\s*(?:как\s+)?(?:выполненн\w*|сделанн\w*|готов\w*)\s*$", re.I)
KIND_WORD_RE = re.compile(r"^(?:новую\s+|новое\s+|новая\s+)?(?:задачу|задача|напоминание|встречу|встреча|созвон)\s*", re.I)
FILLER_RE = re.compile(
    r"^(?:что(?:бы)?\s+(?:мне\s+)?(?:нужно|надо)\s+|(?:мне\s+)?(?:нужно|надо)\s+|о том,?\s+что\s+|про\s+|что\s+)",
    re.I,
)
SEARCH_STRIP_RE = re.compile(
    r"^(?:я\s+)?когда-то\s+\w+\s+|^(?:найди|найти|поищи|ищи|вспомни)(?:\s+мне)?\s*|^когда\s+(?:у\s+меня\s+)?(?:была|был|будет)\s*"
    r"|,?\s*найди\s+(?:е[её]|его)\s*$|(?:задачу|встречу|запись|напоминание|заметку)\s*|\b(?:про|о|об|насчет|по поводу|по)\s+"
)


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;:-—!?")


def _cut(text: str, span: tuple[int, int] | None) -> str:
    if not span:
        return text
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _kind_mentioned(t: str) -> str | None:
    if MEETING_RE.search(t):
        return "meeting"
    if re.search(r"напоминани", t):
        return "reminder"
    if TASK_RE.search(t):
        return "task"
    return None


def _strip_when(original: str, t: str, today: date):
    rec_m = dates.find_recurrence(t)
    rel_m = dates.find_relative_time_span(t)
    date_m = None if rel_m else dates.find_date(t, today)
    time_m = rel_m or dates.find_time(t)
    if rec_m and time_m and rec_m.span[0] <= time_m.span[0] < rec_m.span[1]:
        time_m = None
    dur_m = dates.find_duration(t)
    rest = original
    for m in sorted([x for x in (rec_m, date_m, time_m, dur_m) if x], key=lambda x: -x.span[0]):
        rest = _cut(rest, m.span)
    return _clean_spaces(rest), rec_m, date_m, time_m, dur_m


def _phrases(t: str, date_m, time_m) -> tuple[str | None, str | None]:
    return (
        t[date_m.span[0]:date_m.span[1]] if date_m else None,
        t[time_m.span[0]:time_m.span[1]] if time_m else None,
    )


def parse_rules(text: str, today: date) -> Parsed:
    original = re.sub(r"\s+", " ", text).strip()
    t = original.lower().replace("ё", "е")

    action = (
        "delete_entry" if DELETE_RE.search(t)
        else "complete_entry" if COMPLETE_RE.search(t)
        else "update_entry" if UPDATE_RE.search(t)
        else None
    )
    if action:
        new_title = None
        rename = RENAME_RE.search(original) if action == "update_entry" else None
        if rename:
            new_title = _clean_spaces(rename.group(2))
            original = original[: rename.start()] + " " + rename.group(1)
            t = original.lower()
        shift_m = dates.find_shift(t) if action == "update_entry" else None
        if shift_m:
            original = _cut(original, shift_m.span)
            t = original.lower().replace("ё", "е")
        loose_time = None
        if action == "update_entry" and not shift_m and not dates.find_time(t):
            loose = dates.LOOSE_TIME_RE.search(t)
            if loose:
                loose_time = dates.find_loose_time(t)
                original = _cut(original, loose.span())
                t = original.lower().replace("ё", "е")
        rest, rec_m, date_m, time_m, dur_m = _strip_when(original, t, today)
        rest = ACTION_TRIGGER_RE.sub("", rest)
        rest = ACTION_TAIL_RE.sub("", rest)
        rest = re.sub(r"\b(?:на|в|к)\s*$", "", rest.strip())
        target_text = _clean_spaces(PRONOUN_RE.sub(" ", rest))
        date_phrase, time_phrase = _phrases(t, date_m, time_m) if action == "update_entry" else (None, None)
        if loose_time and not time_phrase:
            time_phrase = loose_time.strftime("%H:%M")
        return Parsed(
            intent=action,
            kind=_kind_mentioned(target_text.lower()),
            target=target_text or None,
            new_title=new_title[:1].upper() + new_title[1:] if new_title else None,
            date_phrase=date_phrase,
            time_phrase=time_phrase,
            duration_minutes=dur_m.value if dur_m and action == "update_entry" else None,
            recurrence=rec_m.value if rec_m and action == "update_entry" else None,
            shift_minutes=shift_m.value if shift_m else None,
        )

    if SEARCH_RE.search(t):
        query = _clean_spaces(SEARCH_STRIP_RE.sub(" ", t))
        return Parsed(intent="search_entries", query=query or t, kind=_kind_mentioned(t))
    if LIST_RE.search(t) and not REMINDER_RE.match(t):
        return Parsed(intent="list_entries", period_phrase=t, kind=_kind_mentioned(t))
    if not CREATE_RE.search(t) and not dates.find_date(t, today) and not dates.find_time(t):
        return Parsed(intent="unknown")

    kind = "meeting" if MEETING_RE.search(t) else "reminder" if REMINDER_RE.search(t) else "task"
    rest, rec_m, date_m, time_m, dur_m = _strip_when(original, t, today)
    rest = TRIGGER_RE.sub("", rest)
    rest = KIND_WORD_RE.sub("", rest)
    rest = FILLER_RE.sub("", rest)
    title = _clean_spaces(rest)
    if kind == "meeting" and title and title.lower().startswith("с "):
        title = "встреча " + title
    title = title[:1].upper() + title[1:] if title else None
    date_phrase, time_phrase = _phrases(t, date_m, time_m)

    return Parsed(
        intent="create_entry",
        kind=kind,
        title=title,
        date_phrase=date_phrase,
        time_phrase=time_phrase,
        duration_minutes=dur_m.value if dur_m else None,
        recurrence=rec_m.value if rec_m else None,
    )


SYSTEM_PROMPT = """Ты разбираешь команды голосового ассистента задач, напоминаний и встреч.
Ответь ТОЛЬКО одним JSON-объектом без пояснений и без Markdown.

Поля:
- "intent":
  "create_entry" — создать задачу/напоминание/встречу;
  "list_entries" — показать записи за период;
  "search_entries" — найти старую запись или узнать, когда она была;
  "update_entry" — перенести, переименовать или изменить существующую запись;
  "delete_entry" — удалить или отменить существующую запись;
  "complete_entry" — отметить запись выполненной;
  "unknown" — всё остальное.
- "kind": "task" | "reminder" | "meeting" | null. Встреча, созвон, совещание — meeting. «Напомни» — reminder. Иначе дело — task.
  Для list_entries и search_entries укажи kind, только если пользователь явно спрашивает про задачи, встречи или напоминания.
- "title": для create_entry — короткое название в именительном падеже с большой буквы, без «напомни», «поставь», без даты и времени. Для встречи — «Встреча с …».
- "date_phrase": слова из запроса, которыми названа дата («завтра», «в пятницу», «до конца недели»), или null.
- "time_phrase": слова из запроса, которыми названо время («в 15:00», «в 3 дня», «через два часа»), или null.
- "date": та же дата в формате YYYY-MM-DD, посчитанная от сегодняшней, или null.
- "time": то же время в формате HH:MM (24 часа), или null.
- "duration_minutes": длительность встречи в минутах, если названа, иначе null.
- "recurrence": "daily" | "weekdays" | "weekly" | "monthly" | null — если запись повторяется («каждый день», «по будням», «каждый понедельник», «каждый месяц»).
- "period_phrase": для list_entries — период словами («сегодня», «завтра», «на этой неделе»), иначе null.
- "query": для search_entries — о чём запись, без слов «найди», «когда была», «задачу про», иначе null.
- "target": для update/delete/complete — какую запись менять, своими словами («встреча с Иваном», «задача про отчёт»).
  null, если пользователь говорит «её», «его», «эту» про запись из предыдущего ответа.
- "new_title": для update_entry — новое название, если пользователь переименовывает, иначе null.
- "shift_minutes": для update_entry — сдвиг в минутах относительно текущего времени записи («на час позже» = 60, «на полчаса раньше» = -30, «на день позже» = 1440), иначе null.
Если пользователь сообщает, что уже сделал дело («я купил молоко», «позвонил маме»), это complete_entry с target = это дело.

Для update_entry date_phrase/time_phrase/date/time — это НОВЫЕ дата и время.
Не придумывай дату и время, если их нет в запросе.

Примеры (сегодня 2026-10-09, пятница):
«Напомни мне завтра в 10 утра позвонить Ивану» → {"intent":"create_entry","kind":"reminder","title":"Позвонить Ивану","date_phrase":"завтра","time_phrase":"в 10 утра","date":"2026-10-10","time":"10:00","duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":null,"new_title":null}
«Поставь задачу купить билеты до конца недели» → {"intent":"create_entry","kind":"task","title":"Купить билеты","date_phrase":"до конца недели","time_phrase":null,"date":"2026-10-11","time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":null,"new_title":null}
«Планёрка каждый понедельник в 10» → {"intent":"create_entry","kind":"meeting","title":"Планёрка","date_phrase":"каждый понедельник","time_phrase":"в 10","date":"2026-10-12","time":"10:00","duration_minutes":null,"recurrence":"weekly","period_phrase":null,"query":null,"target":null,"new_title":null}
«Какие задачи у меня на завтра?» → {"intent":"list_entries","kind":"task","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":"на завтра","query":null,"target":null,"new_title":null}
«Что у меня на этой неделе?» → {"intent":"list_entries","kind":null,"title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":"на этой неделе","query":null,"target":null,"new_title":null}
«Когда у меня была встреча с Иваном?» → {"intent":"search_entries","kind":"meeting","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":"встреча с Иваном","target":null,"new_title":null}
«А перенеси её на завтра» → {"intent":"update_entry","kind":null,"title":null,"date_phrase":"на завтра","time_phrase":null,"date":"2026-10-10","time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":null,"new_title":null}
«Перенеси встречу с дизайнером на 17:00» → {"intent":"update_entry","kind":"meeting","title":null,"date_phrase":null,"time_phrase":"на 17:00","date":null,"time":"17:00","duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":"встреча с дизайнером","new_title":null}
«Сдвинь её на час позже» → {"intent":"update_entry","kind":null,"title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":null,"new_title":null,"shift_minutes":60}
«Я купил молоко» → {"intent":"complete_entry","kind":"task","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":"купить молоко","new_title":null,"shift_minutes":null}
«Удали напоминание про таблетки» → {"intent":"delete_entry","kind":"reminder","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":"напоминание про таблетки","new_title":null}
«Отметь задачу про отчёт выполненной» → {"intent":"complete_entry","kind":"task","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"recurrence":null,"period_phrase":null,"query":null,"target":"задача про отчёт","new_title":null}
"""

CLARIFY_HINT = """Сейчас пользователь отвечает на уточняющий вопрос по черновику записи: {draft}
Если ответ дополняет черновик (например, называет время или дату), верни intent "create_entry" и только те поля, что названы в ответе.
Если это новая, не связанная команда — разбери её как обычно."""

WEEKDAYS_FULL = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе нет JSON")
    return json.loads(raw[start: end + 1])


def _str(value) -> str | None:
    if isinstance(value, str) and value.strip() and value.strip().lower() not in {"null", "none"}:
        return value.strip()
    return None


def _from_llm(data: dict) -> Parsed:
    intent = data.get("intent")
    if intent not in INTENTS:
        raise ValueError(f"неизвестный intent: {intent}")
    kind = data.get("kind")
    duration = data.get("duration_minutes")
    recurrence = data.get("recurrence")
    return Parsed(
        intent=intent,
        kind=kind if kind in KINDS else None,
        title=_str(data.get("title")),
        date_phrase=_str(data.get("date_phrase")),
        time_phrase=_str(data.get("time_phrase")),
        date_iso=_str(data.get("date")),
        time_hhmm=_str(data.get("time")),
        duration_minutes=duration if isinstance(duration, int) and duration > 0 else None,
        recurrence=recurrence if recurrence in dates.RECURRENCES else None,
        period_phrase=_str(data.get("period_phrase")),
        query=_str(data.get("query")),
        target=_str(data.get("target")),
        new_title=_str(data.get("new_title")),
        shift_minutes=data.get("shift_minutes") if isinstance(data.get("shift_minutes"), int) and data.get("shift_minutes") else None,
        source="llm",
    )


def parse(text: str, today: date, llm: LLMProvider | None, draft: dict | None = None) -> Parsed:
    rules = parse_rules(text, today)
    if llm is None:
        return rules
    user = f"Сегодня {today.isoformat()}, {WEEKDAYS_FULL[today.weekday()]}.\n"
    if draft:
        user += CLARIFY_HINT.format(draft=json.dumps(draft, ensure_ascii=False)) + "\n"
    user += f"Запрос: «{text}»"
    try:
        parsed = _from_llm(_extract_json(llm.complete(SYSTEM_PROMPT, user)))
    except (LLMError, ValueError, json.JSONDecodeError) as e:
        log.warning("LLM не разобрал запрос, используем правила: %s", e)
        rules.notes.append(f"llm_fallback: {e}")
        return rules
    if parsed.intent in ("create_entry", "update_entry"):
        if parsed.intent == rules.intent or parsed.intent == "create_entry":
            parsed.date_phrase = parsed.date_phrase or rules.date_phrase
            parsed.time_phrase = parsed.time_phrase or rules.time_phrase
            parsed.duration_minutes = parsed.duration_minutes or rules.duration_minutes
            parsed.recurrence = parsed.recurrence or rules.recurrence
        if parsed.intent == "create_entry":
            parsed.kind = parsed.kind or rules.kind
    if parsed.intent == "update_entry" and not parsed.new_title and rules.intent == "update_entry":
        parsed.new_title = rules.new_title
    if parsed.intent in ACTION_INTENTS:
        if not parsed.target:
            candidate = parsed.query or (parsed.title if parsed.title != parsed.new_title else None)
            parsed.target = candidate or (rules.target if rules.intent == parsed.intent else None)
        if parsed.intent == "update_entry" and not parsed.shift_minutes:
            parsed.shift_minutes = rules.shift_minutes if rules.intent == "update_entry" else dates.find_shift(text) and dates.find_shift(text).value
    if parsed.intent == "search_entries" and not parsed.query:
        parsed.query = rules.query or text
    if parsed.intent == "list_entries" and not parsed.period_phrase:
        parsed.period_phrase = text
    return parsed
