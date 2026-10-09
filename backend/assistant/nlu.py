import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date

from . import dates
from .llm import LLMError, LLMProvider

log = logging.getLogger(__name__)

INTENTS = {"create_entry", "list_entries", "search_entries", "unknown"}
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
    period_phrase: str | None = None
    query: str | None = None
    source: str = "rules"
    notes: list[str] = field(default_factory=list)


LIST_RE = re.compile(
    r"(что у меня|что (?:есть|запланировано|намечено)|какие (?:у меня )?(?:дела|задачи|встречи|напоминания|планы)"
    r"|покажи|список|расписани|мои (?:дела|задачи|встречи|планы)|чем я занят)"
)
SEARCH_RE = re.compile(r"(найди|найти|поищи|ищи|где (?:была|был|запись)|была (?:задача|встреча|запись)|какая (?:была )?задача про|вспомни)")
CREATE_RE = re.compile(
    r"(напомни|напомнить|поставь|создай|добавь|запланируй|запиши|назначь|встреч|созвон|совещани|задач|нужно|надо|не забыть)"
)
MEETING_RE = re.compile(r"(встреч|созвон|совещани|планерк|собрани|митинг|переговор)")
REMINDER_RE = re.compile(r"(напомни|напомнить|напоминани|не забыть)")

TRIGGER_RE = re.compile(
    r"^(?:пожалуйста\s+)?(?:напомни(?:те)?(?:\s+мне)?|поставь(?:те)?|создай(?:те)?|добавь(?:те)?|запланируй(?:те)?|запиши(?:те)?|назначь(?:те)?)"
    r"(?:\s+(?:мне|пожалуйста))*\s*",
    re.I,
)
KIND_WORD_RE = re.compile(r"^(?:новую\s+|новое\s+|новая\s+)?(?:задачу|задача|напоминание|встречу|встреча|созвон)\s*", re.I)
FILLER_RE = re.compile(
    r"^(?:что(?:бы)?\s+(?:мне\s+)?(?:нужно|надо)\s+|(?:мне\s+)?(?:нужно|надо)\s+|о том,?\s+что\s+|про\s+|что\s+)",
    re.I,
)
SEARCH_STRIP_RE = re.compile(
    r"^(?:найди|найти|поищи|ищи|вспомни)(?:\s+мне)?\s*|(?:задачу|встречу|запись|напоминание|заметку)\s*|\b(?:про|о|об|насчет|по поводу)\s+"
)


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s{2,}", " ", text).strip(" ,.;:-—!?")


def _cut(text: str, span: tuple[int, int] | None) -> str:
    if not span:
        return text
    return text[: span[0]] + " " + text[span[1]:]


def parse_rules(text: str, today: date) -> Parsed:
    original = re.sub(r"\s+", " ", text.replace("ё", "е").replace("Ё", "Е")).strip()
    t = original.lower()
    if SEARCH_RE.search(t):
        query = _clean_spaces(SEARCH_STRIP_RE.sub(" ", t))
        return Parsed(intent="search_entries", query=query or t)
    if LIST_RE.search(t) and not REMINDER_RE.search(t):
        return Parsed(intent="list_entries", period_phrase=t)
    if not CREATE_RE.search(t) and not dates.find_date(t, today) and not dates.find_time(t):
        return Parsed(intent="unknown")

    kind = "meeting" if MEETING_RE.search(t) else "reminder" if REMINDER_RE.search(t) else "task"
    date_m = dates.find_date(t, today)
    time_m = dates.find_time(t)
    dur_m = dates.find_duration(t)

    rest = original
    for m in sorted([x for x in (date_m, time_m, dur_m) if x], key=lambda x: -x.span[0]):
        rest = _cut(rest, m.span)
    rest = _clean_spaces(rest)
    rest = TRIGGER_RE.sub("", rest)
    rest = KIND_WORD_RE.sub("", rest)
    rest = FILLER_RE.sub("", rest)
    title = _clean_spaces(rest)
    if kind == "meeting" and title and title.lower().startswith("с "):
        title = "встреча " + title
    title = title[:1].upper() + title[1:] if title else None

    return Parsed(
        intent="create_entry",
        kind=kind,
        title=title,
        date_phrase=t[date_m.span[0]:date_m.span[1]] if date_m else None,
        time_phrase=t[time_m.span[0]:time_m.span[1]] if time_m else None,
        duration_minutes=dur_m.value if dur_m else None,
    )


SYSTEM_PROMPT = """Ты разбираешь команды голосового ассистента задач, напоминаний и встреч.
Ответь ТОЛЬКО одним JSON-объектом без пояснений и без Markdown.

Поля:
- "intent": "create_entry" (создать задачу/напоминание/встречу), "list_entries" (показать записи за период), "search_entries" (найти старую запись), "unknown".
- "kind": "task" | "reminder" | "meeting" | null. Встреча, созвон, совещание — meeting. «Напомни» — reminder. Иначе дело — task.
- "title": короткое название записи в именительном падеже с большой буквы, без слов «напомни», «поставь», без даты и времени. Для встречи — «Встреча с …». null, если не названо.
- "date_phrase": слова из запроса, которыми названа дата («завтра», «в пятницу», «12 октября»), или null.
- "time_phrase": слова из запроса, которыми названо время («в 15:00», «в 3 дня», «вечером»), или null.
- "date": та же дата в формате YYYY-MM-DD, посчитанная от сегодняшней, или null.
- "time": то же время в формате HH:MM (24 часа), или null.
- "duration_minutes": длительность встречи в минутах, если названа, иначе null.
- "period_phrase": для list_entries — период словами («сегодня», «завтра», «на этой неделе»), иначе null.
- "query": для search_entries — что ищем, без слов «найди», «задачу про», иначе null.

Не придумывай дату и время, если их нет в запросе.

Примеры (сегодня 2026-10-09, пятница):
«Напомни мне завтра в 10 утра позвонить Ивану» → {"intent":"create_entry","kind":"reminder","title":"Позвонить Ивану","date_phrase":"завтра","time_phrase":"в 10 утра","date":"2026-10-10","time":"10:00","duration_minutes":null,"period_phrase":null,"query":null}
«Поставь встречу с Иваном в понедельник» → {"intent":"create_entry","kind":"meeting","title":"Встреча с Иваном","date_phrase":"в понедельник","time_phrase":null,"date":"2026-10-12","time":null,"duration_minutes":null,"period_phrase":null,"query":null}
«Что у меня на этой неделе?» → {"intent":"list_entries","kind":null,"title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"period_phrase":"на этой неделе","query":null}
«Найди задачу про презентацию» → {"intent":"search_entries","kind":"task","title":null,"date_phrase":null,"time_phrase":null,"date":null,"time":null,"duration_minutes":null,"period_phrase":null,"query":"презентация"}
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
    return Parsed(
        intent=intent,
        kind=kind if kind in KINDS else None,
        title=_str(data.get("title")),
        date_phrase=_str(data.get("date_phrase")),
        time_phrase=_str(data.get("time_phrase")),
        date_iso=_str(data.get("date")),
        time_hhmm=_str(data.get("time")),
        duration_minutes=duration if isinstance(duration, int) and duration > 0 else None,
        period_phrase=_str(data.get("period_phrase")),
        query=_str(data.get("query")),
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
    if parsed.intent == "create_entry":
        parsed.kind = parsed.kind or rules.kind
        parsed.date_phrase = parsed.date_phrase or rules.date_phrase
        parsed.time_phrase = parsed.time_phrase or rules.time_phrase
        parsed.duration_minutes = parsed.duration_minutes or rules.duration_minutes
    if parsed.intent == "search_entries" and not parsed.query:
        parsed.query = rules.query or text
    if parsed.intent == "list_entries" and not parsed.period_phrase:
        parsed.period_phrase = text
    return parsed
