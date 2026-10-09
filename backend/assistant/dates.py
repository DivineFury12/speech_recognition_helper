import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTH_RE = r"(январ[яь]|феврал[яь]|марта?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|августа?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])"

MONTHS_GENITIVE = [
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресень": 6,
}
WEEKDAY_RE = r"(понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресенье)"

NUMBER_WORDS = {
    "один": 1, "одну": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "одиннадцать": 11, "двенадцать": 12,
}
NUM = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + ")"


def _num(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _month(token: str) -> int:
    for stem, value in MONTHS.items():
        if token.startswith(stem):
            return value
    raise ValueError(token)


def _weekday(token: str) -> int:
    for stem, value in WEEKDAYS.items():
        if token.startswith(stem):
            return value
    raise ValueError(token)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


@dataclass
class Match:
    value: object
    span: tuple[int, int]


def find_date(text: str, today: date) -> Match | None:
    t = normalize(text)
    rules: list[tuple[str, object]] = [
        (r"\bпослезавтра\b", lambda m: today + timedelta(days=2)),
        (r"\bзавтра\b", lambda m: today + timedelta(days=1)),
        (r"\bсегодня\b", lambda m: today),
        (r"\bчерез\s+(?:" + NUM + r"\s+)?(день|дня|дней)\b",
         lambda m: today + timedelta(days=_num(m.group(1)) if m.group(1) else 1)),
        (r"\bчерез\s+(?:" + NUM + r"\s+)?(недел[юиь]|недель)\b",
         lambda m: today + timedelta(weeks=_num(m.group(1)) if m.group(1) else 1)),
        (r"\b(\d{4})-(\d{2})-(\d{2})\b",
         lambda m: _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (r"(?<!в )(?<!к )\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b(?!\s*(?:час|мин))", _dotted),
        (r"\b(\d{1,2})(?:-?го)?\s+" + MONTH_RE + r"(?:\s+(\d{4}))?", _day_month),
        (r"\b(?:(?:в|во|на)\s+)?(?:(следующ\w*|эт[уио]\w*)\s+)?" + WEEKDAY_RE + r"\b", _weekday_rule),
    ]
    for pattern, build in rules:
        m = re.search(pattern, t)
        if not m:
            continue
        if build is _dotted or build is _day_month or build is _weekday_rule:
            value = build(m, today)
        else:
            value = build(m)
        if value is not None:
            return Match(value, m.span())
    return None


def _dotted(m: re.Match, today: date) -> date | None:
    day, month = int(m.group(1)), int(m.group(2))
    if m.group(3):
        year = int(m.group(3))
        year += 2000 if year < 100 else 0
        return _safe_date(year, month, day)
    return _future(today, month, day)


def _day_month(m: re.Match, today: date) -> date | None:
    day, month = int(m.group(1)), _month(m.group(2))
    if m.group(3):
        return _safe_date(int(m.group(3)), month, day)
    return _future(today, month, day)


def _future(today: date, month: int, day: int) -> date | None:
    candidate = _safe_date(today.year, month, day)
    if candidate and candidate < today:
        candidate = _safe_date(today.year + 1, month, day)
    return candidate


def _weekday_rule(m: re.Match, today: date) -> date:
    target = _weekday(m.group(2))
    delta = (target - today.weekday()) % 7
    if m.group(1) and m.group(1).startswith("следующ"):
        delta = delta + 7 if delta else 7
    return today + timedelta(days=delta)


def find_time(text: str) -> Match | None:
    t = normalize(text)
    m = re.search(r"\bв\s+полдень\b", t)
    if m:
        return Match(time(12, 0), m.span())
    m = re.search(r"\bв\s+полночь\b", t)
    if m:
        return Match(time(0, 0), m.span())
    m = re.search(r"\b(?:в|к|на|с)?\s*([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m:
        return Match(time(int(m.group(1)), int(m.group(2))), m.span())
    m = re.search(r"\b(?:в|к)\s+([01]?\d|2[0-3])\.([0-5]\d)\b", t)
    if m:
        return Match(time(int(m.group(1)), int(m.group(2))), m.span())
    m = re.search(
        r"\b(?:в|к)\s+" + NUM + r"(?:\s*(?:час(?:а|ов)?|ч))?(?:\s+(утра|дня|вечера|ночи))?\b(?!\s*(?:" + MONTH_RE + r"|дн|недел|минут))",
        t,
    )
    if m:
        hour = _num(m.group(1))
        if hour > 23:
            return None
        part = m.group(2)
        if part in ("дня", "вечера") and hour < 12:
            hour += 12
        elif part == "ночи" and hour == 12:
            hour = 0
        elif part is None and 1 <= hour <= 6:
            hour += 12
        return Match(time(hour, 0), m.span())
    m = re.search(r"\b(утром|днем|вечером)\b", t)
    if m:
        return Match({"утром": time(9, 0), "днем": time(13, 0), "вечером": time(19, 0)}[m.group(1)], m.span())
    return None


RELATIVE_TIME_RE = r"\bчерез\s+(?:(?:" + NUM + r"\s+)?(?:час|часа|часов|минут[уы]?)|полчаса)\b"


def find_relative_time_span(text: str) -> Match | None:
    m = re.search(RELATIVE_TIME_RE, normalize(text))
    return Match(m.group(0), m.span()) if m else None


def find_relative_datetime(text: str, now: datetime) -> Match | None:
    t = normalize(text)
    m = re.search(r"\bчерез\s+(?:" + NUM + r"\s+)?(час|часа|часов|минут[уы]?)\b", t)
    if m:
        n = _num(m.group(1)) if m.group(1) else 1
        delta = timedelta(hours=n) if m.group(2).startswith("час") else timedelta(minutes=n)
        return Match(now + delta, m.span())
    m = re.search(r"\bчерез\s+полчаса\b", t)
    if m:
        return Match(now + timedelta(minutes=30), m.span())
    return None


def find_duration(text: str) -> Match | None:
    t = normalize(text)
    patterns = [
        (r"\bна\s+полчаса\b", lambda m: 30),
        (r"\bна\s+полтора\s+часа\b", lambda m: 90),
        (r"\bна\s+(?:" + NUM + r"\s+)?(час|часа|часов)\b", lambda m: 60 * (_num(m.group(1)) if m.group(1) else 1)),
        (r"\bна\s+(\d{1,3})\s+минут\w*\b", lambda m: int(m.group(1))),
    ]
    for pattern, build in patterns:
        m = re.search(pattern, t)
        if m:
            return Match(build(m), m.span())
    return None


def parse_date(phrase: str | None, today: date) -> date | None:
    if not phrase:
        return None
    m = find_date(phrase, today)
    return m.value if m else None


def parse_time(phrase: str | None) -> time | None:
    if not phrase:
        return None
    m = find_time(phrase)
    if m:
        return m.value
    m = re.fullmatch(r"\s*([01]?\d|2[0-3])\s*", phrase)
    return time(int(m.group(1)), 0) if m else None


def parse_period(phrase: str | None, today: date) -> tuple[date, date]:
    t = normalize(phrase or "")
    if re.search(r"\bследующ\w*\s+недел", t):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if re.search(r"\b(эт\w+\s+недел|недел[еюи]|неделя)\b", t):
        if re.search(r"\bэт\w+\s+недел", t):
            return today, today + timedelta(days=6 - today.weekday())
        return today, today + timedelta(days=6)
    if re.search(r"\bмесяц", t):
        return today, today + timedelta(days=30)
    if re.search(r"\bвыходн", t):
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        return saturday, saturday + timedelta(days=1)
    m = find_date(t, today)
    if m:
        return m.value, m.value
    return today, today


def format_date(d: date) -> str:
    return f"{WEEKDAYS_SHORT[d.weekday()]}, {d.day} {MONTHS_GENITIVE[d.month]}"


def format_when(d: date | None, t: time | None, today: date) -> str:
    parts: list[str] = []
    if d:
        if d == today:
            parts.append("сегодня")
        elif d == today + timedelta(days=1):
            parts.append("завтра")
        else:
            parts.append(format_date(d))
    if t:
        parts.append(f"в {t.strftime('%H:%M')}")
    return " ".join(parts) if parts else "без срока"
