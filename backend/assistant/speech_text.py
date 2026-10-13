import re

from num2words import num2words

WEEKDAYS = {
    "пн": ("понедельник", "понедельника", "понедельник"),
    "вт": ("вторник", "вторника", "вторник"),
    "ср": ("среда", "среды", "среду"),
    "чт": ("четверг", "четверга", "четверг"),
    "пт": ("пятница", "пятницы", "пятницу"),
    "сб": ("суббота", "субботы", "субботу"),
    "вс": ("воскресенье", "воскресенья", "воскресенье"),
}
MONTHS = r"(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
FEMININE = r"(запис\w*|встреч\w*|задач\w*|минут\w*)"
TIME = r"([01]?\d|2[0-3]):([0-5]\d)"


def _num(n: int, **kw) -> str:
    return num2words(n, lang="ru", **kw)


def _time(h: str, m: str, case: str = "nominative") -> str:
    hours = _num(int(h), case=case)
    if m == "00":
        return f"{hours} ноль-ноль"
    if m.startswith("0"):
        return f"{hours} ноль {_num(int(m))}"
    return f"{hours} {_num(int(m))}"


def _weekday(m: re.Match) -> str:
    prep = m.group(1)
    form = WEEKDAYS[m.group(2)][{"с": 1, "в": 2}.get((prep or "").lower(), 0)]
    return (f"{prep} " if prep else "") + form + ","


def _date(m: re.Match) -> str:
    prep = m.group(1)
    case = {"case": "accusative", "gender": "n"} if prep else {"case": "genitive"}
    return (f"{prep} " if prep else "") + f"{_num(int(m.group(2)), to='ordinal', **case)} {m.group(3)}"


def for_speech(text: str) -> str:
    t = text.replace("•", " ").replace("✓", "").replace("«", "").replace("»", "")
    t = re.sub(r"\s*—\s*", ", ", t)
    t = re.sub(r"\n+", ". ", t)
    t = re.sub(
        rf"(?:\bв\s+)?\b{TIME}\s*[–-]\s*{TIME}\b",
        lambda m: f"с {_time(m.group(1), m.group(2), 'genitive')} до {_time(m.group(3), m.group(4), 'genitive')}",
        t,
    )
    t = re.sub(rf"\b{TIME}\b", lambda m: _time(m.group(1), m.group(2)), t)
    t = re.sub(r"(?:\b(с|в|по)\s+)?\b(пн|вт|ср|чт|пт|сб|вс),", _weekday, t, flags=re.I)
    t = re.sub(rf"(?:\b(на|по)\s+)?\b(\d{{1,2}})\s+{MONTHS}", _date, t, flags=re.I)
    t = re.sub(r"\b(\d+)\s*мин\b\.?", lambda m: f"{_num(int(m.group(1)))} минут", t)
    t = re.sub(rf"\b(\d+)\s+{FEMININE}", lambda m: f"{_num(int(m.group(1)), gender='f')} {m.group(2)}", t)
    t = re.sub(r"\b\d+\b", lambda m: _num(int(m.group(0))), t)
    t = re.sub(r"\(\s*", ", ", t)
    t = re.sub(r"\s*\)", "", t)
    t = re.sub(r"\s+([,.!?:])", r"\1", t)
    t = re.sub(r":\s*\.", ":", t)
    t = re.sub(r"([,.:])\s*([,.])+", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip(" ,")
