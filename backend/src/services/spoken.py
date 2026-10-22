"""Russian ways to say a time and other phrasing the model tends to miss.

«полвосьмого вечера» → 19:30, «без пятнадцати девять» → 08:45, «четверть третьего дня» → 14:15,
«через полтора часа» → +90 minutes, «в 9, нет, лучше в 10» → «в 10», «завтр» → «завтра».
"""

import calendar
import re

ORDINAL_GEN = {
    'первого': 1, 'второго': 2, 'третьего': 3, 'четвертого': 4, 'пятого': 5, 'шестого': 6,
    'седьмого': 7, 'восьмого': 8, 'девятого': 9, 'десятого': 10, 'одиннадцатого': 11, 'двенадцатого': 12,
}
CARDINAL = {
    'час': 1, 'один': 1, 'одного': 1, 'два': 2, 'двух': 2, 'три': 3, 'трех': 3, 'четыре': 4, 'четырех': 4,
    'пять': 5, 'пяти': 5, 'шесть': 6, 'шести': 6, 'семь': 7, 'семи': 7, 'восемь': 8, 'восьми': 8,
    'девять': 9, 'девяти': 9, 'десять': 10, 'десяти': 10, 'одиннадцать': 11, 'одиннадцати': 11,
    'двенадцать': 12, 'двенадцати': 12,
}
MINUTES_GEN = {'пяти': 5, 'десяти': 10, 'пятнадцати': 15, 'двадцати': 20, 'двадцати пяти': 25}
_ORD = '|'.join(ORDINAL_GEN)
_CARD = '|'.join(sorted(CARDINAL, key=len, reverse=True))
_MIN = '|'.join(sorted(MINUTES_GEN, key=len, reverse=True))


def _with_daypart(hour: int, minute: int, rest: str) -> str:
    part = re.match(r'\s*(утра|дня|вечера|ночи)\b', rest)
    if part and part.group(1) in ('дня', 'вечера') and hour < 12:
        hour += 12
    elif part and part.group(1) == 'ночи' and hour == 12:
        hour = 0
    elif part is None and 1 <= hour <= 6:
        # «полвосьмого» without a daypart is ambiguous only for early hours; keep the morning reading
        # for 7–11 and the afternoon one for 1–6 (people rarely mean 03:30).
        hour += 12
    return f'{hour % 24:02d}:{minute:02d}'


def spoken_time(text: str) -> str | None:
    value = text.casefold().replace('ё', 'е')
    half = re.search(rf'\b(?:в\s+)?(?:пол|половин[еау]\s+|пол-)\s*({_ORD})\b', value)
    if half:
        hour = ORDINAL_GEN[half.group(1)] - 1 or 12
        return _with_daypart(hour, 30, value[half.end():])
    quarter_past = re.search(rf'\b(?:в\s+)?четверть\s+({_ORD})\b', value)
    if quarter_past:
        hour = ORDINAL_GEN[quarter_past.group(1)] - 1 or 12
        return _with_daypart(hour, 15, value[quarter_past.end():])
    to = re.search(rf'\b(?:в\s+)?без\s+(четверти|{_MIN}|\d{{1,2}})(?:\s+минут\w*)?\s+({_CARD}|\d{{1,2}})\b', value)
    if to:
        token = to.group(1)
        minutes = 15 if token == 'четверти' else MINUTES_GEN.get(token) or int(token)
        target = to.group(2)
        hour = (CARDINAL[target] if not target.isdigit() else int(target)) - 1
        hour = hour if hour > 0 else 12
        return _with_daypart(hour, 60 - minutes, value[to.end():])
    return None


RELATIVE_RE = re.compile(
    r'\bчерез\s+(?:(?P<hours>\d{1,2}|один|одну|два|две|три|четыре|пять|шесть)\s+час\w*'
    r'(?:\s+(?:и\s+)?(?P<hmin>\d{1,2})\s+минут\w*)?'
    r'|(?P<onehalf>полтора|полторы)\s+час\w*'
    r'|(?P<half>полчаса)'
    r'|(?P<hour>час)\b'
    r'|(?P<minutes>\d{1,3}|пять|десять|пятнадцать|двадцать|тридцать|сорок|сорок\s+пять)\s+минут\w*)',
    re.IGNORECASE,
)
_WORD_NUM = {'один': 1, 'одну': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6,
             'десять': 10, 'пятнадцать': 15, 'двадцать': 20, 'тридцать': 30, 'сорок': 40, 'сорок пять': 45}


def relative_minutes(text: str) -> int | None:
    """«через 2 часа 30 минут» → 150, «через полтора часа» → 90, «через 15 минут» → 15."""
    match = RELATIVE_RE.search(text.casefold())
    if match is None:
        return None
    number = lambda token: int(token) if token.isdigit() else _WORD_NUM[token]  # noqa: E731
    if match.group('hours'):
        return number(match.group('hours')) * 60 + (int(match.group('hmin')) if match.group('hmin') else 0)
    if match.group('onehalf'):
        return 90
    if match.group('half'):
        return 30
    if match.group('hour'):
        return 60
    return number(' '.join(match.group('minutes').split()))


CORRECTION_RE = re.compile(
    r'(?P<old>\b(?:в|на|к)\s+\d{1,2}(?::\d{2})?(?:\s+(?:утра|дня|вечера|ночи))?)\s*,?\s*'
    r'(?:ой\s*,?\s*)?(?:нет|не|точнее|вернее|хотя|а\s+лучше)\s*,?\s*(?:лучше\s+|давай\s+)?'
    r'(?P<new>(?:в|на|к)\s+\d{1,2}(?::\d{2})?(?:\s+(?:утра|дня|вечера|ночи))?)',
    re.IGNORECASE,
)
TYPOS = [
    (r'\bзавтр\b', 'завтра'), (r'\bзавтро\b', 'завтра'), (r'\bпослезавтр\b', 'послезавтра'),
    (r'\bсегодн\b|\bседня\b|\bсёдня\b|\bсевоня\b', 'сегодня'),
    (r'\bпастав\w*\b|\bпостав\b', 'поставь'), (r'\bнапомн\b|\bнапомин\b', 'напомни'),
    (r'\bдобав\b', 'добавь'), (r'\bсоздай\w*й\b', 'создай'),
]
NOT_THIS_RE = re.compile(r'^\s*(?:нет\s*,?\s*)?не\s+(?:эту|этот|это|эта|ту|тот|та)\s*,?\s*а\s+', re.IGNORECASE)
CREATE_THIS_RE = re.compile(r'(?i)\b(поставь|создай|запланируй|добавь|напомни\s+про)\s+(?:эт[уоа]|этот)\s+')
INTERJECTIONS_RE = re.compile(r'\b(?:блин|блять|бля|черт|чёрт|короче|типа|слушай|кстати)\b,?\s*', re.IGNORECASE)


def normalize_request(text: str) -> str:
    """Spell out what the user meant before anything else reads the text."""
    value = CORRECTION_RE.sub(lambda m: m.group('new'), text)
    value = NOT_THIS_RE.sub('', value)
    value = CREATE_THIS_RE.sub(lambda m: m.group(1) + ' ', value)
    for pattern, word in TYPOS:
        value = re.sub(pattern, word, value, flags=re.IGNORECASE)
    value = INTERJECTIONS_RE.sub('', value)
    return ' '.join(value.split()) or text


MONTHS_GEN = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7,
              'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}


MONTHS_PREP = {1: 'январе', 2: 'феврале', 3: 'марте', 4: 'апреле', 5: 'мае', 6: 'июне', 7: 'июле',
               8: 'августе', 9: 'сентябре', 10: 'октябре', 11: 'ноябре', 12: 'декабре'}


def impossible_date(text: str, year: int) -> str | None:
    """«31 февраля», «32 марта»: a date that does not exist. Returns a short explanation."""
    match = re.search(rf'\b(\d{{1,2}})\s+({"|".join(MONTHS_GEN)})\b', text.casefold())
    if match is None:
        return None
    day, month = int(match.group(1)), MONTHS_GEN[match.group(2)]
    if 1 <= day <= calendar.monthrange(year, month)[1]:
        return None
    return f'В {MONTHS_PREP[month]} нет {day}-го числа'
