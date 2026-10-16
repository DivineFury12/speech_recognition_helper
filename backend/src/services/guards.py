"""Checks that run before and around the dialogue: yes/no with typos, undo, past dates."""

import re

YES_WORDS = ('да', 'ага', 'угу', 'ок', 'окей', 'давай', 'конечно', 'верно', 'удаляй',
             'ставь', 'подтверждаю', 'точно', 'хорошо', 'ладно', 'yes')
NO_WORDS = ('нет', 'неа', 'не')
YES_RE = re.compile(r'^\s*(?:да|ага|угу|ок(?:ей)?|конечно|давай|верно|подтверждаю|всё равно|все равно|'
                    r'удаляй|ставь|да,?\s+\w+)\b', re.IGNORECASE)
NO_RE = re.compile(r'^\s*(?:нет|не надо|не нужно|не ставь|не удаляй|не переноси|неа)\b', re.IGNORECASE)
KEY_NEIGHBORS = {'д': 'лжщзюб', 'н': 'екгро'}
EN_TO_RU = str.maketrans("qwertyuiop[]asdfghjkl;'zxcvbnm,.`", 'йцукенгшщзхъфывапролджэячсмитьбюё')

UNDO_RE = re.compile(
    r'^\s*(?:(?:отмени|откати|верни)(?:те)?\s+(?:последн\w*(?:\s+действие)?|это|как было|назад|обратно)'
    r'|верни(?:те)?\s+(?:как было|обратно|назад)|отмена последнего(?:\s+действия)?)\s*[.!]*\s*$',
    re.IGNORECASE,
)
UNDO_TEXT = 'Отмени последнее'
DONE_RE = re.compile(r'^\s*(?:я\s+)?(?:уже\s+)?(?:с)?([а-яё]+(?:л|ла|ли))\s+(?P<what>.{2,}?)[.!]*\s*$', re.IGNORECASE)
PAST_WORDS_RE = re.compile(r'\b(?:вчера|позавчера|прошл\w*)\b', re.IGNORECASE)
COMMAND_RE = re.compile(
    r'\b(?:создай|создать|добавь|добавить|поставь|напомни|покажи|найди|удали|перенеси|'
    r'сдвинь|отметь|переименуй|измени)\b', re.IGNORECASE,
)

HELP_TEXT = (
    'Я веду задачи, напоминания и встречи. Скажите, например: «Напомни завтра в 10 позвонить Ивану», '
    '«Что у меня на неделе?», «Перенеси встречу с Иваном на пятницу» или «Отметь задачу про отчёт выполненной».'
)
HELP_SUGGESTIONS = ['Что у меня сегодня?', 'Поставь встречу завтра в 10', 'Напомни купить молоко в 19:00']
CHOICE_WORDS = ('Первую', 'Вторую', 'Третью', 'Четвёртую', 'Пятую')


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def yes_no(text: str) -> str | None:
    """«да», «жа», «lf» (English layout), «ytn» → yes/no; anything else → None."""
    if NO_RE.match(text):
        return 'no'
    if YES_RE.match(text):
        return 'yes'
    words = re.findall(r"[a-zа-яё\[\];',.`]+", text.casefold())
    if not words or len(words) > 2:
        return None
    word = words[0]
    if re.fullmatch(r"[a-z\[\];',.`]+", word):
        word = word.translate(EN_TO_RU)
    word = word.replace('ё', 'е')
    for answer, vocab in (('yes', YES_WORDS), ('no', NO_WORDS)):
        if word in vocab:
            return answer
    if len(word) == 2:
        for answer, target in (('yes', 'да'), ('no', 'не')):
            if word[1] == target[1] and word[0] in KEY_NEIGHBORS.get(target[0], ''):
                return answer
        return None
    if len(word) <= 7:
        best = min(((_edit_distance(word, w), answer) for answer, vocab in (('yes', YES_WORDS), ('no', NO_WORDS))
                    for w in vocab if len(w) >= 3), default=(9, None))
        if best[0] <= 1:
            return best[1]
    return None


def is_short_unclear(text: str) -> bool:
    return len(text.split()) <= 3 and not COMMAND_RE.search(text)


_AMOUNTS = {'пол': 0.5, 'один': 1, 'одну': 1, 'полтора': 1.5, 'полторы': 1.5, 'два': 2, 'две': 2, 'пару': 2,
            'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'десять': 10, 'пятнадцать': 15, 'двадцать': 20,
            'тридцать': 30, 'сорок': 40, 'сорок пять': 45}
SHIFT_RE = re.compile(
    r'\bна\s+(?P<amount>\d{1,3}|пол|полтора|полторы|одну|один|два|две|пару|три|четыре|пять|шесть|десять|'
    r'пятнадцать|двадцать|тридцать|сорок(?:\s+пять)?)?\s*(?P<unit>час\w*|минут\w*|мин\b|дн\w*|день|сутк\w*|'
    r'недел\w*)(?:\s+(?P<direction>позже|позднее|вперед|вперёд|раньше|пораньше|назад))?',
    re.IGNORECASE,
)
SHIFT_VERB_RE = re.compile(r'\b(?:перенес\w*|перенеси|сдвин\w*|передвин\w*|отлож\w*)\b', re.IGNORECASE)


def find_shift(text: str) -> int | None:
    """«на час позже» → +60, «на 30 минут раньше» → -30, «перенеси на 3 дня» → +4320 minutes."""
    value = text.casefold().replace('ё', 'е')
    if 'полчаса' in value:
        value = value.replace('полчаса', 'пол часа')
    match = SHIFT_RE.search(value)
    if match is None:
        return None
    unit = match.group('unit')
    direction = match.group('direction')
    is_day = unit.startswith(('дн', 'ден', 'сут', 'недел'))
    if direction is None and not (is_day and SHIFT_VERB_RE.search(value)):
        return None
    if direction is None and match.group('amount') is None:
        return None
    raw = match.group('amount')
    amount = 1 if raw is None else float(raw) if raw.isdigit() else _AMOUNTS.get(raw, 1)
    minutes = amount * (60 if unit.startswith('час') else 1 if unit.startswith('мин') else
                        7 * 1440 if unit.startswith('недел') else 1440)
    sign = -1 if direction in ('раньше', 'пораньше', 'назад') else 1
    return int(sign * minutes)


_SCHEDULE_NUMBER = re.compile(
    r'(?:\b(?:в|во|к|на|до|с|со|через|за|по)\s+(?:\d+\s*(?:-|и|или)\s*)?$)'
)
_SCHEDULE_AFTER = re.compile(
    r'^\s*(?::\d|час|ч\b|мин|дн|день|недел|месяц|числ|утра|вечера|дня|ночи|год|'
    r'январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр|[./]\d)',
    re.IGNORECASE,
)


def restore_numbers(title: str | None, text: str) -> str | None:
    """GigaChat sometimes drops numbers from a title («Встреча с Иваном 159516»); put them back.

    Numbers that belong to a date, time or duration («в 15:00», «через 2 дня», «5 октября») are skipped.
    """
    if not title:
        return title
    missing = []
    for match in re.finditer(r'(?<![\d:.,/-])\d+(?![\d:.,/-]?\d)', text):
        before, after = text[:match.start()].casefold(), text[match.end():]
        if _SCHEDULE_NUMBER.search(before) or _SCHEDULE_AFTER.match(after):
            continue
        if match.group() not in re.findall(r'\d+', title):
            missing.append(match.group())
    return f"{title} {' '.join(missing)}" if missing else title


_LEAD_RE = re.compile(
    r'\bза\s+(?P<amount>\d{1,3}|полчаса|пол|полтора|полторы|один|одну|два|две|три|пять|десять|пятнадцать|'
    r'двадцать|тридцать|сорок)?\s*(?P<unit>минут\w*|мин\b|час\w*|ч\b|дн\w*|день|сутк\w*)?(?:\s+до\b)?',
    re.IGNORECASE,
)
REMIND_FOR_RE = re.compile(
    r'^\s*напомни\w*(?:\s+мне)?\s+за\s+.+?\s+до\s+(?P<what>.+?)\s*[.!]*$', re.IGNORECASE,
)
ONLY_THIS_RE = re.compile(r'\bтолько\b', re.IGNORECASE)
STOP_SERIES_RE = re.compile(
    r'^\s*(?:больше\s+не\s+повторя\w*|останови\s+повтор\w*|заверши\s+(?:серию|повтор\w*)|'
    r'прекрати\s+повтор\w*|отмени\s+повтор\w*)\s+(?P<what>.+?)\s*[.!]*$', re.IGNORECASE,
)
UNTIL_RE = re.compile(
    r'\bдо\s+(?P<when>конца\s+(?:недели|месяца|года)|\d{1,2}\s+[а-я]+|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)',
    re.IGNORECASE,
)


def find_lead(text: str) -> int | None:
    """«напомни за 15 минут» → 15, «за час до встречи» → 60, «за день» → 1440."""
    value = text.casefold().replace('ё', 'е')
    for match in _LEAD_RE.finditer(value):
        amount, unit = match.group('amount'), match.group('unit')
        if amount is None and unit is None:
            continue
        if amount in ('полчаса',):
            return 30
        if unit is None:
            continue
        count = 1 if amount is None else float(amount) if amount.isdigit() else _AMOUNTS.get(amount, 1)
        if amount in ('пол',):
            count = 0.5
        factor = 60 if unit.startswith(('час', 'ч')) else 1 if unit.startswith('мин') else 1440
        return int(count * factor)
    return None


def find_until(text: str, today) -> object | None:
    """«каждый понедельник до конца месяца» → the last day of the series."""
    import calendar
    import datetime

    from src.services.temporal import parse_date

    match = UNTIL_RE.search(text.casefold().replace('ё', 'е'))
    if match is None:
        return None
    when = match.group('when')
    if when.startswith('конца недели'):
        return today + datetime.timedelta(days=6 - today.weekday())
    if when.startswith('конца месяца'):
        return today.replace(day=calendar.monthrange(today.year, today.month)[1])
    if when.startswith('конца года'):
        return today.replace(month=12, day=31)
    return parse_date(when, today)
