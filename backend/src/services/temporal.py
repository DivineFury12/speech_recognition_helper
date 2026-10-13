"""Verify date and time against words actually present in a user request."""

import datetime
import re

from src.schemas.assistant_command import AssistantCommand


WEEKDAYS = {
    'понедельник': 0, 'вторник': 1, 'сред': 2, 'четверг': 3,
    'пятниц': 4, 'суббот': 5, 'воскресен': 6,
}
MONTHS = {
    'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'мая': 5,
    'июн': 6, 'июл': 7, 'август': 8, 'сентябр': 9, 'октябр': 10,
    'ноябр': 11, 'декабр': 12,
}

HOUR_WORDS = {
    'ноль': 0, 'один': 1, 'одну': 1, 'два': 2, 'две': 2,
    'три': 3, 'четыре': 4, 'четырех': 4, 'пять': 5, 'пяти': 5,
    'шесть': 6, 'шести': 6, 'семь': 7, 'семи': 7,
    'восемь': 8, 'восьми': 8, 'девять': 9, 'девяти': 9,
    'десять': 10, 'десяти': 10, 'одиннадцать': 11, 'двенадцать': 12,
    'тринадцать': 13, 'четырнадцать': 14, 'пятнадцать': 15,
    'шестнадцать': 16, 'семнадцать': 17, 'восемнадцать': 18,
    'девятнадцать': 19, 'двадцать': 20,
}
HOUR_PATTERN = '|'.join(sorted((re.escape(word) for word in HOUR_WORDS), key=len, reverse=True))
HOUR_PATTERN = rf'(?:\d{{1,2}}|{HOUR_PATTERN})'
COUNT_PATTERN = rf'(?:\d{{1,4}}|{HOUR_PATTERN})'
DATE_UNIT_PATTERN = r'(?P<unit>недел\w*|день|дня|дней|сутк\w*|суток)'
MONTH_PATTERN = '|'.join(rf'{stem}\w*' for stem in MONTHS)

TITLE_STOPWORDS = {
    'создай', 'создать', 'добавь', 'добавить', 'поставь', 'поставить',
    'напомни', 'напомнить', 'напоминай', 'напоминание', 'напоминания',
    'задача', 'задачу', 'задачи', 'встреча', 'встречу', 'встречи',
    'мне', 'меня', 'мою', 'мой', 'мои', 'про', 'для', 'что', 'чтобы',
    'пожалуйста', 'нужно', 'надо', 'сегодня', 'завтра', 'послезавтра',
    'следующую', 'следующей', 'следующий', 'неделю', 'неделе',
    'недели', 'недель', 'через', 'текущая', 'текущую', 'текущей',
    'дата', 'дату', 'даты', 'дате', 'день', 'дней', 'сутки', 'суток',
    'плюс',
    'утра', 'дня', 'вечера', 'ночи', 'час', 'часа', 'часов',
    'минут', 'минуты', 'года', 'год', 'числа',
}


def _in_text(phrase: str | None, text: str) -> bool:
    if not phrase:
        return False
    return ' '.join(phrase.casefold().split()) in ' '.join(text.casefold().split())


def _calendar_date(day: int, month: int, year: int | None, today: datetime.date) -> datetime.date | None:
    try:
        if year is not None:
            if year < 100:
                year += 2000
            return datetime.date(year, month, day)
        result = datetime.date(today.year, month, day)
        if result < today:
            result = datetime.date(today.year + 1, month, day)
        return result
    except ValueError:
        return None


def _relative_date(today: datetime.date, count: str | None, unit: str) -> datetime.date | None:
    amount = 1 if count is None else int(count) if count.isdigit() else HOUR_WORDS[count]
    days = amount * (7 if unit.startswith('недел') else 1)
    try:
        return today + datetime.timedelta(days=days)
    except (OverflowError, ValueError):
        return None


def parse_date(phrase: str, today: datetime.date) -> datetime.date | None:
    value = phrase.casefold().replace('ё', 'е')
    addition = re.search(
        rf'\b(?:сегодня|текущ\w*\s+дат\w*)\s*(?:\+|плюс)\s*'
        rf'(?P<count>{COUNT_PATTERN})\s+{DATE_UNIT_PATTERN}\b',
        value,
    )
    if addition:
        return _relative_date(today, addition.group('count'), addition.group('unit'))
    relative = re.search(
        rf'\bчерез\s+(?:(?P<count>{COUNT_PATTERN})\s+)?{DATE_UNIT_PATTERN}\b',
        value,
    )
    if relative:
        return _relative_date(today, relative.group('count'), relative.group('unit'))
    if re.search(r'\bпослезавтра\b', value):
        return today + datetime.timedelta(days=2)
    if re.search(r'\bзавтра\b', value):
        return today + datetime.timedelta(days=1)
    if re.search(r'\bсегодня\b', value):
        return today

    iso = re.search(r'(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)', value)
    if iso:
        return _calendar_date(
            int(iso.group(3)), int(iso.group(2)), int(iso.group(1)), today,
        )

    numeric = re.search(r'(?<!\d)(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?(?!\d)', value)
    if numeric:
        return _calendar_date(
            int(numeric.group(1)), int(numeric.group(2)),
            int(numeric.group(3)) if numeric.group(3) else None, today,
        )

    for stem, month in MONTHS.items():
        month_match = re.search(rf'(?<!\d)(\d{{1,2}})\s+{stem}\w*', value)
        if month_match:
            return _calendar_date(int(month_match.group(1)), month, None, today)

    for stem, weekday in WEEKDAYS.items():
        if stem in value:
            if 'следующ' in value:
                start_of_next_week = today + datetime.timedelta(days=7 - today.weekday())
                return start_of_next_week + datetime.timedelta(days=weekday)
            days_ahead = (weekday - today.weekday()) % 7
            return today + datetime.timedelta(days=days_ahead)
    return None


def parse_time(phrase: str) -> str | None:
    value = phrase.casefold().replace('ё', 'е')
    if 'полноч' in value:
        return '00:00'
    if 'полден' in value or 'полдн' in value:
        return '12:00'
    clock = re.search(r'(?<!\d)(\d{1,2}):([0-5]\d)(?!\d)', value)
    if clock:
        hour = int(clock.group(1))
        if hour < 24:
            return f'{hour:02d}:{int(clock.group(2)):02d}'

    spoken_matches = re.finditer(
        rf'\b(?:(?:в|на|к)\s+)?(?P<hour>{HOUR_PATTERN})'
        r'(?:\s+час(?:а|ов)?)?\s+(?P<period>утра|дня|вечера|ночи)\b',
        value,
    )
    for spoken in spoken_matches:
        prefix = value[:spoken.start()]
        if spoken.group('period') == 'дня' and re.search(r'\bчерез\s*$', prefix):
            continue
        hour_token = spoken.group('hour')
        hour = int(hour_token) if hour_token.isdigit() else HOUR_WORDS[hour_token]
        if hour < 24:
            if spoken.group('period') in {'дня', 'вечера'} and hour < 12:
                hour += 12
            elif spoken.group('period') in {'утра', 'ночи'} and hour == 12:
                hour = 0
            return f'{hour:02d}:00'

    simple = re.search(
        rf'\b(?:в|к)\s+(?P<hour>{HOUR_PATTERN})\b'
        rf'(?!\s+(?:{MONTH_PATTERN}))', value,
    )
    if simple:
        token = simple.group('hour')
        hour = int(token) if token.isdigit() else HOUR_WORDS[token]
        if hour < 24:
            return f'{hour:02d}:00'
    return None


def _title_has_source(title: str | None, text: str) -> bool:
    if not title:
        return False

    def content_words(value: str) -> set[str]:
        words = re.findall(r'[а-яёa-z]+', value.casefold().replace('ё', 'е'))
        return {
            word for word in words
            if len(word) >= 3 and word not in TITLE_STOPWORDS
            and word not in HOUR_WORDS
            and not any(word.startswith(stem) for stem in (*MONTHS, *WEEKDAYS))
        }

    return bool(content_words(title) & content_words(text))


def verify_creation_fields(
    command: AssistantCommand, text: str, today: datetime.date,
) -> AssistantCommand:
    """Ignore invented title/date/time while retaining explicit user details."""
    date = parse_date(command.date_phrase, today) if _in_text(command.date_phrase, text) else None
    if date is None:
        date = parse_date(text, today)
    time = parse_time(command.time_phrase) if _in_text(command.time_phrase, text) else None
    if time is None:
        time = parse_time(text)
    return command.model_copy(update={
        'title': command.title if _title_has_source(command.title, text) else None,
        'date': date, 'date_phrase': command.date_phrase if date is not None else None,
        'time': time, 'time_phrase': command.time_phrase if time is not None else None,
    })
