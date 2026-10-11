"""Verify date and time against words actually present in a user request."""

import datetime
import re

from src.schemas.assistant_command import AssistantCommand


WEEKDAYS = {
    'понедельник': 0, 'вторник': 1, 'сред': 2, 'четверг': 3,
    'пятниц': 4, 'суббот': 5, 'воскресен': 6,
}
MONTHS = {
    'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'ма': 5,
    'июн': 6, 'июл': 7, 'август': 8, 'сентябр': 9, 'октябр': 10,
    'ноябр': 11, 'декабр': 12,
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


def parse_date(phrase: str, today: datetime.date) -> datetime.date | None:
    value = phrase.casefold().replace('ё', 'е')
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

    relative = re.search(r'через\s+(\d+)\s+(?:день|дня|дней|сутки|суток)', value)
    if relative:
        return today + datetime.timedelta(days=int(relative.group(1)))

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
    match = re.search(r'(?<!\d)(\d{1,2})(?::(\d{2}))?(?!\d)', value)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if hour > 23 or minute > 59:
        return None
    if 'вечер' in value or 'дня' in value:
        if hour < 12:
            hour += 12
    elif 'ночи' in value and hour == 12:
        hour = 0
    return f'{hour:02d}:{minute:02d}'


def verify_creation_time(
    command: AssistantCommand, text: str, today: datetime.date,
) -> AssistantCommand:
    """Ignore model dates and times that lack verifiable source words."""
    date = parse_date(command.date_phrase, today) if _in_text(command.date_phrase, text) else None
    if date is None:
        date = parse_date(text, today)
    time = parse_time(command.time_phrase) if _in_text(command.time_phrase, text) else None
    return command.model_copy(update={
        'date': date, 'date_phrase': command.date_phrase if date is not None else None,
        'time': time, 'time_phrase': command.time_phrase if time is not None else None,
    })
