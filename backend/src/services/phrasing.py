"""Answers in plain Russian: «завтра в 15:00» instead of «2026-10-17 15:00»."""

import datetime

from src.schemas.assistant import Entry, EntryKind
from src.services.recurrence import label

MONTHS_GEN = ('января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля',
              'августа', 'сентября', 'октября', 'ноября', 'декабря')
WEEKDAYS_SHORT = ('пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс')
KIND_ACC = {EntryKind.task: 'задачу', EntryKind.reminder: 'напоминание', EntryKind.meeting: 'встречу'}
KIND_NOM = {EntryKind.task: 'задача', EntryKind.reminder: 'напоминание', EntryKind.meeting: 'встреча'}


def capitalize(title: str) -> str:
    title = ' '.join(title.split())
    return title[:1].upper() + title[1:]


def format_date(day: datetime.date, today: datetime.date) -> str:
    delta = (day - today).days
    named = {0: 'сегодня', 1: 'завтра', 2: 'послезавтра', -1: 'вчера', -2: 'позавчера'}
    if delta in named:
        return named[delta]
    text = f'{WEEKDAYS_SHORT[day.weekday()]}, {day.day} {MONTHS_GEN[day.month - 1]}'
    return text if day.year == today.year else f'{text} {day.year}'


def format_day(day: datetime.date, today: datetime.date) -> str:
    """Same as format_date, but always names the calendar date too."""
    text = f'{day.day} {MONTHS_GEN[day.month - 1]}'
    return text if day.year == today.year else f'{text} {day.year}'


def end_time(entry: Entry) -> str | None:
    if entry.kind != EntryKind.meeting or entry.time is None or entry.date is None:
        return None
    start = datetime.datetime.combine(entry.date, datetime.time.fromisoformat(entry.time))
    return (start + datetime.timedelta(minutes=entry.duration_minutes or 60)).strftime('%H:%M')


def when(entry: Entry, today: datetime.date) -> str:
    if entry.recurrence is not None:
        parts = [label(entry.recurrence, entry.recurrence_interval)]
        if entry.time:
            parts.append(f'в {entry.time}')
        return ' '.join(parts)
    if entry.date is None:
        return 'без срока'
    parts = [format_date(entry.date, today)]
    if entry.time:
        end = end_time(entry)
        parts.append(f'с {entry.time} до {end}' if end else f'в {entry.time}')
    return ' '.join(parts)


def describe(entry: Entry, today: datetime.date, *, accusative: bool = False) -> str:
    kind = (KIND_ACC if accusative else KIND_NOM)[entry.kind]
    return f'{kind} «{entry.title}», {when(entry, today)}'


def plural(count: int, one: str, few: str, many: str) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def entry_line(entry: Entry, today: datetime.date, *, show_date: bool) -> str:
    parts = []
    if show_date and entry.date is not None:
        parts.append(format_date(entry.date, today))
    if entry.time:
        parts.append(entry.time)
    head = ' '.join(parts)
    done = ' ✓' if entry.status is not None and entry.status.value == 'completed' else ''
    repeat = f', {label(entry.recurrence, entry.recurrence_interval)}' if entry.recurrence else ''
    if getattr(entry.priority, 'value', entry.priority) == 'high':
        repeat += ', важно'
    body = f'{entry.title} ({KIND_NOM[entry.kind]}{repeat}){done}'
    return f'• {head} — {body}' if head else f'• {body}'


def lead(minutes: int) -> str:
    """15 → «за 15 минут», 60 → «за час», 1440 → «за день»."""
    if minutes == 0:
        return 'в момент начала'
    if minutes % 1440 == 0:
        days = minutes // 1440
        return 'за день' if days == 1 else f'за {days} {plural(days, "день", "дня", "дней")}'
    if minutes % 60 == 0:
        hours = minutes // 60
        return 'за час' if hours == 1 else f'за {hours} {plural(hours, "час", "часа", "часов")}'
    return f'за {minutes} {plural(minutes, "минуту", "минуты", "минут")}'
