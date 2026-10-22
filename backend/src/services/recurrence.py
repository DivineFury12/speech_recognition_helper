"""Recurring entries: the rule comes from the user's words, occurrences from code."""

import calendar
import datetime
import re

from src.schemas.assistant import Recurrence

_WEEKDAY = r'(?:понедельник|вторник|сред|четверг|пятниц|суббот|воскресень)\w*'
PATTERNS = [
    (r'\bпо\s+будн(?:ям|ым\s+дням)\b|\bкажд\w*\s+будн\w*\s+день\b|\bв\s+будни\b', Recurrence.weekdays),
    (r'\bкажд\w*\s+(?:день|утро|вечер)\b|\bежедневн\w*\b', Recurrence.daily),
    (rf'\bкажд\w*\s+(?:недел\w*|{_WEEKDAY})|\bеженедельн\w*\b|\bпо\s+{_WEEKDAY}', Recurrence.weekly),
    (r'\bкажд\w*\s+(?:месяц\w*|(?:\d{1,2}(?:-?(?:го|е))?|перв\w*|втор\w*|трет\w*|пят\w*|десят\w*|пятнадцат\w*|двадцат\w*|последн\w*)\s+числ\w*)|\bежемесячн\w*\b', Recurrence.monthly),
]
LABEL = {
    Recurrence.daily: 'каждый день',
    Recurrence.weekdays: 'по будням',
    Recurrence.weekly: 'каждую неделю',
    Recurrence.monthly: 'каждый месяц',
}


def find_recurrence(text: str) -> Recurrence | None:
    value = text.casefold().replace('ё', 'е')
    for pattern, rule in PATTERNS:
        if re.search(pattern, value):
            return rule
    return None


def first_date(rule: Recurrence, today: datetime.date) -> datetime.date:
    """When the user did not name a start date, the series starts today."""
    if rule == Recurrence.weekdays:
        while today.weekday() >= 5:
            today += datetime.timedelta(days=1)
    return today


def _add_months(day: datetime.date, months: int) -> datetime.date:
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return datetime.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def occurrences(start: datetime.date, rule: Recurrence | str | None,
                date_from: datetime.date, date_to: datetime.date, interval: int = 1) -> list[datetime.date]:
    if not rule:
        return [start] if date_from <= start <= date_to else []
    rule = Recurrence(rule)
    result: list[datetime.date] = []
    if interval > 1 and rule in (Recurrence.daily, Recurrence.weekly):
        step = datetime.timedelta(days=interval * (7 if rule == Recurrence.weekly else 1))
        skip = max(0, (date_from - start).days // step.days)
        day = start + step * skip
        while day <= date_to:
            if day >= date_from:
                result.append(day)
            day += step
        return result
    if interval > 1 and rule == Recurrence.monthly:
        months = 0
        while True:
            day = _add_months(start, months)
            months += interval
            if day > date_to:
                return result
            if day >= date_from:
                result.append(day)
    if rule in (Recurrence.daily, Recurrence.weekdays):
        day = max(start, date_from)
        while day <= date_to:
            if rule == Recurrence.daily or day.weekday() < 5:
                result.append(day)
            day += datetime.timedelta(days=1)
    elif rule == Recurrence.weekly:
        day = start
        if day < date_from:
            day += datetime.timedelta(weeks=(date_from - day).days // 7)
        while day <= date_to:
            if day >= date_from:
                result.append(day)
            day += datetime.timedelta(weeks=1)
    else:
        months = max(0, (date_from.year - start.year) * 12 + date_from.month - start.month - 1)
        while True:
            day = _add_months(start, months)
            months += 1
            if day > date_to:
                break
            if day >= date_from:
                result.append(day)
    return result


def next_occurrence(start: datetime.date, rule: Recurrence | str, after: datetime.date,
                    interval: int = 1) -> datetime.date | None:
    found = occurrences(start, rule, after + datetime.timedelta(days=1), after + datetime.timedelta(days=400), interval)
    return found[0] if found else None


_NUMBERS = {'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6, 'семь': 7, 'десять': 10}
INTERVAL_RE = re.compile(
    r'\b(?:кажд\w+|раз\s+в)\s+(?P<n>\d{1,3}|два|две|три|четыре|пять|шесть|семь|десять)\s+'
    r'(?P<unit>дн\w*|день|недел\w*|месяц\w*|час\w*|минут\w*)'
    r'|\b(?P<other>через\s+день|через\s+недел[юь])\b',
    re.IGNORECASE,
)


def find_interval(text: str) -> tuple[Recurrence | str, int] | None:
    """«каждые 3 дня» → (daily, 3), «раз в 2 недели» → (weekly, 2), «каждые 2 часа» → ('hours', 2)."""
    match = INTERVAL_RE.search(text.casefold().replace('ё', 'е'))
    if match is None:
        return None
    if match.group('other'):
        return (Recurrence.daily, 2) if 'день' in match.group('other') else (Recurrence.weekly, 2)
    token = match.group('n')
    number = int(token) if token.isdigit() else _NUMBERS[token]
    unit = match.group('unit')
    if unit.startswith(('час', 'минут')):
        return 'hours', number
    rule = (Recurrence.daily if unit.startswith(('дн', 'день')) else
            Recurrence.weekly if unit.startswith('недел') else Recurrence.monthly)
    return rule, number


def label(rule: Recurrence, interval: int | None = None) -> str:
    if not interval or interval == 1:
        return LABEL[rule]
    forms = {Recurrence.daily: ('день', 'дня', 'дней'), Recurrence.weekly: ('неделю', 'недели', 'недель'),
             Recurrence.monthly: ('месяц', 'месяца', 'месяцев')}[rule]
    tail = interval % 10 if interval % 100 not in range(11, 15) else 5
    word = forms[0] if tail == 1 else forms[1] if 2 <= tail <= 4 else forms[2]
    return f'раз в {interval} {word}'
