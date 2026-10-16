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
    (r'\bкажд\w*\s+(?:месяц\w*|\d{1,2}(?:-?(?:го|е))?\s+числ\w*)|\bежемесячн\w*\b', Recurrence.monthly),
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
                date_from: datetime.date, date_to: datetime.date) -> list[datetime.date]:
    if not rule:
        return [start] if date_from <= start <= date_to else []
    rule = Recurrence(rule)
    result: list[datetime.date] = []
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


def next_occurrence(start: datetime.date, rule: Recurrence | str, after: datetime.date) -> datetime.date | None:
    found = occurrences(start, rule, after + datetime.timedelta(days=1), after + datetime.timedelta(days=400))
    return found[0] if found else None
