"""Ground a model-selected entry in words supplied by the user."""

import datetime
import re

from src.schemas.assistant import Entry
from src.schemas.dialogue import ScheduleRole
from src.services.temporal import parse_target_date, parse_time


ORDINALS = {
    'первую': 1, 'первый': 1, 'первая': 1, 'первое': 1, 'первой': 1,
    'вторую': 2, 'второй': 2, 'вторая': 2, 'второе': 2,
    'третью': 3, 'третий': 3, 'третья': 3, 'третье': 3,
    'четвертую': 4, 'четвертый': 4, 'четвертая': 4,
    'пятую': 5, 'пятый': 5, 'пятая': 5,
}


def selection_is_grounded(
    text: str, choice: int, entries: list[Entry], today: datetime.date,
    schedule_role: ScheduleRole,
) -> bool:
    """A bare proposed date never identifies an existing entry for a mutation."""
    value = text.casefold().replace('ё', 'е')
    if re.search(r'\b(?:не|нет|ни)\b', value):
        return False

    ordinals = {ORDINALS[word] for word in re.findall(r'[а-я]+', value)
                if word in ORDINALS}
    if ordinals:
        return ordinals == {choice}
    number = re.search(
        r'(?:^\s*|\b(?:номер|номером|выбери)\s+)(\d{1,2})(?=$|[\s,.])',
        value,
    )
    if number is not None:
        return int(number.group(1)) == choice

    if schedule_role != ScheduleRole.existing or not re.search(
        r'\b(?:ту|тот|той|та|которая|который|что)\b', value,
    ):
        return False
    date = parse_target_date(text, today)
    time = parse_time(text)
    if date is None and time is None:
        return False
    matches = [index for index, entry in enumerate(entries, 1)
               if (date is None or entry.date == date)
               and (time is None or entry.time == time)]
    return matches == [choice]
