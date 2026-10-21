"""Several records from one phrase, and filters for «удали все …» — without asking the model.

«напоминания сходить в магазин на 10:00 и 18:00» → two commands with the same title;
«вынести мусор на завтра 10:00, послезавтра 10:00 и через неделю в 10:00» → three;
«сходить в магазин на 10:00 и вынести мусор в 18:00» → two with different titles.
"""

import calendar
import datetime
import re

from src.schemas.assistant import Entry, EntryKind, EntryStatus
from src.services.temporal import MONTHS, parse_date, parse_period

CREATE_VERB = re.compile(r'\b(?:создай|сделай|добавь|поставь|запланируй|запиши|напомни|назначь)\b', re.IGNORECASE)
MONTH_WORD = r'(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*'
WEEKDAY_WORD = r'(?:понедельник|вторник|сред[уа]|четверг|пятниц[уа]|суббот[уа]|воскресенье)'
TIME_RX = re.compile(r'(?:\b(?:в|на|к)\s+)?\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b', re.IGNORECASE)
DATE_RX = re.compile(
    r'(?:\b(?:на|в|во)\s+)?\b(?:сегодня|завтра|послезавтра|'
    r'через\s+(?:\d+|пару|два|две|три|четыре|пять|неделю|месяц|день|год)(?:\s+(?:дн\w*|недел\w*|месяц\w*|год\w*))?|'
    rf'(?:следующ\w+\s+)?{WEEKDAY_WORD}|\d{{1,2}}\s+{MONTH_WORD}|\d{{1,2}}[./]\d{{1,2}}(?:[./]\d{{2,4}})?)\b',
    re.IGNORECASE,
)
COUNT_RX = re.compile(r'\b\d+\s+(?=(?:напоминани|задач|встреч|запис|событи)\w*)', re.IGNORECASE)
SEGMENT_SPLIT = re.compile(r'\s*,\s*(?:и\s+)?|\s+и\s+', re.IGNORECASE)
FILLER = re.compile(r'^(?:и|а|ещё|еще|также|в|на|к|во)$', re.IGNORECASE)


def _strip_slots(text: str) -> str:
    text = TIME_RX.sub(' ', text)
    text = DATE_RX.sub(' ', text)
    return ' '.join(text.split())


def _slot(segment: str) -> tuple[str | None, str | None]:
    date = DATE_RX.search(segment)
    time = TIME_RX.search(segment)
    return (date.group().strip() if date else None, time.group().strip() if time else None)


def _only_slot(segment: str) -> bool:
    rest = _strip_slots(segment)
    return all(FILLER.match(word) for word in rest.split())


def expand_creations(text: str) -> list[str] | None:
    """Rewrite one phrase into several independent create commands, or None if it is a single one."""
    if not CREATE_VERB.search(text) or re.search(r'\bс\s+\d{1,2}[:.]\d{2}\s+до\b', text, re.IGNORECASE):
        return None
    if len(TIME_RX.findall(text)) + len(DATE_RX.findall(text)) < 3 and len(TIME_RX.findall(text)) < 2:
        return None
    cleaned = COUNT_RX.sub('', text)
    segments = [s for s in SEGMENT_SPLIT.split(cleaned) if s and s.strip(' .!?')]
    if len(segments) < 2:
        return None
    head = segments[0]
    head_date, head_time = _slot(head)
    base = _strip_slots(head)
    commands: list[str] = []
    current_date, current_time = head_date, head_time
    pending_dates: list[str] = []

    def emit(title_text: str, date: str | None, time: str | None):
        commands.append(' '.join(part for part in (title_text, date, time) if part))

    for index, segment in enumerate(segments):
        date, time = _slot(segment)
        if index == 0:
            if time is None:
                pending_dates.append(date) if date else None
                continue
            emit(base, date or current_date, time)
            continue
        if _only_slot(segment):
            if time is None and date is not None:
                pending_dates.append(date)
                continue
            for waiting in pending_dates:
                emit(base, waiting, time)
            pending_dates = []
            emit(base, date or current_date, time)
            current_date = date or current_date
            continue
        verb = CREATE_VERB.search(base)
        prefix_kind = re.search(r'\b(напоминани\w*|задач\w*|встреч\w*)\b', base, re.IGNORECASE)
        title = _strip_slots(segment)
        lead = ' '.join(part for part in (verb.group() if verb else 'поставь',
                                          prefix_kind.group() if prefix_kind else None) if part)
        if not CREATE_VERB.search(title):
            title = f'{lead} {title}'
        emit(title, date or current_date, time)
    if pending_dates and head_time:
        for waiting in pending_dates:
            emit(base, waiting, head_time)
    commands = list(dict.fromkeys(c for c in commands if c))
    return commands if len(commands) >= 2 else None


BULK_DELETE_RX = re.compile(
    r'^\s*(?:удали|убери|сотри|очисти)\s+(?:мне\s+)?(?:все|всё|все\s+мои|мои)?\s*(?P<rest>.*?)\s*[.!]*$',
    re.IGNORECASE,
)
PLURAL_KINDS = re.compile(r'\b(задач[иа]?|задачи|напоминани[яй]|встреч[иа]?|записи|событи[яй]|дела)\b', re.IGNORECASE)
TOPIC_RX = re.compile(r'\b(?:связанн\w*\s+с|касающ\w*|про|по\s+теме|на\s+тему)\s+(?P<topic>.+?)\s*$', re.IGNORECASE)
TOPIC_LIST_RX = re.compile(
    r'^\s*(?:покажи|выведи|найди|какие|список)\s+(?:мне\s+)?(?:все\s+|мои\s+|все\s+мои\s+)?'
    r'(?:события|записи|задачи|встречи|напоминания|дела)\s+(?:у\s+меня\s+)?'
    r'(?:связанн\w*\s+с|касающ\w*|про|по\s+теме|на\s+тему)\s+(?P<topic>.+?)\s*[?.!]*$',
    re.IGNORECASE,
)


def _month_range(word: str, today: datetime.date) -> tuple[datetime.date, datetime.date] | None:
    stem = next((s for s in MONTHS if word.casefold().startswith(s)), None)
    if stem is None:
        return None
    month = MONTHS[stem]
    year = today.year if month >= today.month - 6 else today.year + 1
    return datetime.date(year, month, 1), datetime.date(year, month, calendar.monthrange(year, month)[1])


def bulk_filter(text: str, today: datetime.date, now_time: str) -> dict | None:
    """Understand «удали все задачи после 22 октября», «удали встречи с октября по ноябрь» and the like."""
    match = BULK_DELETE_RX.match(text)
    if match is None:
        return None
    rest = match.group('rest')
    has_all = bool(re.search(r'\b(?:все|всё)\b', text, re.IGNORECASE))
    kinds_found = PLURAL_KINDS.findall(rest)
    if not kinds_found and not has_all:
        return None
    if not has_all and not re.search(r'\b(?:задачи|напоминания|встречи|записи|события|дела)\b', rest, re.IGNORECASE):
        return None
    kinds: set[EntryKind] = set()
    for word in kinds_found:
        value = word.casefold()
        if value.startswith('задач'):
            kinds.add(EntryKind.task)
        elif value.startswith('напомин'):
            kinds.add(EntryKind.reminder)
        elif value.startswith('встреч'):
            kinds.add(EntryKind.meeting)
    result: dict = {'kinds': kinds or None, 'status': None, 'date_from': None, 'date_to': None,
                    'past': False, 'topic': None, 'undated': None}
    value = rest.casefold().replace('ё', 'е')
    if re.search(r'\bвыполненн\w*|\bсделанн\w*|\bзавершенн\w*', value):
        result['status'] = EntryStatus.completed
    if re.search(r'\bпрошедш\w*|\bпрошл\w*\b(?!\s+недел)|\bпросроченн\w*', value):
        result['past'] = True
    if re.search(r'\bбез\s+(?:срока|даты)\b|\bбессрочн\w*', value):
        result['undated'] = True
    topic = TOPIC_RX.search(rest)
    if topic:
        result['topic'] = topic.group('topic').strip(' .!?')
        value = value[:topic.start()]
    after = re.search(r'\bпосле\s+(.+?)$', value)
    before = re.search(r'\bдо\s+(.+?)$', value)
    months = re.findall(rf'\b{MONTH_WORD}', value)
    span = re.search(rf'\bс\s+(?P<a>.+?)\s+по\s+(?P<b>.+?)$', value)
    if after:
        day = parse_date(after.group(1), today) or _month_range(after.group(1).split()[0], today) and \
            _month_range(after.group(1).split()[0], today)[1]
        if isinstance(day, tuple):
            day = day[1]
        if day is None:
            return {**result, 'error': True}
        result['date_from'] = day + datetime.timedelta(days=1)
    elif before:
        day = parse_date(before.group(1), today)
        if day is None:
            return {**result, 'error': True}
        result['date_to'] = day - datetime.timedelta(days=1)
    elif span:
        first = parse_date(span.group('a'), today) or (_month_range(span.group('a'), today) or (None, None))[0]
        last = parse_date(span.group('b'), today) or (_month_range(span.group('b'), today) or (None, None))[1]
        if first is None or last is None:
            return {**result, 'error': True}
        result['date_from'], result['date_to'] = first, last
    elif months:
        ranges = [r for r in (_month_range(m, today) for m in months) if r]
        if ranges:
            result['date_from'] = min(r[0] for r in ranges)
            result['date_to'] = max(r[1] for r in ranges)
    else:
        period = parse_period(value, today) if re.search(r'\b(?:на|за|в|эт|следующ|недел|месяц|сегодня|завтра)', value) else None
        if period:
            result['date_from'], result['date_to'] = period
    nothing = not any((kinds, result['status'], result['date_from'], result['date_to'], result['past'],
                       result['topic'], result['undated']))
    if nothing and not re.search(r'\b(?:все|всё)\s+(?:записи|события|дела)\b', text, re.IGNORECASE):
        return None
    return result


def apply_filter(entries: list[Entry], spec: dict, today: datetime.date, now_time: str) -> list[Entry]:
    found = []
    for entry in entries:
        if spec['kinds'] and entry.kind not in spec['kinds']:
            continue
        if spec['status'] is not None and entry.status != spec['status']:
            continue
        if spec['undated'] and entry.date is not None:
            continue
        if spec['past']:
            if entry.date is None or entry.recurrence is not None:
                continue
            if entry.date > today or (entry.date == today and (entry.time or '23:59') >= now_time):
                continue
        if spec['date_from'] and (entry.date is None or entry.date < spec['date_from']):
            continue
        if spec['date_to'] and (entry.date is None or entry.date > spec['date_to']):
            continue
        found.append(entry)
    return found
