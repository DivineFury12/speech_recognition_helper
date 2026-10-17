"""Verify date and time against words actually present in a user request."""

import datetime
import re

from src.schemas.assistant import EntryKind
from src.schemas.assistant_command import AssistantCommand


WEEKDAYS = {
    'понедельник': 0, 'вторник': 1, 'сред': 2, 'четверг': 3,
    'пятниц': 4, 'суббот': 5, 'воскресен': 6,
}
MONTHS = {
    'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4,
    'май': 5, 'мая': 5, 'мае': 5, 'маю': 5,
    'июн': 6, 'июл': 7, 'август': 8, 'сентябр': 9,
    'октябр': 10, 'ноябр': 11, 'декабр': 12,
}
MONTH_ABBREVIATIONS = {
    'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'авг': 8,
    'сен': 9, 'сент': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
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
DURATION_WORDS = {
    **HOUR_WORDS, 'тридцать': 30, 'сорок': 40, 'пятьдесят': 50,
    'полтора': 1.5, 'полторы': 1.5,
}
DURATION_AMOUNT_PATTERN = '|'.join(
    sorted((re.escape(word) for word in DURATION_WORDS), key=len, reverse=True)
)
DURATION_COMPONENT = re.compile(
    rf'(?:(?P<amount>\d{{1,4}}|{DURATION_AMOUNT_PATTERN})\s+)?'
    r'(?P<unit>час(?:а|ов)?|ч\.?|минут(?:у|а|ы)?|мин\.?)\b',
)
DURATION_MARKER_CREATE = re.compile(
    r'\b(?:на|длительностью|продолжительностью|длительность|'
    r'в\s+течение)\s+',
)
DURATION_MARKER_UPDATE = re.compile(
    r'\b(?:длительностью|продолжительностью|длительность|'
    r'(?:сократи|сократить|продли|продлить|увеличь|увеличить)\s+'
    r'(?:(?:ее|его|эту|этот|встречу)\s+)?до|'
    r'(?:сделай|сделать|установи|измени)\s+(?:её\s+|ее\s+|его\s+)?'
    r'(?:(?:длительность|длительностью)\s+)?(?:на\s+|до\s+)?)',
)
HOUR_PATTERN = '|'.join(sorted((re.escape(word) for word in HOUR_WORDS), key=len, reverse=True))
HOUR_PATTERN = rf'(?:\d{{1,2}}|{HOUR_PATTERN})'
COUNT_PATTERN = rf'(?:\d{{1,4}}|{HOUR_PATTERN})'
DATE_UNIT_PATTERN = r'(?P<unit>недел\w*|день|дня|дней|сутк\w*|суток)'
MONTH_PATTERN = '|'.join(
    [rf'{stem}\w*' for stem in MONTHS if not stem.startswith('ма')]
    + [r'ма[йяею]']
    + [rf'{stem}\.?(?!\w)' for stem in MONTH_ABBREVIATIONS]
)

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
    'удали', 'удалить', 'убери', 'убрать', 'сотри', 'перенеси', 'перенести',
    'отметь', 'отметить', 'выполненной', 'выполнено', 'переименуй',
    'переименовать', 'измени', 'изменить', 'эту', 'этот', 'ее', 'его',
}


def _in_text(phrase: str | None, text: str) -> bool:
    if not phrase:
        return False
    return ' '.join(phrase.casefold().split()) in ' '.join(text.casefold().split())


def _month_number(token: str) -> int:
    value = token.rstrip('.')
    for stem, month in MONTHS.items():
        if value.startswith(stem):
            return month
    return MONTH_ABBREVIATIONS[value]


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
    amount = (1 if count is None else 2 if count in ('пару', 'пара') else
              int(count) if count.isdigit() else HOUR_WORDS[count])
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
        rf'\b(?:через|спустя)\s+(?:(?P<count>{COUNT_PATTERN}|пару|пара)\s+)?{DATE_UNIT_PATTERN}\b',
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

    month_match = re.search(
        rf'(?<!\d)(?P<day>\d{{1,2}})\s+(?P<month>{MONTH_PATTERN})\.?(?!\w)', value,
    )
    if month_match:
        return _calendar_date(int(month_match.group('day')),
                              _month_number(month_match.group('month')), None, today)

    for stem, weekday in WEEKDAYS.items():
        if stem in value:
            if 'следующ' in value:
                start_of_next_week = today + datetime.timedelta(days=7 - today.weekday())
                return start_of_next_week + datetime.timedelta(days=weekday)
            days_ahead = (weekday - today.weekday()) % 7
            return today + datetime.timedelta(days=days_ahead)
    return None


def parse_target_date(phrase: str, today: datetime.date) -> datetime.date | None:
    """Resolve references to existing records, including dates in the past."""
    value = phrase.casefold().replace('ё', 'е')
    if re.search(r'\bпозавчера\b', value):
        return today - datetime.timedelta(days=2)
    if re.search(r'\bвчера\b', value):
        return today - datetime.timedelta(days=1)
    if re.search(r'\bпрошл\w*\b', value):
        for stem, weekday in WEEKDAYS.items():
            if stem in value:
                start = today - datetime.timedelta(days=today.weekday() + 7)
                return start + datetime.timedelta(days=weekday)
    return parse_date(phrase, today)


def parse_period(text: str, today: datetime.date) -> tuple[datetime.date, datetime.date] | None:
    """Resolve common list periods from the user's words, not model dates."""
    value = text.casefold().replace('ё', 'е')
    numeric_range = re.search(
        r'\bс\s+(?P<start>\d{1,2})[./](?P<start_month>\d{1,2})'
        r'(?:[./](?P<start_year>\d{2,4}))?\s+по\s+'
        r'(?P<end>\d{1,2})[./](?P<end_month>\d{1,2})'
        r'(?:[./](?P<end_year>\d{2,4}))?(?!\d)', value,
    )
    if numeric_range:
        start_year = numeric_range.group('start_year') or numeric_range.group('end_year')
        end_year = numeric_range.group('end_year') or numeric_range.group('start_year')
        start_year = 2000 + int(start_year) if start_year and len(start_year) == 2 else int(start_year or today.year)
        end_year = 2000 + int(end_year) if end_year and len(end_year) == 2 else int(end_year or today.year)
        if (not numeric_range.group('start_year') and not numeric_range.group('end_year')
                and int(numeric_range.group('end_month')) < int(numeric_range.group('start_month'))):
            end_year += 1
        try:
            start = datetime.date(start_year, int(numeric_range.group('start_month')),
                                  int(numeric_range.group('start')))
            end = datetime.date(end_year, int(numeric_range.group('end_month')),
                                int(numeric_range.group('end')))
        except ValueError:
            return None
        return (start, end) if start <= end else None
    range_match = re.search(
        rf'\bс\s+(?P<start>\d{{1,2}})(?:\s+(?P<start_month>{MONTH_PATTERN}))?'
        rf'\s+по\s+(?P<end>\d{{1,2}})\s+(?P<end_month>{MONTH_PATTERN})\.?(?!\w)',
        value,
    )
    if range_match:
        start_month_text = range_match.group('start_month') or range_match.group('end_month')
        try:
            start = datetime.date(today.year, _month_number(start_month_text),
                                  int(range_match.group('start')))
            end = datetime.date(today.year, _month_number(range_match.group('end_month')),
                                int(range_match.group('end')))
        except ValueError:
            return None
        return (start, end) if start <= end else None
    relative_month = re.search(
        r'\b(?P<which>эт\w*|текущ\w*|следующ\w*|прошл\w*)\s+месяц\w*\b',
        value,
    )
    if relative_month:
        which = relative_month.group('which')
        offset = 1 if which.startswith('следующ') else -1 if which.startswith('прошл') else 0
        index = today.year * 12 + today.month - 1 + offset
        year, zero_month = divmod(index, 12)
        start = datetime.date(year, zero_month + 1, 1)
        next_year, next_zero_month = divmod(index + 1, 12)
        end = datetime.date(next_year, next_zero_month + 1, 1) - datetime.timedelta(days=1)
        return start, end
    if re.search(r'\bна\s+месяц\b', value):
        return today, today + datetime.timedelta(days=29)
    month_match = next((match for match in re.finditer(
        rf'(?<!\w)(?P<month>{MONTH_PATTERN})\.?(?:\s+(?P<year>\d{{4}}))?(?!\w)',
        value,
    ) if not re.search(r'\d{1,2}\s+$', value[:match.start()])), None)
    if month_match:
        month = _month_number(month_match.group('month'))
        year = int(month_match.group('year') or today.year)
        prefix = value[:month_match.start()]
        if month_match.group('year') is None:
            if re.search(r'\bследующ\w*\s*$', prefix):
                year += 1
            elif re.search(r'\bпрошл\w*\s*$', prefix):
                year -= 1
        try:
            start = datetime.date(year, month, 1)
            end = (datetime.date(year + 1, 1, 1) if month == 12
                   else datetime.date(year, month + 1, 1)) - datetime.timedelta(days=1)
        except ValueError:
            return None
        return start, end
    season_match = re.search(r'\b(?P<season>весн\w*|лет\w*|осен\w*|зим\w*)\b', value)
    if season_match:
        season = season_match.group('season')
        month = (3 if season.startswith('весн') else 6 if season.startswith('лет')
                 else 9 if season.startswith('осен') else 12)
        year = today.year - (today.month < 3) if month == 12 else today.year
        prefix = value[:season_match.start()]
        if re.search(r'\bследующ\w*\s*$', prefix):
            year += 1
        elif re.search(r'\bпрошл\w*\s*$', prefix):
            year -= 1
        start = datetime.date(year, month, 1)
        end = (datetime.date(year + 1, 3, 1) if month == 12
               else datetime.date(year, month + 3, 1)) - datetime.timedelta(days=1)
        return start, end
    if re.search(r'\b(?:на|за)\s+последн\w*\s+недел\w*\b', value):
        return today - datetime.timedelta(days=6), today
    if re.search(r'\bближайш\w*\s+недел\w*\b', value):
        return today, today + datetime.timedelta(days=6)
    if re.search(r'\bследующ\w*\s+недел\w*\b', value):
        start = today + datetime.timedelta(days=7 - today.weekday())
        if re.search(r'\b(?:втор\w*|2-?й)\s+половин\w*\b', value):
            return start + datetime.timedelta(days=3), start + datetime.timedelta(days=6)
        if re.search(r'\b(?:перв\w*|1-?й)\s+половин\w*\b', value):
            return start, start + datetime.timedelta(days=2)
        return start, start + datetime.timedelta(days=6)
    if re.search(r'\b(?:на|за|в)\s+прошл\w*\s+недел\w*\b', value):
        start = today - datetime.timedelta(days=today.weekday() + 7)
        return start, start + datetime.timedelta(days=6)
    if re.search(r'\b(?:на|за|в)\s+(?:эт\w*|текущ\w*)\s+недел\w*\b', value):
        start = today - datetime.timedelta(days=today.weekday())
        return start, start + datetime.timedelta(days=6)
    if re.search(r'\b(?:на|за)\s+недел\w*\b', value) and 'через' not in value:
        return today, today + datetime.timedelta(days=6)
    if re.search(r'\bвыходн\w*\b', value):
        start = today + datetime.timedelta(days=(5 - today.weekday()) % 7)
        return start, start + datetime.timedelta(days=1)
    recent = re.search(rf'\b(?:за\s+)?последн\w*\s+(?P<count>{COUNT_PATTERN})\s+дн\w*\b', value)
    if recent:
        token = recent.group('count')
        count = int(token) if token.isdigit() else HOUR_WORDS[token]
        if 1 <= count <= 366:
            return today - datetime.timedelta(days=count - 1), today
    coming = re.search(rf'\b(?:на\s+)?(?:ближайш\w*|следующ\w*)\s+(?P<count>{COUNT_PATTERN})\s+дн\w*\b', value)
    if coming:
        token = coming.group('count')
        count = int(token) if token.isdigit() else HOUR_WORDS[token]
        if 1 <= count <= 366:
            return today, today + datetime.timedelta(days=count - 1)
    day = parse_date(text, today)
    if day is not None:
        if re.search(r'\bдо\s+', value):
            return (today, day) if today <= day else None
        explicit_day_month = bool(re.search(
            rf'(?<!\d)\d{{1,2}}\s+(?:{MONTH_PATTERN})\b|'
            r'(?<!\d)\d{1,2}[./]\d{1,2}(?![./]\d)', value,
        ))
        explicit_year = bool(re.search(
            r'(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)|'
            r'(?<!\d)\d{1,2}[./]\d{1,2}[./]\d{2,4}(?!\d)', value,
        ))
        past_query = bool(re.search(r'\b(?:за|было|были|была|прошедш\w*)\b', value))
        if explicit_day_month and not explicit_year and past_query and day > today:
            try:
                day = day.replace(year=day.year - 1)
            except ValueError:
                return None
        return day, day
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
        rf'\b(?:в|к|на)\s+(?P<hour>{HOUR_PATTERN})\b'
        rf'(?!\s+(?:{MONTH_PATTERN}|минут\w*|дн\w*))', value,
    )
    if simple:
        token = simple.group('hour')
        hour = int(token) if token.isdigit() else HOUR_WORDS[token]
        if hour < 24:
            if 1 <= hour <= 11:
                return None
            return f'{hour:02d}:00'
    return None


def ambiguous_time_phrase(text: str) -> str | None:
    """Find a spoken 1–11 o'clock without a morning/evening qualifier."""
    value = text.casefold().replace('ё', 'е')
    match = re.search(
        rf'\b(?:в|к|на)\s+(?P<hour>{HOUR_PATTERN})\b'
        rf'(?!:\d|\s+(?:{MONTH_PATTERN}|минут\w*|дн\w*|утра\b|дня\b|вечера\b|ночи\b))',
        value,
    )
    if match is None:
        return None
    token = match.group('hour')
    hour = int(token) if token.isdigit() else HOUR_WORDS[token]
    return match.group() if 1 <= hour <= 11 else None


def ambiguous_time_options(phrase: str | None) -> tuple[str, str] | None:
    if not phrase:
        return None
    match = ambiguous_time_phrase(phrase)
    if match is None:
        return None
    token = match.split()[-1]
    hour = int(token) if token.isdigit() else HOUR_WORDS[token]
    return f'{hour:02d}:00', f'{hour + 12:02d}:00'


def resolve_time_clarification(phrase: str | None, reply: str) -> str | None:
    """Resolve a reply to a two-option time question without inventing an hour."""
    if ambiguous_time_options(phrase) is None:
        return None
    explicit = parse_time(reply)
    if explicit is not None:
        return explicit
    value = reply.casefold().replace('ё', 'е')
    if re.search(r'\b(?:не|нет|ни)\b', value):
        return None
    periods = {
        'утра': 'утра', 'утром': 'утра',
        'дня': 'дня', 'днем': 'дня',
        'вечера': 'вечера', 'вечером': 'вечера',
        'ночи': 'ночи', 'ночью': 'ночи',
    }
    named = {normalized for word, normalized in periods.items()
             if re.search(rf'\b{word}\b', value)}
    if len(named) != 1:
        return None
    return parse_time(f'{phrase} {named.pop()}')


def replacement_schedule_reply(
    reply: str, today: datetime.date,
) -> tuple[datetime.date | None, str | None, str | None] | None:
    """Read a standalone schedule without interpreting a new command as a reply."""
    value = reply.casefold().replace('ё', 'е')
    if (value.rstrip().endswith('?') or not re.match(
        r'^\s*(?:на\s+|в\s+|к\s+|через\s+|спустя\s+|сегодня\b|завтра\b|'
        r'послезавтра\b|следующ\w*\s+|\d{1,2}(?:[:./]|\s+))',
        value,
    ) or re.search(
        r'\b(?:создай|создать|добавь|добавить|напомни|покажи|найди|'
        r'удали|удалить|перенеси|перенести|отметь)\b', value,
    )):
        return None
    date = parse_date(reply, today)
    time = parse_time(reply)
    uncertain_time = ambiguous_time_phrase(reply)
    if date is None and time is None and uncertain_time is None:
        return None
    return date, time, uncertain_time


def parse_duration(text: str, *, for_update: bool = False) -> int | None:
    """Read an explicit meeting length; a bare move 'на час' is not a length."""
    value = text.casefold().replace('ё', 'е')
    markers = DURATION_MARKER_UPDATE if for_update else DURATION_MARKER_CREATE
    for marker in markers.finditer(value):
        remaining = value[marker.end():].lstrip()
        if remaining.startswith('полчаса'):
            return 30
        total = 0
        matched = False
        for _ in range(2):
            component = DURATION_COMPONENT.match(remaining)
            if component is None:
                break
            amount_token = component.group('amount')
            amount = (int(amount_token) if amount_token and amount_token.isdigit()
                      else DURATION_WORDS[amount_token] if amount_token else 1)
            minutes = amount * (60 if component.group('unit').startswith(('час', 'ч')) else 1)
            total += minutes
            matched = True
            remaining = remaining[component.end():].lstrip()
            remaining = re.sub(r'^и\s+', '', remaining)
        if matched and total == int(total):
            total = int(total)
            return total if 1 <= total <= 1440 else None
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
            and not re.fullmatch(rf'(?:{MONTH_PATTERN})', word)
            and not any(word.startswith(stem) for stem in WEEKDAYS)
        }

    return bool(content_words(title) & content_words(text))


def _query_has_source(query: str | None, text: str) -> bool:
    if not query:
        return False
    source_words = re.findall(r'[а-яёa-z]+', text.casefold().replace('ё', 'е'))
    query_words = re.findall(r'[а-яёa-z]+', query.casefold().replace('ё', 'е'))
    return any(
        len(word) >= 3 and any(source.startswith(word[:3]) for source in source_words)
        for word in query_words
        if word not in TITLE_STOPWORDS
    )


SEARCH_FILTER_WORDS = TITLE_STOPWORDS | {
    'какие', 'какая', 'какой', 'есть', 'выведи', 'перечисли', 'найди',
    'поищи', 'все', 'моих', 'планы', 'планов', 'дела', 'делах',
    'события', 'событий', 'событие', 'запланировано', 'связанные',
    'связанных', 'связано', 'этот', 'этом', 'текущий', 'текущем', 'прошлый',
    'прошлом', 'следующий', 'следующем', 'ближайший', 'ближайшую',
}
SEARCH_FILTER_STEMS = (
    'напомин', 'задач', 'встреч', 'созвон', 'запис', 'событ',
    'недел', 'месяц', 'сегодня', 'завтра', 'позавчера',
)


def _topic_words(value: str) -> list[str]:
    words = re.findall(r'[а-яёa-z]+', value.casefold().replace('ё', 'е'))
    return [word for word in words
            if len(word) >= 3 and word not in SEARCH_FILTER_WORDS
            and not word.startswith(SEARCH_FILTER_STEMS)
            and not re.fullmatch(rf'(?:{MONTH_PATTERN})', word)
            and not re.fullmatch(r'(?:осен\w*|весн\w*|зим\w*|лето|летом|лета)', word)
            and word not in {'день', 'дня', 'дней', 'сутки', 'суток'}
            and not any(word.startswith(stem) for stem in WEEKDAYS)]


def verify_search_query(query: str | None, text: str, *, topic: bool = False) -> str | None:
    if not _query_has_source(query, text):
        return None
    if topic:
        source = _topic_words(text)
        if not source or not any(
            any(word.startswith(candidate[:3]) for word in source)
            for candidate in _topic_words(query)
        ):
            return None
    return query


def mentioned_entry_kind(text: str) -> EntryKind | None:
    """Use a type filter only when the user actually named that type."""
    value = text.casefold().replace('ё', 'е')
    if re.search(r'\bнапомин\w*\b', value):
        return EntryKind.reminder
    if re.search(r'\bзадач\w*\b', value):
        return EntryKind.task
    if re.search(r'\b(?:встреч\w*|созвон\w*)\b', value):
        return EntryKind.meeting
    return None


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
    time_phrase = command.time_phrase if _in_text(command.time_phrase, text) else None
    duration_text = text
    if time is not None and time_phrase and parse_time(time_phrase) == time:
        # A clock expression such as «на 17 часов» is not also a 17-hour
        # duration. A separate length later in the sentence is still parsed.
        duration_text = re.sub(re.escape(time_phrase), '', text, count=1, flags=re.IGNORECASE)
    duration = parse_duration(duration_text) if command.kind == EntryKind.meeting else None
    if (duration is not None and time is not None and not time_phrase
            and command.duration_phrase is None and duration == int(time[:2]) * 60
            and parse_duration(text) == duration):
        duration = None
    return command.model_copy(update={
        'title': command.title if _title_has_source(command.title, text) else None,
        'date': date, 'date_phrase': command.date_phrase if date is not None else None,
        'time': time,
        'time_phrase': (
            command.time_phrase if time is not None else ambiguous_time_phrase(text)
        ),
        'duration_minutes': duration,
    })


def verify_mutation_fields(
    command: AssistantCommand, text: str, today: datetime.date,
) -> AssistantCommand:
    """Accept only target and replacement details supported by the latest utterance."""
    target_date = (
        parse_target_date(command.target_date_phrase, today)
        if _in_text(command.target_date_phrase, text) else None
    )
    target_time = (
        parse_time(command.target_time_phrase)
        if _in_text(command.target_time_phrase, text) else None
    )
    if command.action.value == 'update_entry':
        date = parse_date(command.date_phrase, today) if _in_text(command.date_phrase, text) else None
        time = parse_time(command.time_phrase) if _in_text(command.time_phrase, text) else None
        new_title = command.new_title if _title_has_source(command.new_title, text) else None
        duration_minutes = parse_duration(text, for_update=True)
        time_phrase = command.time_phrase if time is not None else ambiguous_time_phrase(text)
    else:
        date = time = new_title = None
        duration_minutes = None
        time_phrase = None
    return command.model_copy(update={
        'target_query': command.target_query
        if _query_has_source(command.target_query, text) else None,
        'target_date': target_date,
        'target_time': target_time,
        'date': date,
        'time': time,
        'time_phrase': time_phrase,
        'new_title': new_title,
        'duration_minutes': duration_minutes,
    })
