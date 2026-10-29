"""A stand-in for GigaChat built on plain rules.

It answers the same structured requests as the GigaChat client, so the extractor
verifies its output exactly like a model answer. Used when there is no key, and as a
fallback when GigaChat is unreachable or returns an error.
"""

import datetime
import json
import logging
import re

from src.schemas.assistant_command import AssistantCommand
from src.services.entries import EntryService
from src.services.guards import yes_no
from src.services.recurrence import find_recurrence
from src.services.temporal import ambiguous_time_phrase, parse_date, parse_period, parse_target_date, parse_time

logger = logging.getLogger(__name__)

CREATE_RE = re.compile(r'\b(?:напомни\w*|поставь|создай|добавь|запланируй|запиши|назначь|нужно|надо|необходимо)\b')
LIST_RE = re.compile(r'\b(?:что у меня|покажи|какие|какая|какой|есть ли|список|что запланировано|что на)\b')
SEARCH_RE = re.compile(r'\b(?:найди|поищи|когда у меня был\w*|когда был\w*|где)\b')
DELETE_RE = re.compile(r'\b(?:удали\w*|убери|сотри|отмени встречу|отмени задачу|отмени напоминание)\b')
COMPLETE_RE = re.compile(r'\b(?:отметь|выполнен\w*|сделан\w*|готово)\b')
UPDATE_RE = re.compile(r'\b(?:перенеси|перенести|сдвинь|передвинь|переименуй|измени|поменяй|сделай её|сделай его)\b')
RENAME_RE = re.compile(r'\bпереименуй\w*\s+(?:.*?\s+)?в\s+[«"]?(?P<title>.+?)[»"]?\s*$', re.IGNORECASE)

DATE_WORDS = (r'(?:сегодня|завтра|послезавтра|вчера|позавчера|через\s+\S+(?:\s+\S+)?|в\s+(?:следующ\w+\s+)?'
              r'(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)|'
              r'\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*|'
              r'\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|до\s+конца\s+\w+)')
TIME_WORDS = (r'(?:в|к|на)\s+\d{1,2}(?::\d{2})?(?:\s*(?:утра|вечера|дня|ночи|часов|часа|час))?|'
              r'в\s+(?:полдень|полночь)|(?:утром|вечером|днём|днем)')
NOISE = re.compile(
    rf'\b(?:{DATE_WORDS})\b|\b(?:{TIME_WORDS})\b|\b(?:напомни\w*|поставь|создай|добавь|запланируй|запиши|назначь|'
    r'мне|пожалуйста|нужно|надо|необходимо|задачу|задача|напоминание|встречу|о том,? что|чтобы|что|'
    r'важн\w*|срочн\w*)\b',
    re.IGNORECASE,
)


def _clean(text: str) -> str | None:
    value = NOISE.sub(' ', text)
    value = re.sub(r'\s+', ' ', value).strip(' ,.!?-')
    return value or None


def _phrase(pattern: str, text: str) -> str | None:
    match = re.search(rf'\b(?:{pattern})\b', text, re.IGNORECASE)
    return match.group(0) if match else None


def _kind(value: str) -> str | None:
    if re.search(r'\bвстреч|\bсозвон|\bпланёрк|\bпланерк|\bсовещан', value):
        return 'meeting'
    if re.search(r'\bнапомн', value):
        return 'reminder'
    if re.search(r'\bзадач', value):
        return 'task'
    return None


def _target(text: str, verbs: re.Pattern) -> str | None:
    value = verbs.sub(' ', text)
    value = re.sub(r'\b(?:выполненной|выполненным|как выполненную|на\s+.+$|её|ее|его|эту|этот)\b', ' ', value,
                   flags=re.IGNORECASE)
    value = re.sub(rf'\b(?:{DATE_WORDS})\b|\b(?:{TIME_WORDS})\b', ' ', value, flags=re.IGNORECASE)
    value = re.sub(r'\s+', ' ', value).strip(' ,.!?-')
    return value or None


def command_from_rules(text: str, today: datetime.date) -> dict:
    value = text.casefold().replace('ё', 'е')
    payload = {name: None for name in AssistantCommand.provider_schema()['properties']}
    payload['action'] = 'unknown'
    date_phrase = _phrase(DATE_WORDS, text)
    time_phrase = _phrase(TIME_WORDS, text)

    if DELETE_RE.search(value):
        payload.update(action='delete_entry', kind=_kind(value), target_query=_target(text, DELETE_RE),
                       target_date_phrase=date_phrase,
                       target_date=(parse_target_date(text, today) or None) and str(parse_target_date(text, today)))
    elif UPDATE_RE.search(value):
        rename = RENAME_RE.search(text)
        payload.update(action='update_entry', kind=_kind(value), update_mode='edit' if rename else 'move',
                       target_query=_target(re.sub(r'\s+в\s+[«"].*$', '', text), UPDATE_RE),
                       new_title=rename.group('title') if rename else None)
        if not rename:
            day, time = parse_date(text, today), parse_time(text)
            payload.update(date=str(day) if day else None, date_phrase=date_phrase if day else None,
                           time=time, time_phrase=time_phrase if time else ambiguous_time_phrase(text))
    elif COMPLETE_RE.search(value) and not CREATE_RE.search(value):
        payload.update(action='complete_entry', kind=_kind(value), target_query=_target(text, COMPLETE_RE))
    elif SEARCH_RE.search(value):
        query = _target(text, SEARCH_RE)
        payload.update(action='search_entries', kind=_kind(value), query=query)
    elif LIST_RE.search(value) and not CREATE_RE.search(value):
        payload.update(action='list_entries', kind=_kind(value))
    elif CREATE_RE.search(value) or date_phrase or time_phrase:
        day, time = parse_date(text, today), parse_time(text)
        title = _clean(text)
        kind = _kind(value)
        if kind == 'meeting' and title and re.match(r'(?i)с\s|со\s', title):
            title = f'Встреча {title}'
        if kind is None and title:
            kind = 'reminder' if time else 'task'
        payload.update(action='create_entry', kind=kind,
                       title=title, date=str(day) if day else None,
                       date_phrase=date_phrase if day else None, time=time,
                       time_phrase=time_phrase if time else ambiguous_time_phrase(text))
    return payload


ORDINALS = {'перв': 1, 'втор': 2, 'трет': 3, 'четв': 4, 'пят': 5}
COMMAND_RE = re.compile(r'\b(?:создай|добавь|поставь|напомни|покажи|найди|удали|перенеси|сдвинь|отметь|'
                        r'переименуй|что у меня)\b', re.IGNORECASE)


def _reply_act(reply: str) -> dict:
    """How a reply relates to a pending question, by rules."""
    unclear = {'act': 'unclear', 'choice': None, 'schedule_role': 'none'}
    value = reply.casefold().strip()
    if re.fullmatch(r'(?:нет[,\s]+)?(?:отмена|отмени|не надо|забудь|передумал\w*)[.!]?', value):
        return {**unclear, 'act': 'cancel'}
    if COMMAND_RE.search(value) and len(value.split()) > 2:
        return {**unclear, 'act': 'new_command'}
    answer = yes_no(reply)
    if answer == 'yes':
        return {**unclear, 'act': 'confirm'}
    if answer == 'no':
        return {**unclear, 'act': 'reject'}
    number = re.fullmatch(r'(?:номер\s+)?(\d{1,2})', value)
    if number:
        return {'act': 'select', 'choice': int(number.group(1)), 'schedule_role': 'none'}
    for stem, choice in ORDINALS.items():
        if re.fullmatch(rf'{stem}\w*', value):
            return {'act': 'select', 'choice': choice, 'schedule_role': 'none'}
    if parse_date(reply, datetime.date.today()) or parse_time(reply):
        return {**unclear, 'act': 'continue_command'}
    return unclear


class RulesModel:
    """Same interface as GigaChatClient.complete_json, answered by rules."""

    name = 'rules'
    healthy = False

    def complete_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
        data = json.loads(user_prompt)
        properties = json_schema.get('properties', {})
        if 'parts' in properties:
            return {'parts': [data.get('text', '')]}
        if 'relevant_numbers' in properties:
            # Conservative fallback: accept only visible word overlap when the
            # external model cannot judge semantic relevance.
            topic = EntryService._search_stems(data.get('topic') or '')
            return {'relevant_numbers': [
                record['number'] for record in data.get('records', [])
                if topic & EntryService._search_stems(record.get('title') or '')
            ]}
        if 'decision' in properties:
            answer = yes_no(data.get('reply') or data.get('text') or '')
            return {'decision': answer or 'unclear'}
        if 'act' in properties:
            return _reply_act(data.get('reply', ''))
        today = datetime.date.fromisoformat(data.get('today') or datetime.date.today().isoformat())
        if 'recurrence_phrase' in properties:
            text = data.get('text', '')
            rule = find_recurrence(text)
            period = parse_period(text, today)
            if rule is None or period is None or period[0] == period[1]:
                return dict(recurrence=None, recurrence_phrase=None, date_from=None,
                            date_to=None, from_phrase=None, to_phrase=None)
            return dict(recurrence=rule.value, recurrence_phrase=text,
                        date_from=period[0].isoformat(), date_to=period[1].isoformat(),
                        from_phrase=text, to_phrase=text)
        return command_from_rules(data.get('text', ''), today)

    def close(self) -> None:
        pass


class FallbackModel:
    """Ask GigaChat first; if it fails, answer with rules so the assistant keeps working."""

    def __init__(self, primary, fallback: RulesModel | None = None):
        self._primary = primary
        self._fallback = fallback or RulesModel()
        self.healthy = True

    def complete_json(self, *args, **kwargs) -> dict:
        try:
            result = self._primary.complete_json(*args, **kwargs)
        except Exception as exc:
            self.healthy = False
            cause = exc.__cause__ or exc
            logger.warning('GigaChat недоступен (%s), разбираю команду правилами',
                           type(cause).__name__)
            return self._fallback.complete_json(*args, **kwargs)
        self.healthy = True
        return result

    def close(self) -> None:
        self._primary.close()
