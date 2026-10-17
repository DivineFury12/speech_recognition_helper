import contextvars
import dataclasses
import datetime
import hashlib
import json
import logging
import re
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.command_extractor import CommandExtractor
from src.ports.conversations import ConversationRepository
from src.schemas.assistant import (AssistantIntent, AssistantStatus, AssistantTurn, Entry, EntryKind,
                                   EntryStatus, EntryUpdate)
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import DialogueTransition, PendingStep, ReplyAct, ScheduleRole
from src.services import phrasing
from src.services.command_interpreter import CommandInterpreter
from src.services.dialogue_interpreter import DialogueInterpreter
from src.services.entries import EntryService
from src.services.guards import (CHOICE_WORDS, DONE_RE, HELP_SUGGESTIONS, HELP_TEXT, PAST_WORDS_RE,
                                 CLEAR_PRIORITY_RE, IMPORTANT_RE, MARK_PRIORITY_RE, ONLY_THIS_RE,
                                 REMIND_FOR_RE, STOP_SERIES_RE, UNDO_RE, UNDO_TEXT, find_lead,
                                 find_shift, find_until, is_short_unclear, restore_numbers,
                                 yes_no)
from src.services.recurrence import find_recurrence, first_date, next_occurrence
from src.user_context import current_user
from src.services.temporal import (ambiguous_time_options, ambiguous_time_phrase,
                                   parse_date, parse_duration, parse_target_date, parse_time,
                                   replacement_schedule_reply, resolve_time_clarification)

SPLIT_RE = re.compile(
    r'\s*,?\s+(?:и|а\s+также|а\s+потом|потом|затем)\s+(?=(?:ещё\s+|еще\s+|мне\s+)?(?:поставь|создай|добавь|'
    r'напомни|удали|перенеси|сдвинь|отметь|покажи|найди|запланируй|запиши)\b)'
    r'|\s*;\s*|(?<=[.!?])\s+(?=[А-ЯЁ])',
    re.IGNORECASE,
)
_now: contextvars.ContextVar[datetime.datetime | None] = contextvars.ContextVar('assistant_now', default=None)
UNDO_FIELDS = ('title', 'date', 'time', 'duration_minutes', 'recurrence', 'status')
logger = logging.getLogger(__name__)


class InvalidTimezoneError(ValueError):
    pass


class AssistantService:
    """Interpret a message and execute supported entry operations."""

    def __init__(
        self, extractor: CommandExtractor, entries: EntryService | None = None,
        conversations: ConversationRepository | None = None,
        requests: object | None = None,
    ):
        self._extractor = extractor
        self._entries = entries
        self._conversations = conversations
        self._requests = requests
        self._dialogue = DialogueInterpreter(extractor, entries)
        self._command_interpreter = CommandInterpreter()

    @staticmethod
    def _missing(command: AssistantCommand) -> list[str]:
        missing = []
        if command.action == AssistantIntent.create_entry:
            if command.kind is None:
                missing.append('тип записи')
            if command.title is None:
                missing.append('название')
            if command.kind in (EntryKind.meeting, EntryKind.reminder):
                if command.date is None:
                    missing.append('дату')
                if command.time is None:
                    missing.append('время')
        elif command.action == AssistantIntent.list_entries:
            if (command.date_from is None or command.date_to is None) and command.query is None:
                missing.append('период')
        elif (command.action == AssistantIntent.search_entries and command.query is None
              and (command.date_from is None or command.date_to is None)):
            missing.append('поисковый запрос')
        return missing

    @staticmethod
    def _question(missing: list[str], command: AssistantCommand) -> str:
        options = ambiguous_time_options(command.time_phrase)
        if 'время' in missing and options:
            return f'Вы имели в виду {options[0]} или {options[1]}?'
        if 'дату' in missing and 'время' in missing:
            return 'На какую дату и время поставить запись?'
        questions = {
            'тип записи': 'Это задача, встреча или напоминание?',
            'название': 'Как назвать запись?',
            'дату': 'На какую дату поставить запись?',
            'время': 'На какое время поставить запись?',
            'период': 'За какой период показать записи?',
            'поисковый запрос': 'Что именно найти?',
        }
        return questions[missing[0]]

    @staticmethod
    def _now() -> datetime.datetime:
        return _now.get() or datetime.datetime.now()

    @classmethod
    def _today(cls) -> datetime.date:
        return cls._now().date()

    @staticmethod
    def _suggestions(status: AssistantStatus, intent: AssistantIntent, pending: AssistantCommand | None,
                     undo: dict | None, answer: str) -> list[str]:
        if status == AssistantStatus.needs_clarification and pending is not None:
            if pending.confirm or (pending.candidate_ids and len(pending.candidate_ids) == 1):
                return ['Да', 'Нет']
            if pending.candidate_ids:
                return list(CHOICE_WORDS[:min(len(pending.candidate_ids), len(CHOICE_WORDS))]) + ['Отмена']
            return ['Отмена']
        if status == AssistantStatus.completed and undo:
            return [UNDO_TEXT]
        if intent == AssistantIntent.unknown and answer == HELP_TEXT:
            return list(HELP_SUGGESTIONS)
        return []

    def _turn(
        self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
        answer: str, status: AssistantStatus, entries: list[Entry],
        pending: AssistantCommand | None = None, undo: dict | None = None,
    ) -> AssistantTurn:
        shown_pending = pending if status == AssistantStatus.needs_clarification else None
        turn = AssistantTurn(
            conversation_id=conversation_id, turn_id=uuid.uuid4(), user_text=text,
            assistant_text=answer, status=status, intent=command.action, entries=entries,
            suggestions=self._suggestions(status, command.action, shown_pending, undo, answer),
        )
        if self._conversations is not None:
            if pending is not None and status == AssistantStatus.needs_clarification:
                pending = pending.model_copy(update={
                    'pending_step': self._pending_step(pending),
                })
            self._conversations.save_turn(turn, pending)
            if undo is not None and hasattr(self._conversations, 'set_undo'):
                self._conversations.set_undo(conversation_id, undo or None)
        return turn

    @staticmethod
    def _pending_step(command: AssistantCommand) -> PendingStep:
        return DialogueInterpreter.pending_step(command)

    def _repeat_pending_question(self, command: AssistantCommand) -> str:
        if command.selected_entry_id is not None and self._entries is not None:
            entry = self._entries.get(command.selected_entry_id)
            if entry is not None:
                return self._update_question(command, entry)
        missing = self._missing(command)
        if missing:
            return self._question(missing, command)
        if command.action == AssistantIntent.update_entry:
            return 'Какую запись изменить?'
        if command.action in (AssistantIntent.delete_entry, AssistantIntent.complete_entry):
            return 'Какую запись вы имеете в виду?'
        return 'Уточните, пожалуйста, ваш ответ.'

    @classmethod
    def _candidate_question(cls, entries: list[Entry], *, semantic: bool = False) -> str:
        today = cls._today()
        choices = [f'{index}. {entry.title} — {phrasing.when(entry, today)}'
                   for index, entry in enumerate(entries, 1)]
        if semantic and len(entries) == 1:
            return (f'Вы имели в виду «{entries[0].title}» ({phrasing.when(entries[0], today)})? '
                    'Ответьте «да» или «нет».')
        return 'Какую запись вы имели в виду? Ответьте номером:\n' + '\n'.join(choices)

    @staticmethod
    def _has_update_value(command: AssistantCommand) -> bool:
        return any((command.new_title, command.date, command.time, command.duration_minutes,
                    command.shift_minutes))

    @classmethod
    def _needs_update_details(cls, command: AssistantCommand) -> bool:
        return (ambiguous_time_options(command.time_phrase) is not None
                and command.time is None) or not cls._has_update_value(command) or (
            command.move_intent and command.date is None and command.time is None
            and not command.shift_minutes
        )

    @classmethod
    def _meeting_schedule(cls, entry: Entry, *, default: bool = False) -> str:
        if entry.date is None or entry.time is None:
            return 'без даты и времени'
        note = ' (длительность по умолчанию — час)' if default else ''
        return phrasing.when(entry, cls._today()) + note

    @classmethod
    def _update_question(cls, command: AssistantCommand, entry: Entry) -> str:
        options = ambiguous_time_options(command.time_phrase)
        if command.time is None and options:
            return f'Для «{entry.title}» уточните время: {options[0]} или {options[1]}?'
        if command.move_intent:
            when = phrasing.when(entry, cls._today())
            if entry.kind == EntryKind.meeting:
                when += f', длительность {entry.duration_minutes or 60} мин'
            return (
                f'На какую дату или время перенести «{entry.title}»? '
                f'Сейчас: {when}. Если указать только дату, время сохранится.'
            )
        return f'Что изменить в записи «{entry.title}»: название, дату, время или длительность?'

    @classmethod
    def _past_note(cls, date: datetime.date | None, time: str | None) -> str | None:
        """Require confirmation before creating or moving a record into the past."""
        if date is None:
            return None
        now = cls._now()
        today = now.date()
        if date < today:
            return (f'{phrasing.format_date(date, today).capitalize()} — это '
                    f'{phrasing.format_day(date, today)}, этот день уже прошёл.')
        if date == today and time is not None and time < now.strftime('%H:%M'):
            return f'Сегодня в {time} — это время уже прошло.'
        return None

    def _conflict(self, kind: EntryKind | None, date: datetime.date | None, time: str | None,
                  duration: int | None, exclude: uuid.UUID | None = None) -> Entry | None:
        if kind != EntryKind.meeting or date is None or time is None or self._entries is None:
            return None
        start = datetime.datetime.combine(date, datetime.time.fromisoformat(time))
        end = start + datetime.timedelta(minutes=duration or 60)
        for other in self._entries.list_period(date, date, kind=EntryKind.meeting):
            if other.id == exclude or other.time is None or other.status == EntryStatus.completed:
                continue
            other_start = datetime.datetime.combine(date, datetime.time.fromisoformat(other.time))
            other_end = other_start + datetime.timedelta(minutes=other.duration_minutes or 60)
            if start < other_end and other_start < end:
                return other
        return None

    @staticmethod
    def _list_answer(command: AssistantCommand, entries: list[Entry], today: datetime.date) -> str:
        start, end = command.date_from, command.date_to
        if start == end:
            label = f'На {phrasing.format_date(start, today)}' if abs((start - today).days) <= 2 else \
                f'На {phrasing.format_day(start, today)}'
        else:
            label = f'С {phrasing.format_day(start, today)} по {phrasing.format_day(end, today)}'
        noun = {EntryKind.task: ('задача', 'задачи', 'задач'),
                EntryKind.reminder: ('напоминание', 'напоминания', 'напоминаний'),
                EntryKind.meeting: ('встреча', 'встречи', 'встреч')}.get(command.kind, ('запись', 'записи', 'записей'))
        if not entries:
            return f'{label} {noun[2]} нет.'
        lines = [phrasing.entry_line(e, today, show_date=start != end) for e in entries]
        return f'{label} {len(entries)} {phrasing.plural(len(entries), *noun)}:\n' + '\n'.join(lines)

    @staticmethod
    def _undo(op: str, entry: Entry) -> dict:
        return {'op': op, 'entry': entry.model_dump(mode='json')}

    def _mutate(
        self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
        entry: Entry,
    ) -> AssistantTurn:
        today = self._today()
        forced = set(command.forced or [])
        if command.action in (AssistantIntent.update_entry, AssistantIntent.delete_entry):
            only = self._only_this(conversation_id, text, command, entry)
            if only is not None:
                return only
        if command.action == AssistantIntent.complete_entry:
            if entry.kind == EntryKind.meeting:
                return self._turn(conversation_id, text, command,
                                  'Отметить выполненной можно задачу или напоминание.',
                                  AssistantStatus.no_action, [])
            if entry.status == EntryStatus.completed:
                return self._turn(conversation_id, text, command,
                                  f'Уже выполнено: «{entry.title}».',
                                  AssistantStatus.completed, [entry])
            stored = self._entries.get(entry.id) or entry
            if stored.recurrence is not None and stored.date is not None:
                base = max(stored.date, today)
                following = next_occurrence(stored.date, stored.recurrence, base)
                if following is not None:
                    moved = self._entries.update(entry.id, EntryUpdate(date=following))
                    answer = (f'Отметил: «{stored.title}» за {phrasing.format_date(base, today)}. '
                              f'Следующий раз — {phrasing.format_date(following, today)}.')
                    return self._turn(conversation_id, text, command, answer, AssistantStatus.completed,
                                      [moved], undo=self._undo('update', stored))
            updated = self._entries.update(entry.id, EntryUpdate(status=EntryStatus.completed))
            answer = f'Отметил выполненной: {phrasing.KIND_NOM[entry.kind]} «{entry.title}».'
            undo = self._undo('update', stored)
        elif command.action == AssistantIntent.update_entry:
            changes = {}
            if command.shift_minutes and entry.date is not None:
                start = datetime.datetime.combine(
                    entry.date, datetime.time.fromisoformat(entry.time) if entry.time else datetime.time(0, 0))
                moved = start + datetime.timedelta(minutes=command.shift_minutes)
                command = command.model_copy(update={
                    'date': moved.date(), 'time': moved.strftime('%H:%M') if entry.time else None,
                    'shift_minutes': None,
                })
            elif command.shift_minutes:
                return self._turn(conversation_id, text, command,
                                  f'У записи «{entry.title}» нет даты — назовите день и время.',
                                  AssistantStatus.no_action, [entry])
            if command.new_title is not None:
                changes['title'] = phrasing.capitalize(command.new_title)
            if command.date is not None:
                changes['date'] = command.date
            if command.time is not None:
                changes['time'] = command.time
            if command.duration_minutes is not None:
                changes['duration_minutes'] = command.duration_minutes
            new_date = changes.get('date', entry.date)
            new_time = changes.get('time', entry.time)
            if ('date' in changes or 'time' in changes) and 'past' not in forced:
                note = self._past_note(new_date, new_time)
                if note:
                    pending = command.model_copy(update={
                        'confirm': 'past', 'confirm_entry_id': entry.id, 'candidate_ids': None,
                    })
                    return self._turn(conversation_id, text, command,
                                      f'{note} Всё равно перенести? Или назовите другую дату.',
                                      AssistantStatus.needs_clarification, [entry], pending)
            if {'date', 'time', 'duration_minutes'} & set(changes) and 'conflict' not in forced:
                clash = self._conflict(entry.kind, new_date, new_time,
                                       changes.get('duration_minutes', entry.duration_minutes), entry.id)
                if clash is not None:
                    pending = command.model_copy(update={
                        'confirm': 'conflict', 'confirm_entry_id': entry.id, 'candidate_ids': None,
                    })
                    return self._turn(conversation_id, text, command,
                                      f'В это время уже есть {phrasing.describe(clash, today)}. '
                                      'Всё равно перенести? Или назовите другое время.',
                                      AssistantStatus.needs_clarification, [clash], pending)
            try:
                updated = self._entries.update(entry.id, EntryUpdate(**changes))
            except ValueError as exc:
                return self._turn(conversation_id, text, command, str(exc),
                                  AssistantStatus.no_action, [])
            undo = self._undo('update', entry)
            if updated is None:
                answer = 'Запись больше не найдена.'
            elif 'title' in changes and len(changes) == 1:
                answer = f'Готово, переименовал {phrasing.KIND_ACC[updated.kind]} в «{updated.title}».'
            else:
                answer = (f'Готово: {phrasing.KIND_NOM[updated.kind]} «{updated.title}» — '
                          f'теперь {phrasing.when(updated, today)}.')
                if 'duration_minutes' in changes and updated.kind == EntryKind.meeting:
                    answer = answer[:-1] + f', длительность {updated.duration_minutes} мин.'
        else:
            if 'delete' not in forced:
                pending = command.model_copy(update={
                    'confirm': 'delete', 'confirm_entry_id': entry.id, 'candidate_ids': None,
                })
                return self._turn(conversation_id, text, command,
                                  f'Удалить {phrasing.describe(entry, today, accusative=True)}?',
                                  AssistantStatus.needs_clarification, [entry], pending)
            if not self._entries.delete(entry.id):
                return self._turn(conversation_id, text, command,
                                  'Запись больше не найдена.', AssistantStatus.no_action, [])
            return self._turn(conversation_id, text, command,
                              f'Удалил {phrasing.KIND_ACC[entry.kind]} «{entry.title}».',
                              AssistantStatus.completed, [], undo=self._undo('delete', entry))
        if updated is None:
            return self._turn(conversation_id, text, command,
                              'Запись больше не найдена.', AssistantStatus.no_action, [])
        return self._turn(conversation_id, text, command,
                          answer, AssistantStatus.completed, [updated], undo=undo)

    def handle_message(
        self,
        text: str,
        timezone: str,
        conversation_id: uuid.UUID | None,
        request_id: uuid.UUID | None = None,
        request_fingerprint: str | None = None,
    ) -> AssistantTurn:
        if self._requests is not None:
            request_id = request_id or uuid.uuid4()
            fingerprint = request_fingerprint or hashlib.sha256(json.dumps(
                {'source': 'text', 'text': text, 'timezone': timezone,
                 'conversation_id': str(conversation_id) if conversation_id else None,
                 'user': current_user()},
                ensure_ascii=False, sort_keys=True,
            ).encode('utf-8')).hexdigest()
            return self._requests.execute_once(
                request_id, fingerprint, conversation_id,
                lambda: self._handle_message(text, timezone, conversation_id),
            )
        turn = self._handle_message(text, timezone, conversation_id)
        return turn.model_copy(update={'request_id': request_id}) if request_id else turn

    def _handle_message(
        self, text: str, timezone: str, conversation_id: uuid.UUID | None,
    ) -> AssistantTurn:
        try:
            zone = ZoneInfo(timezone)
            now = (_now.get() or datetime.datetime.now(zone)).replace(tzinfo=None)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise InvalidTimezoneError('Укажите корректный часовой пояс IANA') from exc
        token = _now.set(now)
        try:
            return self._handle_many(text, timezone, conversation_id, now.date())
        finally:
            _now.reset(token)

    def _handle_many(
        self, text: str, timezone: str, conversation_id: uuid.UUID | None, today: datetime.date,
    ) -> AssistantTurn:
        """«Поставь встречу завтра в 10 и напомни за час» — two commands, handled one after another."""
        parts = [p.strip(' ,.') for p in SPLIT_RE.split(text) if p and p.strip(' ,.')]
        pending = None
        if conversation_id is not None and self._conversations is not None:
            state = self._conversations.get(conversation_id)
            pending = state.pending_command if state else None
        if len(parts) < 2 or pending is not None:
            return self._handle_at(text, timezone, conversation_id, today)
        turns: list[AssistantTurn] = []
        for index, part in enumerate(parts):
            turn = self._handle_at(part, timezone, conversation_id, today)
            conversation_id = turn.conversation_id
            turns.append(turn)
            if turn.status == AssistantStatus.needs_clarification:
                rest = parts[index + 1:]
                if rest:
                    turns.append(turn.model_copy(update={
                        'assistant_text': 'Остальное (' + '; '.join(f'«{r}»' for r in rest)
                                          + ') скажите после ответа на вопрос.',
                    }))
                break
        last = turns[-1]
        entries = {e.id: e for t in turns for e in t.entries}
        statuses = [t.status for t in turns]
        status = (AssistantStatus.needs_clarification if AssistantStatus.needs_clarification in statuses
                  else AssistantStatus.completed if AssistantStatus.completed in statuses else last.status)
        return last.model_copy(update={
            'turn_id': uuid.uuid4(), 'user_text': text, 'status': status,
            'assistant_text': '\n'.join(t.assistant_text for t in turns),
            'entries': list(entries.values()),
        })

    def _guard(
        self, conversation_id: uuid.UUID, text: str, timezone: str,
        state,
    ) -> tuple[AssistantTurn | None, object]:
        """Our checks that run before the dialogue engine: undo, confirmations, «я купил»."""
        previous = state.pending_command if state is not None else None
        if UNDO_RE.match(text):
            return self._undo_last(conversation_id, text), state
        if previous is not None and previous.confirm:
            turn = self._on_confirm(conversation_id, text, timezone, previous)
            if turn is not None:
                return turn, state
            return None, dataclasses.replace(state, pending_command=None)
        if previous is None and yes_no(text) and len(text.split()) <= 2:
            command = AssistantCommand(action=AssistantIntent.unknown)
            return self._turn(conversation_id, text, command,
                              'Сейчас я ничего не спрашивал. Скажите команду целиком, '
                              'например «Удали встречу с Иваном».', AssistantStatus.no_action, []), state
        if previous is None and self._entries is not None:
            turn = self._series_and_lead(conversation_id, text, state)
            if turn is not None:
                return turn, state
        done = DONE_RE.match(text) if previous is None else None
        if done and self._entries is not None:
            matches, exact = self._entries.find_candidates(done.group('what'), kind=None)
            pending = [e for e in matches if e.status == EntryStatus.pending and e.kind != EntryKind.meeting]
            if exact and len(pending) == 1:
                command = AssistantCommand(action=AssistantIntent.complete_entry, target_query=done.group('what'))
                return self._mutate(conversation_id, text, command, pending[0]), state
        return None, state

    def _find_one(self, what: str, state, *, recurring: bool = False) -> Entry | None:
        """One entry by name, or the entry the dialogue is about when the user says «её»."""
        if re.fullmatch(r'\s*(?:неё|нее|него|неё же|этой|этого|неё)\s*', what.casefold()) or not what.strip():
            return self._entries.get(state.last_entry_id) if state and state.last_entry_id else None
        matches, _ = self._entries.find_candidates(what)
        matches = [e for e in matches if e.status != EntryStatus.completed and (e.recurrence or not recurring)]
        unique = {e.id: e for e in matches}
        return next(iter(unique.values())) if len(unique) == 1 else None

    def _series_and_lead(self, conversation_id: uuid.UUID, text: str, state) -> AssistantTurn | None:
        today = self._today()
        stop = STOP_SERIES_RE.match(text)
        if stop:
            entry = self._find_one(stop.group('what'), state, recurring=True)
            command = AssistantCommand(action=AssistantIntent.update_entry)
            if entry is None:
                return self._turn(conversation_id, text, command,
                                  f'Не нашёл повторяющуюся запись «{stop.group("what")}».', AssistantStatus.no_action, [])
            updated = self._entries.update(entry.id, EntryUpdate(recurrence_until=today))
            return self._turn(conversation_id, text, command,
                              f'Готово: «{updated.title}» больше не повторяется после '
                              f'{phrasing.format_day(today, today)}.',
                              AssistantStatus.completed, [updated], undo=self._undo('update', entry))
        mark = MARK_PRIORITY_RE.match(text) or CLEAR_PRIORITY_RE.match(text)
        if mark:
            entry = self._find_one(mark.group('what'), state)
            command = AssistantCommand(action=AssistantIntent.update_entry)
            if entry is None:
                return self._turn(conversation_id, text, command,
                                  f'Не нашёл запись «{mark.group("what")}».', AssistantStatus.no_action, [])
            level = mark.groupdict().get('level') or ''
            high = bool(level) and not re.match(r'(?i)(?:обычн|неважн|не\s)', level)
            updated = self._entries.update(entry.id, EntryUpdate(priority='high' if high else 'normal'))
            return self._turn(conversation_id, text, command,
                              f'Готово: «{updated.title}» — ' + ('важное.' if high else 'обычный приоритет.'),
                              AssistantStatus.completed, [updated], undo=self._undo('update', entry))
        lead_for = REMIND_FOR_RE.match(text)
        bare_lead = re.fullmatch(r'\s*(?:и\s+)?напомни\w*(?:\s+мне)?(?:\s+об\s+этом)?\s+за\s+[^,]+?\s*[.!]*', text,
                                 re.IGNORECASE)
        if lead_for or bare_lead:
            minutes = find_lead(text)
            if minutes is None:
                return None
            entry = self._find_one(lead_for.group('what') if lead_for else '', state)
            command = AssistantCommand(action=AssistantIntent.update_entry)
            if entry is None:
                return None if lead_for is None else self._turn(
                    conversation_id, text, command, 'Не понял, перед какой записью напомнить. Назовите её.',
                    AssistantStatus.no_action, [])
            if entry.time is None:
                return self._turn(conversation_id, text, command,
                                  f'У записи «{entry.title}» нет времени — назовите время, и я напомню заранее.',
                                  AssistantStatus.no_action, [entry])
            updated = self._entries.update(entry.id, EntryUpdate(remind_minutes=minutes))
            return self._turn(conversation_id, text, command,
                              f'Хорошо, напомню {phrasing.lead(minutes)} до «{updated.title}» '
                              f'({phrasing.when(updated, today)}).',
                              AssistantStatus.completed, [updated], undo=self._undo('update', entry))
        return None

    def _only_this(self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
                   entry: Entry) -> AssistantTurn | None:
        """«Перенеси только эту планёрку» / «удали только завтрашнюю»: one occurrence of a series."""
        stored = self._entries.get(entry.id) or entry
        forced = set(command.forced or [])
        if stored.recurrence is None or not (ONLY_THIS_RE.search(text) or 'only' in forced):
            return None
        today = self._today()
        day = command.target_date or (entry.date if entry.date and entry.date >= today and entry.date != stored.date
                                      else None)
        if day is None:
            upcoming = self._entries.list_period(today, today + datetime.timedelta(days=400))
            day = next((e.date for e in upcoming if e.id == stored.id), None)
        if day is None:
            return None
        skipped = sorted(set(stored.exdates or []) | {day})
        if command.action == AssistantIntent.delete_entry and 'delete' not in set(command.forced or []):
            pending = command.model_copy(update={'confirm': 'delete', 'confirm_entry_id': stored.id,
                                                 'candidate_ids': None, 'target_date': day,
                                                 'forced': sorted(forced | {'only'})})
            return self._turn(conversation_id, text, command,
                              f'Удалить только «{stored.title}» за {phrasing.format_date(day, today)}? '
                              'Остальные повторы останутся.', AssistantStatus.needs_clarification, [stored], pending)
        if command.action == AssistantIntent.delete_entry:
            updated = self._entries.update(stored.id, EntryUpdate(exdates=skipped))
            return self._turn(conversation_id, text, command,
                              f'Убрал «{stored.title}» за {phrasing.format_date(day, today)}, остальные повторы остались.',
                              AssistantStatus.completed, [updated], undo=self._undo('update', stored))
        if command.action != AssistantIntent.update_entry:
            return None
        once = stored.model_copy(update={'date': day})
        new_date = command.date or day
        new_time = command.time or stored.time
        if command.shift_minutes:
            start = datetime.datetime.combine(day, datetime.time.fromisoformat(stored.time or '00:00'))
            moved = start + datetime.timedelta(minutes=command.shift_minutes)
            new_date, new_time = moved.date(), moved.strftime('%H:%M') if stored.time else None
        self._entries.update(stored.id, EntryUpdate(exdates=skipped))
        single = self._entries.create(
            kind=stored.kind, title=phrasing.capitalize(command.new_title or stored.title), date=new_date,
            time=new_time, timezone=stored.timezone, duration_minutes=command.duration_minutes or stored.duration_minutes,
        )
        return self._turn(conversation_id, text, command,
                          f'Готово: только этот раз «{once.title}» теперь {phrasing.when(single, today)}, '
                          'остальные повторы без изменений.',
                          AssistantStatus.completed, [single], undo=self._undo('update', stored))

    def _on_confirm(self, conversation_id: uuid.UUID, text: str, timezone: str,
                    previous: AssistantCommand) -> AssistantTurn | None:
        answer = yes_no(text)
        kind = previous.confirm
        base = previous.model_copy(update={'confirm': None, 'confirm_entry_id': None})
        entry = self._entries.get(previous.confirm_entry_id) if previous.confirm_entry_id else None
        new_date = parse_date(text, self._today()) if kind in ('past', 'conflict') else None
        new_time = parse_time(text) if kind in ('past', 'conflict') else None
        if new_date or new_time:
            changed = base.model_copy(update={'date': new_date or base.date, 'time': new_time or base.time})
            if changed.action == AssistantIntent.create_entry:
                return self._create(conversation_id, text, changed, timezone)
            if entry is not None:
                return self._mutate(conversation_id, text, changed, entry)
        if answer == 'yes':
            forced = base.model_copy(update={'forced': sorted(set(base.forced or []) | {kind})})
            if forced.action == AssistantIntent.create_entry:
                return self._create(conversation_id, text, forced, timezone)
            if entry is None:
                return self._turn(conversation_id, text, base, 'Этой записи уже нет.',
                                  AssistantStatus.no_action, [])
            return self._mutate(conversation_id, text, forced, entry)
        if answer == 'no':
            verb = {'delete': 'не удаляю', 'past': 'не ставлю', 'conflict': 'не ставлю'}[kind]
            if base.action == AssistantIntent.update_entry and kind != 'delete':
                verb = 'не переношу'
            return self._turn(conversation_id, text, base, f'Хорошо, {verb}.', AssistantStatus.no_action, [])
        if is_short_unclear(text):
            last = next((m.text for m in reversed(self._conversations.get(conversation_id).messages)
                         if m.role == 'assistant'), '') if self._conversations else ''
            question = last.removeprefix('Не понял ответ. ').split(' Ответьте «да»')[0]
            return self._turn(conversation_id, text, base,
                              f'Не понял ответ. {question} Ответьте «да» или «нет».',
                              AssistantStatus.needs_clarification, [entry] if entry else [], previous)
        return None

    def _undo_last(self, conversation_id: uuid.UUID, text: str) -> AssistantTurn:
        record = (self._conversations.get_undo(conversation_id)
                  if hasattr(self._conversations, 'get_undo') else None)
        nothing = AssistantCommand(action=AssistantIntent.unknown)
        if not record or self._entries is None:
            return self._turn(conversation_id, text, nothing,
                              'Отменять нечего: в этом диалоге я ещё ничего не менял.', AssistantStatus.no_action, [])
        saved = Entry.model_validate(record['entry'])
        today = self._today()
        if record['op'] == 'create':
            gone = self._entries.delete(saved.id)
            command = AssistantCommand(action=AssistantIntent.delete_entry)
            answer = (f'Отменил: убрал {phrasing.KIND_ACC[saved.kind]} «{saved.title}».' if gone
                      else 'Этой записи уже нет, отменять нечего.')
            return self._turn(conversation_id, text, command, answer,
                              AssistantStatus.completed if gone else AssistantStatus.no_action, [], undo={})
        if record['op'] == 'delete':
            restored = self._entries.restore(saved) if self._entries.get(saved.id) is None else saved
            command = AssistantCommand(action=AssistantIntent.create_entry)
            return self._turn(conversation_id, text, command,
                              f'Вернул {phrasing.describe(restored, today, accusative=True)}.',
                              AssistantStatus.completed, [restored], undo={})
        if self._entries.get(saved.id) is None:
            return self._turn(conversation_id, text, nothing, 'Этой записи уже нет, отменять нечего.',
                              AssistantStatus.no_action, [], undo={})
        fields = {name: getattr(saved, name) for name in UNDO_FIELDS}
        fields = {k: v for k, v in fields.items() if v is not None or k in ('date', 'time', 'recurrence')}
        try:
            restored = self._entries.update(saved.id, EntryUpdate(**fields))
        except ValueError as exc:
            return self._turn(conversation_id, text, nothing, f'Не получилось вернуть: {exc}',
                              AssistantStatus.no_action, [])
        command = AssistantCommand(action=AssistantIntent.update_entry)
        return self._turn(conversation_id, text, command,
                          f'Вернул как было: {phrasing.describe(restored, today)}.',
                          AssistantStatus.completed, [restored], undo={})

    def _create(self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
                timezone: str) -> AssistantTurn:
        today = self._today()
        forced = set(command.forced or [])
        title = phrasing.capitalize(command.title)
        if command.recurrence is not None and command.date is None:
            command = command.model_copy(update={'date': first_date(command.recurrence, today)})
        if 'past' not in forced and command.recurrence is None:
            note = self._past_note(command.date, command.time)
            if note:
                pending = command.model_copy(update={'confirm': 'past'})
                return self._turn(conversation_id, text, command,
                                  f'{note} Всё равно поставить на эту дату? Или назовите другую.',
                                  AssistantStatus.needs_clarification, [], pending)
        if 'conflict' not in forced:
            clash = self._conflict(command.kind, command.date, command.time, command.duration_minutes)
            if clash is not None:
                pending = command.model_copy(update={'confirm': 'conflict'})
                return self._turn(conversation_id, text, command,
                                  f'В это время уже есть {phrasing.describe(clash, today)}. '
                                  'Всё равно поставить? Или назовите другое время.',
                                  AssistantStatus.needs_clarification, [clash], pending)
        extra = {name: getattr(command, name) for name in ('recurrence', 'remind_minutes', 'recurrence_until',
                                                            'priority')
                 if getattr(command, name) is not None}
        entry, created = self._entries.create_once(
            kind=command.kind, title=title, date=command.date,
            time=command.time, timezone=timezone,
            duration_minutes=command.duration_minutes, **extra,
        )
        if not created:
            return self._turn(conversation_id, text, command,
                              f'Такая запись уже есть: {phrasing.describe(entry, today)}.',
                              AssistantStatus.completed, [entry])
        answer = f'Готово, создал {phrasing.KIND_ACC[entry.kind]}: «{entry.title}», {phrasing.when(entry, today)}.'
        if entry.recurrence_until:
            answer = answer[:-1] + f', до {phrasing.format_day(entry.recurrence_until, today)}.'
        if entry.remind_minutes:
            answer = answer[:-1] + f'. Напомню {phrasing.lead(entry.remind_minutes)}.'
        if getattr(entry.priority, 'value', entry.priority) == 'high':
            answer = answer[:-1] + '. Отметил как важное.'
        if entry.kind == EntryKind.meeting and command.duration_minutes is None and entry.time:
            answer = answer[:-1] + ' (длительность по умолчанию — час).'
        return self._turn(conversation_id, text, command, answer, AssistantStatus.completed, [entry],
                          undo=self._undo('create', entry))

    def _handle_at(
        self, text: str, timezone: str, conversation_id: uuid.UUID | None, today: datetime.date,
    ) -> AssistantTurn:
        if (conversation_id is not None and self._conversations is not None
                and hasattr(self._conversations, 'is_foreign')
                and self._conversations.is_foreign(conversation_id)):
            conversation_id = None
        conversation_id = conversation_id or uuid.uuid4()
        state = self._conversations.get(conversation_id) if self._conversations else None
        if self._conversations is not None:
            guarded, state = self._guard(conversation_id, text, timezone, state)
            if guarded is not None:
                return guarded
        plan = self._dialogue.plan(text, timezone, today, state)
        previous = plan.previous
        pending_step = plan.pending_step
        pending_reply = plan.reply
        transition = plan.transition
        preextracted = plan.preextracted
        future_task = plan.task_fallback
        if previous is not None and transition == DialogueTransition.cancel:
            turn = AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text=text, assistant_text='Хорошо, отменил эту команду.',
                status=AssistantStatus.no_action, intent=AssistantIntent.unknown,
                entries=[],
            )
            self._conversations.save_turn(turn, None)
            return turn
        starts_new_request = transition == DialogueTransition.new_command
        if (previous is not None and transition == DialogueTransition.repeat_question
                and not previous.candidate_ids):
            return self._turn(
                conversation_id, text, previous,
                self._repeat_pending_question(previous),
                AssistantStatus.needs_clarification, [], previous,
            )
        if previous is not None and previous.candidate_ids and not starts_new_request:
            candidates = [self._entries.get(entry_id) for entry_id in previous.candidate_ids]
            candidates = [entry for entry in candidates if entry is not None]
            choice = self._dialogue.resolve_candidate(
                text, candidates, previous, pending_reply, today,
            )
            if choice.rejected:
                return self._turn(
                    conversation_id, text, previous, 'Хорошо, запись не менял.',
                    AssistantStatus.no_action, [],
                )
            selected = choice.selected
            if selected is not None:
                selected_command = previous.model_copy(update={
                    'candidate_ids': None, 'selected_entry_id': selected.id,
                })
                if pending_reply is not None:
                    replacement_schedule = (
                        pending_reply.act == ReplyAct.select
                        and pending_reply.schedule_role == ScheduleRole.replacement
                    )
                else:
                    replacement_schedule = bool(re.match(
                        r'^\s*(?:(?:(?:выбери|номер|под номером)\s+)?\d{1,2}|'
                        r'перв\w*|втор\w*|трет\w*|четверт\w*|пят\w*)\b',
                        text.casefold(),
                    ))
                if selected_command.action == AssistantIntent.update_entry and replacement_schedule:
                    selected_command = selected_command.model_copy(update={
                        'date': parse_date(text, today) or selected_command.date,
                        'time': parse_time(text) or selected_command.time,
                        'time_phrase': ambiguous_time_phrase(text) or selected_command.time_phrase,
                        'duration_minutes': (
                            parse_duration(text, for_update=True)
                            or selected_command.duration_minutes
                        ),
                    })
                if (selected_command.action == AssistantIntent.update_entry
                        and self._needs_update_details(selected_command)):
                    return self._turn(
                        conversation_id, text, selected_command,
                        self._update_question(selected_command, selected),
                        AssistantStatus.needs_clarification, [selected], selected_command,
                    )
                if selected_command.action == AssistantIntent.delete_entry:
                    selected_command = selected_command.model_copy(update={
                        'forced': sorted(set(selected_command.forced or []) | {'delete'}),
                    })
                return self._mutate(conversation_id, text, selected_command, selected)
            if not candidates:
                return self._turn(
                    conversation_id, text, previous, 'Эти записи больше не найдены.',
                    AssistantStatus.no_action, [],
                )
            pending_candidate = previous
            schedule_note = ''
            if previous.action == AssistantIntent.update_entry and previous.move_intent:
                proposed = replacement_schedule_reply(text, today)
                if proposed is not None:
                    date, time, uncertain_time = proposed
                    pending_candidate = previous.model_copy(update={
                        'date': date or previous.date,
                        'time': time or previous.time,
                        'time_phrase': uncertain_time or previous.time_phrase,
                    })
                    schedule_note = 'Новую дату или время запомнил. '
            return self._turn(
                conversation_id, text, pending_candidate,
                schedule_note + self._candidate_question(candidates,
                                                         semantic=len(candidates) == 1),
                AssistantStatus.needs_clarification, candidates, pending_candidate,
            )
        history = (
            [{'role': item.role, 'text': item.text} for item in state.messages]
            if state else []
        )
        resolved_time = None
        if previous is not None and pending_step == PendingStep.time_choice:
            options = ambiguous_time_options(previous.time_phrase)
            if (pending_reply is not None and pending_reply.act == ReplyAct.select
                    and pending_reply.choice in (1, 2)):
                resolved_time = options[pending_reply.choice - 1]
            elif pending_reply is None or pending_reply.act == ReplyAct.continue_command:
                resolved_time = resolve_time_clarification(previous.time_phrase, text)
        if resolved_time:
            command = AssistantCommand(
                action=previous.action, date=parse_date(text, today), time=resolved_time,
            )
        elif preextracted is not None:
            command = preextracted
        else:
            command = self._extractor.extract(
                text=text, timezone=timezone, today=today,
                previous=previous, history=history,
            )
        resolved = self._command_interpreter.resolve(
            command, text=text, today=today, previous=previous, state=state,
            continuation=(pending_reply is not None
                          and pending_reply.act == ReplyAct.continue_command),
            resolved_time=resolved_time, task_fallback=future_task,
        )
        command = resolved.command
        previous = resolved.previous
        explicit_target = resolved.explicit_target
        pronoun_reference = resolved.pronoun_reference
        if command.action == AssistantIntent.create_entry and command.source_reference_phrase:
            source = (self._entries.get(state.last_entry_id)
                      if self._entries is not None and state is not None
                      and state.last_entry_id is not None else None)
            if source is None:
                return self._turn(
                    conversation_id, text, command,
                    'Не вижу запись, данные которой нужно повторить. Назовите её или создайте новую запись полностью.',
                    AssistantStatus.no_action, [],
                )
            kind = command.kind or source.kind
            command = command.model_copy(update={
                'kind': kind,
                'title': command.title or source.title,
                'date': command.date or source.date,
                'time': command.time or source.time,
                'duration_minutes': ((command.duration_minutes or source.duration_minutes)
                                     if kind == EntryKind.meeting else None),
            })
        if (command.action in (AssistantIntent.update_entry, AssistantIntent.delete_entry)
                and ONLY_THIS_RE.search(text)):
            # Keep «только эту» through clarifications: the reply «да» no longer contains the word.
            command = command.model_copy(update={'forced': sorted(set(command.forced or []) | {'only'})})
        if command.action == AssistantIntent.update_entry:
            shift = find_shift(text)
            if shift:
                # «на 2 дня» is a shift, not «в 2 часа дня»: the explicit shift phrase wins.
                command = command.model_copy(update={'shift_minutes': shift, 'move_intent': True,
                                                     'date': None, 'time': None, 'time_phrase': None})
        if command.action == AssistantIntent.create_entry:
            grounded = {}
            restored = restore_numbers(command.title, text)
            if restored != command.title:
                grounded['title'] = restored
            rule = find_recurrence(text)
            if rule is not None:
                grounded['recurrence'] = rule
                until = find_until(text, today)
                if until is not None:
                    grounded['recurrence_until'] = until
            lead = find_lead(text)
            if lead is not None:
                grounded['remind_minutes'] = lead
            if IMPORTANT_RE.search(text):
                grounded['priority'] = 'high'
                if command.title and IMPORTANT_RE.search(command.title):
                    cleaned = ' '.join(IMPORTANT_RE.sub(' ', command.title).split())
                    grounded['title'] = cleaned or command.title
            if command.date is None and PAST_WORDS_RE.search(text):
                past = parse_target_date(text, today)
                if past is not None and past < today:
                    grounded['date'] = past
                    grounded['forced'] = sorted(set(command.forced or []) | {'past_words'})
            if grounded:
                command = command.model_copy(update=grounded)
        missing = self._missing(command)

        result_entries = []
        status = AssistantStatus.no_action
        pending = None
        if command.action == AssistantIntent.unknown:
            answer = HELP_TEXT
        elif missing and (self._entries is None or self._conversations is None):
            answer = f"Не хватает данных: {', '.join(missing)}. Диалог уточнения не подключён."
        elif missing:
            answer = self._question(missing, command)
            status = AssistantStatus.needs_clarification
        elif self._entries is None:
            answer = 'Команда распознана, но запись и поиск пока не подключены. Действие не выполнено.'
        elif command.action == AssistantIntent.create_entry:
            return self._create(conversation_id, text, command, timezone)
        elif command.action in (AssistantIntent.list_entries, AssistantIntent.search_entries):
            has_period = command.date_from is not None and command.date_to is not None
            if command.query:
                result_entries = self._entries.search(
                    command.query, kind=command.kind,
                    date_from=command.date_from, date_to=command.date_to,
                )
                classify_relevant = getattr(self._extractor, 'classify_relevant_entries', None)
                if result_entries and callable(classify_relevant):
                    try:
                        result_entries = classify_relevant(command.query, result_entries)
                    except Exception:
                        logger.warning('Search relevance classifier unavailable', exc_info=True)
                        return self._turn(
                            conversation_id, text, command,
                            'Не удалось проверить результаты поиска по теме. Попробуйте ещё раз.',
                            AssistantStatus.no_action, [],
                        )
            else:
                result_entries = self._entries.list_period(
                    command.date_from, command.date_to, kind=command.kind,
                )
            if has_period:
                answer = self._list_answer(command, result_entries, today)
                if command.query:
                    answer = f'По теме «{command.query}». {answer}'
            else:
                answer = ('Нашёл:\n' + '\n'.join(phrasing.entry_line(e, today, show_date=True)
                                                 for e in result_entries)
                          if result_entries else f'Не нашёл ничего похожего на «{command.query}».')
            status = AssistantStatus.completed
        elif command.action in (
            AssistantIntent.complete_entry, AssistantIntent.update_entry,
            AssistantIntent.delete_entry,
        ):
            if self._conversations is None:
                answer = 'Диалог для изменения записей не подключён. Действие не выполнено.'
            else:
                candidates = []
                exact = False
                if command.selected_entry_id is not None:
                    selected = self._entries.get(command.selected_entry_id)
                    if selected is not None:
                        candidates, exact = [selected], True
                elif (explicit_target or not pronoun_reference and
                      (command.target_query is not None or command.target_date is not None)):
                    candidates, exact = self._entries.find_candidates(
                        command.target_query,
                        kind=EntryKind.task if command.action == AssistantIntent.complete_entry else command.kind,
                        date=command.target_date, time=command.target_time,
                    )
                    if command.action == AssistantIntent.complete_entry:
                        pending_tasks = [entry for entry in candidates if entry.status == EntryStatus.pending]
                        if pending_tasks:
                            candidates = pending_tasks
                elif state is not None and state.last_entry_id is not None and pronoun_reference:
                    entry = self._entries.get(state.last_entry_id)
                    if entry is not None:
                        candidates, exact = [entry], True
                elif state is not None and state.last_entry_id is not None and re.search(
                    r'\bпоследн\w*\b', text.casefold(),
                ):
                    entry = self._entries.get(state.last_entry_id)
                    if entry is not None:
                        candidates, exact = [entry], True
                if not candidates:
                    answer = ('Запись не найдена.' if command.target_query or command.target_date
                              else 'Какую запись изменить?' if command.action == AssistantIntent.update_entry
                              else 'Какую запись вы имеете в виду?')
                    status = AssistantStatus.needs_clarification
                    pending = command
                elif len(candidates) == 1 and exact:
                    selected = candidates[0]
                    if (command.action == AssistantIntent.update_entry
                            and self._needs_update_details(command)):
                        pending = command.model_copy(update={'selected_entry_id': selected.id})
                        answer = self._update_question(pending, selected)
                        status = AssistantStatus.needs_clarification
                        result_entries = [selected]
                    else:
                        return self._mutate(conversation_id, text, command, selected)
                else:
                    answer = self._candidate_question(candidates, semantic=not exact)
                    status = AssistantStatus.needs_clarification
                    result_entries = candidates
                    pending = command.model_copy(update={
                        'candidate_ids': [entry.id for entry in candidates],
                    })
        else:
            answer = 'Не удалось выполнить команду.'
        return self._turn(
            conversation_id, text, command, answer, status, result_entries,
            pending or (command if status == AssistantStatus.needs_clarification else None),
        )
