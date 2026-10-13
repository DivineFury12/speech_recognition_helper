import datetime
import hashlib
import json
import re
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.command_extractor import CommandExtractor
from src.ports.conversations import ConversationRepository
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, Entry, EntryKind, EntryStatus, EntryUpdate
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import DialogueTransition, PendingStep, ReplyAct, ScheduleRole
from src.services.command_interpreter import CommandInterpreter
from src.services.dialogue_interpreter import DialogueInterpreter
from src.services.entries import EntryService
from src.services.temporal import (ambiguous_time_options, ambiguous_time_phrase,
                                   parse_date, parse_duration, parse_time,
                                   replacement_schedule_reply, resolve_time_clarification)


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
            if command.date_from is None or command.date_to is None:
                missing.append('период')
        elif command.action == AssistantIntent.search_entries and command.query is None:
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

    def _turn(
        self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
        answer: str, status: AssistantStatus, entries: list[Entry],
        pending: AssistantCommand | None = None,
    ) -> AssistantTurn:
        turn = AssistantTurn(
            conversation_id=conversation_id, turn_id=uuid.uuid4(), user_text=text,
            assistant_text=answer, status=status, intent=command.action, entries=entries,
        )
        if self._conversations is not None:
            if pending is not None and status == AssistantStatus.needs_clarification:
                pending = pending.model_copy(update={
                    'pending_step': self._pending_step(pending),
                })
            self._conversations.save_turn(turn, pending)
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

    @staticmethod
    def _candidate_question(entries: list[Entry], *, semantic: bool = False) -> str:
        choices = []
        for index, entry in enumerate(entries, 1):
            when = f' — {entry.date.isoformat()}' if entry.date else ''
            when += f' в {entry.time}' if entry.time else ''
            choices.append(f'{index}. {entry.title}{when}')
        if semantic and len(entries) == 1:
            return f'Вы имели в виду «{entries[0].title}»? Ответьте «да» или «нет».'
        return 'Какую запись вы имели в виду? Ответьте номером:\n' + '\n'.join(choices)

    @staticmethod
    def _has_update_value(command: AssistantCommand) -> bool:
        return any((command.new_title, command.date, command.time, command.duration_minutes))

    @classmethod
    def _needs_update_details(cls, command: AssistantCommand) -> bool:
        return (ambiguous_time_options(command.time_phrase) is not None
                and command.time is None) or not cls._has_update_value(command) or (
            command.move_intent and command.date is None and command.time is None
        )

    @staticmethod
    def _meeting_schedule(entry: Entry, *, default: bool = False) -> str:
        if entry.date is None or entry.time is None:
            return 'без даты и времени'
        duration = entry.duration_minutes or 60
        start = datetime.datetime.combine(entry.date, datetime.time.fromisoformat(entry.time))
        end = start + datetime.timedelta(minutes=duration)
        next_day = ' (+1 день)' if end.date() > start.date() else ''
        note = ' по умолчанию' if default else ''
        return (f'{entry.date.isoformat()} {entry.time}–{end:%H:%M}{next_day} '
                f'({duration} мин{note})')

    @staticmethod
    def _update_question(command: AssistantCommand, entry: Entry) -> str:
        options = ambiguous_time_options(command.time_phrase)
        if command.time is None and options:
            return f'Для «{entry.title}» уточните время: {options[0]} или {options[1]}?'
        if command.move_intent:
            when = entry.date.isoformat() if entry.date else 'без даты'
            if entry.time:
                when += f' в {entry.time}'
            if entry.kind == EntryKind.meeting:
                when += f', длительность {entry.duration_minutes or 60} мин'
            return (
                f'На какую дату или время перенести «{entry.title}»? '
                f'Сейчас: {when}. Если указать только дату, время сохранится.'
            )
        return f'Что изменить в записи «{entry.title}»: название, дату, время или длительность?'

    def _mutate(
        self, conversation_id: uuid.UUID, text: str, command: AssistantCommand,
        entry: Entry,
    ) -> AssistantTurn:
        if command.action == AssistantIntent.complete_entry:
            if entry.kind != EntryKind.task:
                return self._turn(conversation_id, text, command,
                                  'Отметить выполненной можно только задачу.',
                                  AssistantStatus.no_action, [])
            if entry.status == EntryStatus.completed:
                return self._turn(conversation_id, text, command,
                                  f'Задача уже выполнена: {entry.title}.',
                                  AssistantStatus.completed, [entry])
            updated = self._entries.update(entry.id, EntryUpdate(status=EntryStatus.completed))
            answer = f'Отмечено выполненным: {entry.title}.'
        elif command.action == AssistantIntent.update_entry:
            changes = {}
            if command.new_title is not None:
                changes['title'] = command.new_title
            if command.date is not None:
                changes['date'] = command.date
            if command.time is not None:
                changes['time'] = command.time
            if command.duration_minutes is not None:
                changes['duration_minutes'] = command.duration_minutes
            try:
                updated = self._entries.update(entry.id, EntryUpdate(**changes))
            except ValueError as exc:
                return self._turn(conversation_id, text, command, str(exc),
                                  AssistantStatus.no_action, [])
            if updated and command.move_intent:
                if updated.kind == EntryKind.meeting:
                    when = self._meeting_schedule(updated)
                else:
                    when = updated.date.isoformat() if updated.date else 'без даты'
                    if updated.time:
                        when += f' в {updated.time}'
                answer = f'Перенесено: {updated.title} — {when}.'
            else:
                answer = (f'Изменено: {updated.title} — {self._meeting_schedule(updated)}.'
                          if updated and updated.kind == EntryKind.meeting
                          else f'Изменено: {updated.title}.' if updated
                          else 'Запись больше не найдена.')
        else:
            if not self._entries.delete(entry.id):
                return self._turn(conversation_id, text, command,
                                  'Запись больше не найдена.', AssistantStatus.no_action, [])
            return self._turn(conversation_id, text, command,
                              f'Удалено: {entry.title}.', AssistantStatus.completed, [])
        if updated is None:
            return self._turn(conversation_id, text, command,
                              'Запись больше не найдена.', AssistantStatus.no_action, [])
        return self._turn(conversation_id, text, command,
                          answer, AssistantStatus.completed, [updated])

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
                 'conversation_id': str(conversation_id) if conversation_id else None},
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
            today = datetime.datetime.now(ZoneInfo(timezone)).date()
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise InvalidTimezoneError('Укажите корректный часовой пояс IANA') from exc

        conversation_id = conversation_id or uuid.uuid4()
        state = self._conversations.get(conversation_id) if self._conversations else None
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
        missing = self._missing(command)

        result_entries = []
        status = AssistantStatus.no_action
        pending = None
        if command.action == AssistantIntent.unknown:
            answer = 'Не удалось определить действие по запросу.'
        elif missing and (self._entries is None or self._conversations is None):
            answer = f"Не хватает данных: {', '.join(missing)}. Диалог уточнения не подключён."
        elif missing:
            answer = self._question(missing, command)
            status = AssistantStatus.needs_clarification
        elif self._entries is None:
            answer = 'Команда распознана, но запись и поиск пока не подключены. Действие не выполнено.'
        elif command.action == AssistantIntent.create_entry:
            entry, created = self._entries.create_once(
                kind=command.kind, title=command.title, date=command.date,
                time=command.time, timezone=timezone,
                duration_minutes=command.duration_minutes,
            )
            result_entries = [entry]
            answer = (
                f'Сохранено: {entry.title}.' if created
                else f'Такая запись уже есть: {entry.title}.'
            )
            if entry.kind == EntryKind.meeting:
                answer += (' ' + self._meeting_schedule(
                    entry, default=created and command.duration_minutes is None,
                ) + '.')
            status = AssistantStatus.completed
        elif command.action == AssistantIntent.list_entries:
            result_entries = self._entries.list_period(
                command.date_from, command.date_to, kind=command.kind,
            )
            answer = f'Найдено записей: {len(result_entries)}.'
            status = AssistantStatus.completed
        elif command.action == AssistantIntent.search_entries:
            result_entries = self._entries.search(command.query, kind=command.kind)
            answer = f'Найдено записей: {len(result_entries)}.'
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
