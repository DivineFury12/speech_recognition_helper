"""Decide how an incoming turn relates to the pending dialogue."""

import dataclasses
import datetime
import logging
import re

from src.ports.command_extractor import CommandExtractor
from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, Entry, EntryKind
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import (DialogueTransition, PendingReply, PendingStep,
                                  ReplyAct, ScheduleRole, transition_for_reply)
from src.services.entries import EntryService
from src.services.selection import selection_is_grounded
from src.services.temporal import (ambiguous_time_options, parse_date, parse_period,
                                   parse_target_date, parse_time, replacement_schedule_reply)


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DialoguePlan:
    previous: AssistantCommand | None
    pending_step: PendingStep | None
    reply: PendingReply | None
    transition: DialogueTransition | None
    preextracted: AssistantCommand | None
    task_fallback: AssistantCommand | None


@dataclasses.dataclass(frozen=True)
class CandidateResolution:
    selected: Entry | None = None
    rejected: bool = False


class DialogueInterpreter:
    """Classify a reply once, then verify its effect against the pending state."""

    def __init__(self, extractor: CommandExtractor, entries: EntryService | None):
        self._extractor = extractor
        self._entries = entries

    @staticmethod
    def pending_step(command: AssistantCommand) -> PendingStep:
        if command.candidate_ids:
            return PendingStep.candidate
        if command.time is None and ambiguous_time_options(command.time_phrase):
            return PendingStep.time_choice
        if command.selected_entry_id:
            return PendingStep.update_details
        return PendingStep.fields

    @staticmethod
    def _short_confirmation(text: str) -> bool | None:
        """A whole short answer may confirm one proposal; a new command may not."""
        words = ' '.join(re.findall(r'[а-яa-z0-9]+', text.casefold().replace('ё', 'е')))
        affirmative = {
            'да', 'ага', 'угу', 'именно', 'точно', 'верно', 'правильно',
            'все верно', 'все правильно', 'вот именно', 'так точно',
            'это она', 'это он', 'это оно', 'это та', 'это тот',
            'эту', 'его', 'ее', 'подтверждаю', 'согласен', 'согласна',
            'конечно', 'ок', 'окей', 'okay', 'yes', 'хорошо',
            'ладно', 'давай', 'давайте', 'подходит',
        }
        negative = {
            'нет', 'не та', 'не тот', 'не то', 'не она', 'не он',
            'не оно', 'не эту', 'не его', 'не ее', 'неверно',
            'неправильно', 'нет спасибо', 'нет не та', 'нет не тот',
        }
        if words in affirmative or (words.startswith('да ') and words[3:] in affirmative):
            return True
        if words in negative:
            return False
        return None

    def _model_confirmation(self, text: str, title: str) -> bool | None:
        if len(text) > 160 or len(text.split()) > 16 or re.search(
            r'\b(?:создай|создать|добавь|добавить|удали|удалить|перенеси|'
            r'перенести|измени|изменить|покажи|найди|отметь)\b',
            text.casefold(),
        ):
            return None
        classify = getattr(self._extractor, 'classify_confirmation', None)
        if not callable(classify):
            return None
        try:
            decision = classify(text, title)
            if decision is True and re.search(r'\b(?:не|нет|ни)\b', text.casefold()):
                return None
            return decision
        except Exception:
            logger.warning('Confirmation classifier unavailable', exc_info=True)
            return None

    @classmethod
    def _grounded_candidate(
        cls, text: str, entries: list[Entry], today: datetime.date,
        *, allow_schedule_match: bool,
    ) -> Entry | None:
        value = text.strip().casefold().replace('ё', 'е')
        if re.match(r'^(?:нет[,.\s]+)?не\s+', value):
            return None
        number = re.match(
            r'^(?:(?:выбери|номер|под номером)\s+)?(?P<number>\d{1,2})(?=$|[\s,.])',
            value,
        )
        ordinals = {
            'первую': 0, 'первый': 0, 'первая': 0, 'первое': 0, 'первой': 0,
            'вторую': 1, 'второй': 1, 'вторая': 1, 'второе': 1,
            'третью': 2, 'третий': 2, 'третья': 2, 'третье': 2,
            'четвертую': 3, 'четвертый': 3, 'четвертая': 3,
            'пятую': 4, 'пятый': 4, 'пятая': 4,
        }
        for word in re.findall(r'[а-я]+', value):
            if word in ordinals:
                index = ordinals[word]
                return entries[index] if index < len(entries) else None
        if number:
            remainder = value[number.end():].strip(' ,.')
            if not remainder or re.match(r'^(?:на|в|к|перенеси|поставь)\b', remainder):
                index = int(number.group('number')) - 1
                return entries[index] if 0 <= index < len(entries) else None
        if len(entries) == 1 and cls._short_confirmation(text) is True:
            return entries[0]
        if not allow_schedule_match:
            return None
        selected_date = parse_target_date(text, today)
        selected_time = parse_time(text)
        if selected_date is not None or selected_time is not None:
            matches = [entry for entry in entries if
                       (selected_date is None or entry.date == selected_date) and
                       (selected_time is None or entry.time == selected_time)]
            if len(matches) == 1:
                return matches[0]
        return None

    def resolve_candidate(
        self, text: str, candidates: list[Entry], pending: AssistantCommand,
        reply: PendingReply | None, today: datetime.date,
    ) -> CandidateResolution:
        """Resolve a pending candidate choice before the service changes a record."""
        if not candidates:
            return CandidateResolution()
        if ((reply is None and self._short_confirmation(text) is False)
                or (reply is not None and reply.act == ReplyAct.reject
                    and len(candidates) == 1)):
            return CandidateResolution(rejected=True)
        negated = bool(re.search(r'\b(?:не|нет|ни)\b', text.casefold()))
        if reply is not None:
            if reply.act == ReplyAct.confirm and len(candidates) == 1 and not negated:
                return CandidateResolution(selected=candidates[0])
            if reply.act == ReplyAct.select and not negated:
                index = reply.choice - 1
                if 0 <= index < len(candidates):
                    return CandidateResolution(selected=candidates[index])
            if reply.act != ReplyAct.unclear:
                return CandidateResolution()
            # The model was unsure; a plain ordinal or number in the reply is still unambiguous.
            grounded = self._grounded_candidate(text, candidates, today, allow_schedule_match=False)
            return CandidateResolution(selected=grounded)
        old_schedule_reference = bool(re.search(
            r'\b(?:ту|тот|которая|который|сейчас|из)\b', text.casefold(),
        ))
        selected = self._grounded_candidate(
            text, candidates, today,
            allow_schedule_match=not pending.move_intent or old_schedule_reference,
        )
        if selected is not None:
            return CandidateResolution(selected=selected)
        if len(candidates) == 1:
            decision = self._model_confirmation(text, candidates[0].title)
            return CandidateResolution(selected=candidates[0] if decision is True else None,
                                       rejected=decision is False)
        return CandidateResolution()

    def _classify(
        self, text: str, command: AssistantCommand, state: ConversationState,
        today: datetime.date,
    ) -> PendingReply | None:
        classify = getattr(self._extractor, 'classify_pending_reply', None)
        if not callable(classify):
            return None
        step = self.pending_step(command)
        candidates = []
        if command.candidate_ids and self._entries is not None:
            for number, entry_id in enumerate(command.candidate_ids, 1):
                entry = self._entries.get(entry_id)
                if entry is not None:
                    candidates.append({
                        'number': number, 'title': entry.title,
                        'date': entry.date.isoformat() if entry.date else None,
                        'time': entry.time,
                    })
        question = (
            state.messages[-1].text
            if state.messages and state.messages[-1].role == 'assistant' else None
        )
        try:
            reply = classify(
                text, step=step, action=command.action,
                question=question, candidates=candidates, pending=command,
            )
            if reply.act == ReplyAct.select:
                if step == PendingStep.time_choice:
                    valid = reply.choice in (1, 2) and reply.schedule_role == ScheduleRole.none
                elif step == PendingStep.candidate:
                    valid = reply.choice is not None and 1 <= reply.choice <= len(candidates)
                    if valid and self._entries is not None:
                        entry_options = [self._entries.get(entry_id)
                                         for entry_id in command.candidate_ids]
                        valid = all(entry is not None for entry in entry_options)
                        if valid:
                            valid = selection_is_grounded(
                                text, reply.choice, entry_options, today, reply.schedule_role,
                            )
                else:
                    valid = False
                if not valid:
                    return PendingReply(act=ReplyAct.unclear)
            elif reply.schedule_role != ScheduleRole.none:
                return PendingReply(act=ReplyAct.unclear)
            return reply
        except Exception:
            logger.warning('Pending reply classifier unavailable', exc_info=True)
            return None

    @staticmethod
    def _independent_command(command: AssistantCommand, previous: AssistantCommand) -> bool:
        if command.action != AssistantIntent.create_entry:
            return False
        if command.kind is None or command.title is None:
            return False
        if previous.action != AssistantIntent.create_entry:
            return True
        if previous.kind is not None and command.kind != previous.kind:
            return True
        return (previous.title is not None
                and command.title.casefold() != previous.title.casefold())

    @staticmethod
    def _task_fallback(text: str, today: datetime.date) -> AssistantCommand | None:
        match = re.match(
            r'^\s*(?:мне\s+)?(?:надо|нужно|необходимо)\s+(?:бы\s+)?(?P<title>.+)$',
            text, re.IGNORECASE,
        )
        if match is None or re.search(
            r'\b(?:отметить|удалить|убрать|перенести|переименовать|изменить)\b',
            match.group('title').casefold(),
        ):
            return None
        return AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.task,
            title=match.group('title').strip(' .!?'),
            date=parse_date(text, today), time=parse_time(text),
        )

    def plan(
        self, text: str, timezone: str, today: datetime.date,
        state: ConversationState | None,
    ) -> DialoguePlan:
        previous = state.pending_command if state else None
        step = self.pending_step(previous) if previous is not None else None
        reply = self._classify(text, previous, state, today) if previous is not None else None
        transition = transition_for_reply(step, reply.act) if reply is not None else None
        preextracted = None
        task_fallback = self._task_fallback(text, today)

        # A bare, grounded schedule answers a schedule question even if the
        # classifier says unclear. Full questions and commands are excluded.
        if (previous is not None and reply is not None and reply.act == ReplyAct.unclear
                and ((previous.selected_entry_id is not None and previous.move_intent)
                     or (previous.action == AssistantIntent.create_entry
                         and step == PendingStep.fields))
                and replacement_schedule_reply(text, today) is not None):
            reply = PendingReply(act=ReplyAct.continue_command)
            transition = DialogueTransition.continue_command

        if (previous is not None and reply is not None and reply.act == ReplyAct.unclear
                and previous.action in (AssistantIntent.list_entries, AssistantIntent.search_entries)
                and parse_period(text, today) is not None):
            reply = PendingReply(act=ReplyAct.continue_command)
            transition = DialogueTransition.continue_command

        if previous is not None and reply is not None and reply.act == ReplyAct.unclear:
            try:
                independent = self._extractor.extract(
                    text=text, timezone=timezone, today=today, previous=None, history=[],
                )
                if self._independent_command(independent, previous):
                    reply = PendingReply(act=ReplyAct.new_command)
                    transition = DialogueTransition.new_command
                    preextracted = independent
            except Exception:
                logger.warning('Independent command check unavailable', exc_info=True)

        if previous is not None:
            if reply is None:
                if re.fullmatch(
                    r'\s*(?:нет[,\s]+)?(?:отмена|отмени|не надо|забудь|передумал(?:а)?)\s*[.!]?',
                    text.casefold(),
                ):
                    transition = DialogueTransition.cancel
                elif re.match(
                    r'^\s*(?:нет[,\s]+)?(?:создай|создать|добавь|добавить|напомни|'
                    r'поставь|покажи|найди|поищи)\b', text.casefold(),
                ):
                    transition = DialogueTransition.new_command
            if task_fallback is not None and previous.kind != EntryKind.task:
                transition = DialogueTransition.new_command
                preextracted = task_fallback
            if transition == DialogueTransition.new_command:
                previous = None

        return DialoguePlan(previous, step, reply, transition, preextracted, task_fallback)
