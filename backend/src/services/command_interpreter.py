"""Ground an extracted action and its fields before any entry operation."""

import dataclasses
import datetime
import re

from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, EntryKind
from src.schemas.assistant_command import AssistantCommand, UpdateMode
from src.services.temporal import (ambiguous_time_phrase, parse_date, parse_duration,
                                   parse_period, parse_target_date, parse_time,
                                   verify_search_query)


@dataclasses.dataclass(frozen=True)
class ResolvedCommand:
    command: AssistantCommand
    previous: AssistantCommand | None
    explicit_target: bool
    pronoun_reference: bool


class CommandInterpreter:
    """Apply one grounded command after the dialogue transition is decided."""

    @staticmethod
    def _merge(previous: AssistantCommand, update: AssistantCommand) -> AssistantCommand:
        if update.action == AssistantIntent.unknown:
            return previous
        if update.action != previous.action:
            return update
        changes = {
            name: value for name, value in update.model_dump().items()
            if name != 'action' and value is not None
        }
        return AssistantCommand.model_validate({**previous.model_dump(), **changes})

    @staticmethod
    def _continuation_fields(
        previous: AssistantCommand, update: AssistantCommand,
        text: str, today: datetime.date,
    ) -> AssistantCommand:
        """A clarification cannot change the pending action without new action evidence."""
        if update.action == previous.action:
            if (previous.action == AssistantIntent.create_entry or
                    previous.action == AssistantIntent.update_entry
                    and previous.selected_entry_id):
                grounded_date = None if update.date_unverified else parse_date(text, today)
                grounded_time = None if update.time_unverified else parse_time(text)
                grounded_time_phrase = ambiguous_time_phrase(text)
                if grounded_date is not None or grounded_time is not None or grounded_time_phrase:
                    return update.model_copy(update={
                        'date': grounded_date or update.date,
                        'date_unverified': False if grounded_date else update.date_unverified,
                        'time': grounded_time or update.time,
                        'time_unverified': False if grounded_time else update.time_unverified,
                        'time_phrase': grounded_time_phrase or update.time_phrase,
                    })
            return update
        if update.action_phrase is not None:
            return update
        if previous.action == AssistantIntent.create_entry:
            return AssistantCommand(
                action=previous.action, kind=update.kind, title=update.title,
                date=(update.date if update.date_unverified
                      else parse_date(text, today) or update.date),
                date_phrase=update.date_phrase,
                date_relation=update.date_relation,
                date_unverified=update.date_unverified,
                time=(update.time if update.time_unverified
                      else parse_time(text) or update.time),
                time_phrase=ambiguous_time_phrase(text) or update.time_phrase,
                time_relation=update.time_relation,
                time_unverified=update.time_unverified,
                signals_verified=update.signals_verified,
                duration_minutes=(parse_duration(text)
                                  if (update.kind or previous.kind) == EntryKind.meeting else None),
            )
        if previous.action == AssistantIntent.update_entry:
            return AssistantCommand(
                action=previous.action, kind=update.kind,
                target_query=verify_search_query(update.target_query or update.query, text),
                date=(update.date if update.date_unverified
                      else parse_date(text, today) or update.date),
                date_phrase=update.date_phrase,
                date_relation=update.date_relation,
                date_unverified=update.date_unverified,
                time=(update.time if update.time_unverified
                      else parse_time(text) or update.time),
                time_phrase=ambiguous_time_phrase(text) or update.time_phrase,
                time_relation=update.time_relation,
                time_unverified=update.time_unverified,
                signals_verified=update.signals_verified,
                duration_minutes=parse_duration(text, for_update=True),
                new_title=update.new_title or update.title,
            )
        if previous.action in (AssistantIntent.complete_entry, AssistantIntent.delete_entry):
            return AssistantCommand(
                action=previous.action, kind=update.kind,
                target_query=verify_search_query(
                    update.target_query or update.query or update.title, text,
                ),
                target_date=parse_target_date(text, today),
                target_time=parse_time(text),
            )
        if previous.action == AssistantIntent.list_entries:
            period = ((update.date_from, update.date_to)
                      if update.date_from is not None and update.date_to is not None
                      else parse_period(text, today) if not update.signals_verified else None)
            return AssistantCommand(
                action=previous.action, kind=update.kind,
                date_from=period[0] if period else None,
                date_to=period[1] if period else None,
                query=verify_search_query(update.query, text, topic=True),
            )
        if previous.action == AssistantIntent.search_entries:
            period = ((update.date_from, update.date_to)
                      if update.date_from is not None and update.date_to is not None
                      else parse_period(text, today) if not update.signals_verified else None)
            return AssistantCommand(
                action=previous.action, kind=update.kind,
                query=verify_search_query(
                    update.query or update.target_query or update.title, text, topic=True,
                ),
                date_from=period[0] if period else None,
                date_to=period[1] if period else None,
            )
        return update

    def resolve(
        self, command: AssistantCommand, *, text: str, today: datetime.date,
        previous: AssistantCommand | None, state: ConversationState | None,
        continuation: bool, resolved_time: str | None,
        task_fallback: AssistantCommand | None,
    ) -> ResolvedCommand:
        if previous is not None and continuation:
            command = self._continuation_fields(previous, command, text, today)

        # The model must not settle an ambiguous spoken hour on its own.
        unclear_time = ambiguous_time_phrase(text)
        if unclear_time and command.action in (AssistantIntent.create_entry,
                                               AssistantIntent.update_entry):
            command = command.model_copy(update={'time': None, 'time_phrase': unclear_time})
        if (not command.signals_verified and task_fallback is not None
                and command.action in (AssistantIntent.unknown, AssistantIntent.complete_entry)):
            command = task_fallback

        action_evidence = command.action_phrase or text
        move_requested = command.action == AssistantIntent.update_entry and command.update_mode == UpdateMode.move
        # The structured model owns intent classification. Lexical recovery is
        # limited to the deterministic offline adapter, which has no semantic classifier.
        lexical_fallback = not command.signals_verified
        if lexical_fallback:
            move_requested = move_requested or bool(re.search(
                r'\b(?:перенес\w*|перенос\w*|перенести|перенеси|передвин\w*|сдвин\w*|перемест\w*)\b',
                action_evidence.casefold(),
            ))
        explicit_move = lexical_fallback and bool(re.search(
            r'\b(?:перенеси|перенести|передвинь|сдвинь|перемести)\b',
            action_evidence.casefold(),
        ))
        explicit_delete = lexical_fallback and bool(re.search(
            r'\b(?:удали|удалить|убери|убрать|сотри)\b',
            action_evidence.casefold(),
        ))
        if re.search(r'\bне\s+(?:перенес\w*|переноси|передвин\w*|сдвин\w*)\b',
                     action_evidence.casefold()):
            explicit_move = False
            move_requested = False
        if re.search(r'\bне\s+(?:удал\w*|убер\w*|убир\w*|сотр\w*)\b',
                     action_evidence.casefold()):
            explicit_delete = False
        if command.action != AssistantIntent.delete_entry and explicit_delete:
            command = AssistantCommand(
                action=AssistantIntent.delete_entry, kind=command.kind,
                target_query=command.target_query, target_date=command.target_date,
                target_date_phrase=command.target_date_phrase,
                target_time=command.target_time, target_time_phrase=command.target_time_phrase,
            )
        if command.action != AssistantIntent.update_entry and explicit_move:
            command = AssistantCommand(action=AssistantIntent.update_entry)
        if command.target_query is None and (explicit_move or explicit_delete):
            target_words = re.sub(
                r'^\s*(?:а\s+)?(?:перенеси|перенести|передвинь|сдвинь|перемести|'
                r'удали|удалить|убери|убрать|сотри)\s+',
                '', text, flags=re.IGNORECASE,
            ).strip(' \t\r\n.?!')
            if (target_words and not re.match(r'^(?:её|ее|его|эту|этот)\b', target_words.casefold())
                    and parse_date(target_words, today) is None
                    and parse_time(target_words) is None
                    and verify_search_query(target_words, text)):
                command = command.model_copy(update={'target_query': target_words})
        if (lexical_fallback and previous is not None and previous.selected_entry_id is not None
                and previous.move_intent and command.action != AssistantIntent.update_entry
                and not re.search(
                    r'\b(?:создай|добавь|напомни|покажи|найди|удали|отметь)\b',
                    text.casefold(),
                )):
            reply_date = parse_date(text, today)
            reply_time = parse_time(text)
            if reply_date is not None or reply_time is not None:
                command = AssistantCommand(
                    action=AssistantIntent.update_entry, date=reply_date, time=reply_time,
                    time_phrase=ambiguous_time_phrase(text),
                )

        explicit_target = verify_search_query(command.target_query, text) is not None
        if command.target_query is not None and not explicit_target:
            command = command.model_copy(update={'target_query': None})
        if previous is not None and previous.selected_entry_id is not None and explicit_target:
            previous = None
        if previous is not None:
            command = self._merge(previous, command)
        if resolved_time:
            command = command.model_copy(update={'time_phrase': None})
        if command.action == AssistantIntent.update_entry and move_requested:
            command = command.model_copy(update={'move_intent': True})
        if command.action == AssistantIntent.update_entry and command.duration_minutes is None:
            explicit_duration = parse_duration(text, for_update=True)
            if explicit_duration is not None:
                command = command.model_copy(update={'duration_minutes': explicit_duration})

        pronoun_reference = command.target_reference_phrase is not None or (
            lexical_fallback and bool(re.search(
                r'\b(?:её|ее|его|эту|этот)\b', text.casefold(),
            ))
        )
        bound_target = (previous is not None and previous.selected_entry_id is not None) or (
            state is not None and state.last_entry_id is not None and pronoun_reference
        )
        if (bound_target and command.action == AssistantIntent.update_entry
                and command.move_intent and command.date is None and command.time is None
                and not re.search(r'\b(?:не|нет|отмена)\b', text.casefold())):
            command = command.model_copy(update={
                'date': parse_date(text, today), 'time': parse_time(text),
                'time_phrase': ambiguous_time_phrase(text) or command.time_phrase,
            })
        return ResolvedCommand(command, previous, explicit_target, pronoun_reference)
