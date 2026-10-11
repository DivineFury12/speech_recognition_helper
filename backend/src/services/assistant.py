import datetime
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.command_extractor import CommandExtractor
from src.ports.conversations import ConversationRepository
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, EntryKind
from src.schemas.assistant_command import AssistantCommand
from src.services.entries import EntryService


class InvalidTimezoneError(ValueError):
    pass


class AssistantService:
    """Interpret a message and execute supported entry operations."""

    def __init__(
        self, extractor: CommandExtractor, entries: EntryService | None = None,
        conversations: ConversationRepository | None = None,
    ):
        self._extractor = extractor
        self._entries = entries
        self._conversations = conversations

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
        return AssistantCommand.model_validate(
            {**previous.model_dump(), **changes},
        )

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
    def _question(missing: list[str]) -> str:
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

    def handle_message(
        self,
        text: str,
        timezone: str,
        conversation_id: uuid.UUID | None,
    ) -> AssistantTurn:
        try:
            today = datetime.datetime.now(ZoneInfo(timezone)).date()
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise InvalidTimezoneError('Укажите корректный часовой пояс IANA') from exc

        conversation_id = conversation_id or uuid.uuid4()
        state = self._conversations.get(conversation_id) if self._conversations else None
        previous = state.pending_command if state else None
        if previous is not None and text.strip().casefold() in {
            'отмена', 'отмени', 'не надо', 'забудь',
        }:
            turn = AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text=text, assistant_text='Хорошо, отменил эту команду.',
                status=AssistantStatus.no_action, intent=AssistantIntent.unknown,
                entries=[],
            )
            self._conversations.save_turn(turn, None)
            return turn
        history = (
            [{'role': item.role, 'text': item.text} for item in state.messages]
            if state else []
        )
        command = self._extractor.extract(
            text=text, timezone=timezone, today=today,
            previous=previous, history=history,
        )
        if previous is not None:
            command = self._merge(previous, command)
        missing = self._missing(command)

        result_entries = []
        status = AssistantStatus.no_action
        if command.action == AssistantIntent.unknown:
            answer = 'Не удалось определить действие по запросу.'
        elif missing and (self._entries is None or self._conversations is None):
            answer = f"Не хватает данных: {', '.join(missing)}. Диалог уточнения не подключён."
        elif missing:
            answer = self._question(missing)
            status = AssistantStatus.needs_clarification
        elif self._entries is None:
            answer = 'Команда распознана, но запись и поиск пока не подключены. Действие не выполнено.'
        elif command.action == AssistantIntent.create_entry:
            entry, created = self._entries.create_once(
                kind=command.kind, title=command.title, date=command.date,
                time=command.time, timezone=timezone,
            )
            result_entries = [entry]
            answer = (
                f'Сохранено: {entry.title}.' if created
                else f'Такая запись уже есть: {entry.title}.'
            )
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
        else:
            answer = 'Не удалось выполнить команду.'

        turn = AssistantTurn(
            conversation_id=conversation_id,
            turn_id=uuid.uuid4(),
            user_text=text,
            assistant_text=answer,
            status=status,
            intent=command.action,
            entries=result_entries,
        )
        if self._conversations is not None:
            self._conversations.save_turn(
                turn, command if status == AssistantStatus.needs_clarification else None,
            )
        return turn
