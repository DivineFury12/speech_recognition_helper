import datetime
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.command_extractor import CommandExtractor
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, EntryKind
from src.services.entries import EntryService


class InvalidTimezoneError(ValueError):
    pass


class AssistantService:
    """Interpret a message and execute supported entry operations."""

    def __init__(self, extractor: CommandExtractor, entries: EntryService | None = None):
        self._extractor = extractor
        self._entries = entries

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

        command = self._extractor.extract(text=text, timezone=timezone, today=today)
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

        result_entries = []
        status = AssistantStatus.no_action
        if command.action == AssistantIntent.unknown:
            answer = 'Не удалось определить действие по запросу.'
        elif missing:
            answer = f"Не хватает данных: {', '.join(missing)}. Действие не выполнено."
        elif self._entries is None:
            answer = 'Команда распознана, но запись и поиск пока не подключены. Действие не выполнено.'
        elif command.action == AssistantIntent.create_entry:
            result_entries = [self._entries.create(
                kind=command.kind, title=command.title, date=command.date,
                time=command.time, timezone=timezone,
            )]
            answer = f'Сохранено: {result_entries[0].title}.'
            status = AssistantStatus.completed
        elif command.action == AssistantIntent.list_entries:
            result_entries = self._entries.list_period(command.date_from, command.date_to)
            answer = f'Найдено записей: {len(result_entries)}.'
            status = AssistantStatus.completed
        elif command.action == AssistantIntent.search_entries:
            result_entries = self._entries.search(command.query)
            answer = f'Найдено записей: {len(result_entries)}.'
            status = AssistantStatus.completed
        else:
            answer = 'Не удалось выполнить команду.'

        return AssistantTurn(
            conversation_id=conversation_id or uuid.uuid4(),
            turn_id=uuid.uuid4(),
            user_text=text,
            assistant_text=answer,
            status=status,
            intent=command.action,
            entries=result_entries,
        )
