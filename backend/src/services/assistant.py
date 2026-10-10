import datetime
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.command_extractor import CommandExtractor
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, EntryKind


class InvalidTimezoneError(ValueError):
    pass


class AssistantService:
    """Interpret a message; persistence is intentionally a later step."""

    def __init__(self, extractor: CommandExtractor):
        self._extractor = extractor

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

        if command.action == AssistantIntent.unknown:
            answer = 'Не удалось определить действие по запросу.'
        elif missing:
            answer = f"Не хватает данных: {', '.join(missing)}. Действие не выполнено."
        else:
            answer = 'Команда распознана, но запись и поиск пока не подключены. Действие не выполнено.'

        return AssistantTurn(
            conversation_id=conversation_id or uuid.uuid4(),
            turn_id=uuid.uuid4(),
            user_text=text,
            assistant_text=answer,
            status=AssistantStatus.no_action,
            intent=command.action,
            entries=[],
        )
