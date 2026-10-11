import datetime
import json

import pydantic

from src.ports.structured_model import StructuredModelClient
from src.schemas.assistant import AssistantIntent
from src.schemas.assistant_command import AssistantCommand
from src.services.temporal import verify_creation_time


SYSTEM_PROMPT = """Извлеки команду пользователя для ассистента задач и встреч.
Верни ровно один JSON-объект по переданной схеме, без пояснений.
action: create_entry, list_entries, search_entries или unknown.
Для create_entry укажи kind (task, reminder, meeting), title, date и time,
если они следуют из запроса. Для list_entries укажи date_from, date_to и kind;
для search_entries — query и kind. Если явно названы встречи, kind=meeting;
задачи — kind=task; напоминания — kind=reminder. Например, «создай встречу
с Иваном» и «покажи мои встречи сегодня» означают kind=meeting, а «найди
напоминание про билеты» — kind=reminder. Для списка всех типов kind=null.
Для date и time укажи date_phrase и time_phrase: точные фразы из последнего
сообщения пользователя, на основании которых они получены. Если фразы нет,
значение и соответствующая phrase должны быть null. Не подставляй время по
умолчанию и не выдумывай отсутствующие данные.
Если передан pending_command, извлеки только изменения из нового сообщения.
Сохрани action предыдущей команды, если пользователь явно не начал новую.
Историю используй только для понимания ссылок вроде «его» или «эту встречу»;
приоритет у последнего сообщения. Все поля схемы верни явно, используя null.
Преобразуй относительные даты по today и timezone из сообщения пользователя.
Дата имеет формат YYYY-MM-DD, время — HH:MM в локальном часовом поясе.
Только извлекай поля; действие выполняет сервер."""


class InvalidModelResponseError(Exception):
    """The provider returned data that do not match AssistantCommand."""


class ExternalCommandExtractor:
    def __init__(self, client: StructuredModelClient):
        self._client = client

    def extract(
        self,
        text: str,
        timezone: str,
        today: datetime.date,
        previous: AssistantCommand | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AssistantCommand:
        user_prompt = json.dumps(
            {
                'text': text, 'timezone': timezone, 'today': today.isoformat(),
                'pending_command': previous.model_dump(mode='json') if previous else None,
                'history': (history or [])[-6:],
            },
            ensure_ascii=False,
        )
        payload = self._client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=AssistantCommand.provider_schema(),
        )
        try:
            command = AssistantCommand.model_validate(payload)
        except (pydantic.ValidationError, TypeError, ValueError) as exc:
            raise InvalidModelResponseError('Внешняя модель вернула некорректную команду') from exc
        if command.action == AssistantIntent.create_entry:
            command = verify_creation_time(command, text, today)
        return command
