import datetime
import json

import pydantic

from src.ports.structured_model import StructuredModelClient
from src.schemas.assistant_command import AssistantCommand


SYSTEM_PROMPT = """Извлеки команду пользователя для ассистента задач и встреч.
Верни ровно один JSON-объект по переданной схеме, без пояснений.
action: create_entry, list_entries, search_entries или unknown.
Для create_entry укажи kind (task, reminder, meeting), title, date и time,
если они следуют из запроса. Для list_entries укажи date_from и date_to,
для search_entries — query. Не выдумывай отсутствующие данные: используй null.
Преобразуй относительные даты по today и timezone из сообщения пользователя.
Дата имеет формат YYYY-MM-DD, время — HH:MM в локальном часовом поясе.
Не выполняй действия и не утверждай, что запись сохранена."""


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
    ) -> AssistantCommand:
        user_prompt = json.dumps(
            {'text': text, 'timezone': timezone, 'today': today.isoformat()},
            ensure_ascii=False,
        )
        payload = self._client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=AssistantCommand.model_json_schema(),
        )
        try:
            return AssistantCommand.model_validate(payload)
        except (pydantic.ValidationError, TypeError, ValueError) as exc:
            raise InvalidModelResponseError('Внешняя модель вернула некорректную команду') from exc
