from gigachat import GigaChat
from gigachat.exceptions import GigaChatException
from gigachat.models import Chat, Messages, MessagesRole

from .base import LLMError


class GigaChatProvider:
    name = "gigachat"

    def __init__(self, *, credentials: str, scope: str, model: str, verify_ssl: bool,
                 ca_bundle: str = "", timeout: float = 30) -> None:
        if not credentials:
            raise LLMError("GIGACHAT_CREDENTIALS не задан")
        kwargs = {
            "credentials": credentials,
            "scope": scope,
            "model": model,
            "verify_ssl_certs": verify_ssl,
            "timeout": timeout,
            "max_retries": 2,
        }
        if ca_bundle:
            kwargs["ca_bundle_file"] = ca_bundle
        self._client = GigaChat(**kwargs)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        chat = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system),
                Messages(role=MessagesRole.USER, content=user),
            ],
            temperature=0.1,
            max_tokens=512,
        )
        try:
            response = self._client.chat(chat)
        except GigaChatException as e:
            raise LLMError(f"GigaChat: {e}") from e
        except Exception as e:
            raise LLMError(f"GigaChat недоступен: {e}") from e
        if not response.choices:
            raise LLMError("GigaChat вернул пустой ответ")
        return response.choices[0].message.content or ""

    def close(self) -> None:
        self._client.close()
