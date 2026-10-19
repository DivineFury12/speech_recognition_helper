"""GigaChat REST adapter for structured command extraction."""

import json
import threading
import time
import uuid

import httpx


DEFAULT_AUTH_URL = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
DEFAULT_CHAT_URL = 'https://api.giga.chat/v1/chat/completions'


class GigaChatAPIError(Exception):
    """Authentication, transport or response failure at the external API."""


class GigaChatClient:
    def __init__(
        self,
        auth_key: str,
        scope: str,
        model: str,
        *,
        auth_url: str = DEFAULT_AUTH_URL,
        chat_url: str = DEFAULT_CHAT_URL,
        ca_bundle: str | None = None,
        http_client: httpx.Client | None = None,
    ):
        self._auth_key = auth_key.removeprefix('Basic ').strip()
        self._scope = scope
        self._model = model
        self._auth_url = auth_url
        self._chat_url = chat_url
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=8.0, read=30.0, write=10.0, pool=8.0),
            verify=ca_bundle or True,
        )
        self._owns_http = http_client is None
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def _access_token(self, *, refresh: bool = False) -> str:
        with self._token_lock:
            if not refresh and self._token and time.time() < self._token_expires_at - 60:
                return self._token
            try:
                response = self._http.post(
                    self._auth_url,
                    headers={
                        'Authorization': f'Basic {self._auth_key}',
                        'RqUID': str(uuid.uuid4()),
                        'Accept': 'application/json',
                    },
                    data={'scope': self._scope},
                )
                response.raise_for_status()
                payload = response.json()
                token = payload['access_token']
                expires_at = float(payload['expires_at'])
                if not isinstance(token, str) or not token:
                    raise ValueError('empty access token')
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                raise GigaChatAPIError('Не удалось получить токен GigaChat') from exc
            self._token = token
            self._token_expires_at = expires_at
            return token

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
    ) -> dict:
        request_body = {
            'model': self._model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'response_format': {
                'type': 'json_schema',
                'schema': json_schema,
                'strict': True,
            },
        }
        token = self._access_token()
        refreshed_token = False
        for attempt in range(3):
            try:
                response = self._http.post(
                    self._chat_url,
                    headers={
                        'Authorization': f'Bearer {token}',
                        'Accept': 'application/json',
                    },
                    json=request_body,
                )
                if response.status_code == 401 and not refreshed_token:
                    token = self._access_token(refresh=True)
                    refreshed_token = True
                    continue
                if response.status_code in (502, 503, 504) and attempt < 1:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError('model response is not an object')
                return result
            except httpx.TransportError as exc:
                if attempt < 1 and not isinstance(exc, httpx.ReadTimeout):
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise GigaChatAPIError('Не удалось связаться с GigaChat') from exc
            except json.JSONDecodeError as exc:
                if attempt < 1:
                    time.sleep(0.4)
                    continue
                raise GigaChatAPIError('Не удалось получить корректный ответ GigaChat') from exc
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                raise GigaChatAPIError('Не удалось получить корректный ответ GigaChat') from exc
        raise GigaChatAPIError('Не удалось авторизовать запрос к GigaChat')
