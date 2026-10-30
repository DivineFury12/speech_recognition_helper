"""HTTP boundary between the API and the private model service."""

from __future__ import annotations

import math
import typing

import httpx
import pydantic

from src.errors import InvalidAudioError
from src.ports.embeddings import EmbeddingUnavailableError
from src.schemas.transcribe import TranscriptSegment
from src.services.tts import TTSUnavailableError, Voice
from src.adapters.local_embeddings import E5_NAME, FRIDA_NAME


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 600):
        self._http = httpx.Client(base_url=base_url.rstrip('/') + '/',
                                  timeout=httpx.Timeout(timeout_seconds, connect=5))

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self._http.request(method, path.lstrip('/'), **kwargs)
            if path in ('transcribe', 'transcribe_longform') and response.status_code == 400:
                raise InvalidAudioError('Аудиофайл не удалось обработать')
            response.raise_for_status()
            return response
        except (httpx.HTTPError, ValueError) as exc:
            raise InferenceUnavailableError('Сервис моделей недоступен') from exc

    def health(self) -> dict:
        try:
            payload = self._request('GET', 'health').json()
            if (not isinstance(payload, dict) or payload.get('status') != 'ok'
                    or not isinstance(payload.get('model'), str)
                    or not isinstance(payload.get('device'), str)):
                raise ValueError('Invalid inference health')
            return payload
        except (ValueError, TypeError) as exc:
            raise InferenceUnavailableError('Сервис моделей не готов') from exc

    def transribe(self, source: typing.BinaryIO, filename: str | None = None) -> str:
        if source.seekable():
            source.seek(0)
        try:
            payload = self._request('POST', 'transcribe', files={
                'file': (filename or 'audio.bin', source),
            }).json()
            text = payload.get('text') if isinstance(payload, dict) else None
            if not isinstance(text, str):
                raise ValueError('Invalid transcript')
            return text
        except (ValueError, TypeError) as exc:
            raise InferenceUnavailableError('Сервис моделей вернул неверный текст') from exc

    def transcribe_longform(self, source: typing.BinaryIO,
                            filename: str | None = None) -> list[TranscriptSegment]:
        if source.seekable():
            source.seek(0)
        try:
            payload = self._request('POST', 'transcribe_longform', files={
                'file': (filename or 'audio.bin', source),
            }).json()
            if not isinstance(payload, dict) or not isinstance(payload.get('segments'), list):
                raise ValueError('Invalid longform transcript')
            return [TranscriptSegment.model_validate(item) for item in payload['segments']]
        except (ValueError, TypeError, pydantic.ValidationError) as exc:
            raise InferenceUnavailableError('Сервис моделей вернул неверную расшифровку') from exc

    def embed(self, text: str, purpose: str, model: str, dimensions: int) -> list[float]:
        try:
            payload = self._request('POST', 'embed', json={
                'text': text, 'purpose': purpose,
            }).json()
            vector = payload['vector']
            if (payload['model'] != model or not isinstance(vector, list)
                    or len(vector) != dimensions or any(
                        not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in vector
                    )):
                raise ValueError('Invalid embedding response')
            return [float(value) for value in vector]
        except (InferenceUnavailableError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingUnavailableError('Сервис эмбеддингов недоступен') from exc

    def voices(self) -> tuple[bool, bool, list[Voice]]:
        try:
            payload = self._request('GET', 'speech/voices').json()
            if not isinstance(payload, dict) or not isinstance(payload.get('voices'), list):
                raise ValueError('Invalid voice list')
            voices = [Voice(item['id'], item['name'], item['description'], item['id'])
                      for item in payload['voices']]
            return bool(payload.get('available')), bool(payload.get('loading')), voices
        except (ValueError, TypeError, KeyError) as exc:
            raise InferenceUnavailableError('Сервис моделей вернул неверный список голосов') from exc

    def synthesize(self, text: str, voice: str) -> bytes:
        try:
            response = self._http.post('speech/synthesize', json={'text': text, 'voice': voice})
            if response.status_code == 404:
                raise KeyError(voice)
            if response.status_code == 422:
                raise ValueError(response.json().get('detail', 'Некорректный текст'))
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, InferenceUnavailableError) as exc:
            raise TTSUnavailableError('Озвучка временно недоступна') from exc


class RemoteEmbeddingModel:
    def __init__(self, client: InferenceClient, name: str):
        if name in ('e5', E5_NAME):
            self.name, self.dimensions = E5_NAME, 384
        elif name in ('frida', FRIDA_NAME):
            self.name, self.dimensions = FRIDA_NAME, 1536
        else:
            raise ValueError(f'Неизвестная модель эмбеддингов: {name}')
        self._client = client

    def embed_document(self, text: str) -> list[float]:
        return self._client.embed(text, 'document', self.name, self.dimensions)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed(text, 'query', self.name, self.dimensions)


class RemoteTTS:
    def __init__(self, client: InferenceClient):
        self._client = client

    @property
    def ready(self) -> bool:
        try:
            return self._client.voices()[0]
        except InferenceUnavailableError:
            return False

    @property
    def error(self) -> str | None:
        try:
            available, loading, _ = self._client.voices()
            return None if available or loading else 'Озвучка недоступна'
        except InferenceUnavailableError:
            return 'Сервис моделей недоступен'

    def voices(self) -> list[Voice]:
        try:
            return self._client.voices()[2]
        except InferenceUnavailableError:
            return []

    def synthesize(self, text: str, voice: str) -> bytes:
        return self._client.synthesize(text, voice)
