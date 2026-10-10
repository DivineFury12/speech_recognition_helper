import json
import pathlib
import tempfile
import time
import unittest
import uuid
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from app import create_app
from src.adapters.gigachat import GigaChatAPIError, GigaChatClient
from src.settings import Settings


class GigaChatClientTests(unittest.TestCase):
    def test_authentication_and_structured_completion(self):
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path == '/api/v2/oauth':
                self.assertEqual(request.headers['authorization'], 'Basic secret-key')
                uuid.UUID(request.headers['rquid'])
                self.assertEqual(request.content, b'scope=GIGACHAT_API_PERS')
                return httpx.Response(200, json={
                    'access_token': 'token', 'expires_at': time.time() + 1800,
                })
            self.assertEqual(request.headers['authorization'], 'Bearer token')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'GigaChat-2')
            self.assertEqual(body['response_format']['type'], 'json_schema')
            self.assertTrue(body['response_format']['strict'])
            self.assertEqual(body['messages'][1]['content'], 'user text')
            return httpx.Response(200, json={
                'choices': [{'message': {'content': '{"action":"unknown"}'}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient(
            auth_key='secret-key', scope='GIGACHAT_API_PERS', model='GigaChat-2',
            http_client=http_client,
        )
        try:
            for _ in range(2):
                result = client.complete_json('system text', 'user text', {'type': 'object'})
                self.assertEqual(result, {'action': 'unknown'})
            self.assertEqual(len([r for r in requests if r.url.path == '/api/v2/oauth']), 1)
        finally:
            http_client.close()

    def test_configured_app_uses_external_model_for_text(self):
        class FakeTranscriber:
            device = 'cpu'

        class FakeGigaChat:
            def complete_json(self, system_prompt, user_prompt, json_schema):
                return {'action': 'create_entry', 'kind': 'task', 'title': 'Купить билеты'}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                model_name='test', device='cpu', cache_dir=pathlib.Path(directory),
                ffmpeg_timeout_seconds=10, gigachat_auth_key='test-key',
                gigachat_scope='GIGACHAT_API_PERS', gigachat_model='GigaChat-2',
                gigachat_ca_bundle=None,
            )
            with mock.patch('app.src.adapters.transcriber.GigaAMTransciber.load', return_value=FakeTranscriber()), \
                 mock.patch('app.src.adapters.gigachat.GigaChatClient', return_value=FakeGigaChat()):
                with TestClient(create_app(settings)) as client:
                    response = client.post('/v1/assistant/messages', json={
                        'text': 'Создай задачу купить билеты',
                        'timezone': 'Europe/Moscow',
                    })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['intent'], 'create_entry')
        self.assertEqual(response.json()['status'], 'no_action')
        self.assertEqual(response.json()['entries'], [])

    def test_invalid_model_json_is_rejected(self):
        def handle(request):
            if request.url.path == '/api/v2/oauth':
                return httpx.Response(200, json={
                    'access_token': 'token', 'expires_at': time.time() + 1800,
                })
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'not json'}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient(
            auth_key='secret-key', scope='GIGACHAT_API_PERS', model='GigaChat-2',
            http_client=http_client,
        )
        try:
            with self.assertRaises(GigaChatAPIError):
                client.complete_json('system', 'user', {'type': 'object'})
        finally:
            http_client.close()


if __name__ == '__main__':
    unittest.main()
