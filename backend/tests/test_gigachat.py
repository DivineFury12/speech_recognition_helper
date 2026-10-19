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

    def test_retries_one_malformed_structured_response(self):
        completions = 0

        def handle(request):
            nonlocal completions
            if request.url.path == '/api/v2/oauth':
                return httpx.Response(200, json={
                    'access_token': 'token', 'expires_at': time.time() + 1800,
                })
            completions += 1
            content = 'not json' if completions == 1 else '{"parts":["one","two"]}'
            return httpx.Response(200, json={
                'choices': [{'message': {'content': content}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient(
            auth_key='secret-key', scope='GIGACHAT_API_PERS', model='GigaChat-2',
            http_client=http_client,
        )
        try:
            with mock.patch('src.adapters.gigachat.time.sleep'):
                self.assertEqual(client.complete_json('system', 'user', {'type': 'object'}),
                                 {'parts': ['one', 'two']})
            self.assertEqual(completions, 2)
        finally:
            http_client.close()

    def test_retries_transient_completion_disconnect(self):
        completions = 0

        def handle(request):
            nonlocal completions
            if request.url.path == '/api/v2/oauth':
                return httpx.Response(200, json={
                    'access_token': 'token', 'expires_at': time.time() + 1800,
                })
            completions += 1
            if completions == 1:
                raise httpx.RemoteProtocolError('connection closed')
            return httpx.Response(200, json={
                'choices': [{'message': {'content': '{"action":"unknown"}'}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient(
            auth_key='secret-key', scope='GIGACHAT_API_PERS', model='GigaChat-2',
            http_client=http_client,
        )
        try:
            with mock.patch('src.adapters.gigachat.time.sleep'):
                result = client.complete_json('system', 'user', {'type': 'object'})
            self.assertEqual(result, {'action': 'unknown'})
            self.assertEqual(completions, 2)
        finally:
            http_client.close()

    def test_can_refresh_token_after_transient_disconnect(self):
        auth_calls = completion_calls = 0

        def handle(request):
            nonlocal auth_calls, completion_calls
            if request.url.path == '/api/v2/oauth':
                auth_calls += 1
                return httpx.Response(200, json={
                    'access_token': f'token-{auth_calls}',
                    'expires_at': time.time() + 1800,
                })
            completion_calls += 1
            if completion_calls == 1:
                raise httpx.RemoteProtocolError('connection closed')
            if completion_calls == 2:
                return httpx.Response(401)
            self.assertEqual(request.headers['authorization'], 'Bearer token-2')
            return httpx.Response(200, json={
                'choices': [{'message': {'content': '{"action":"unknown"}'}}],
            })

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient(
            auth_key='secret-key', scope='GIGACHAT_API_PERS', model='GigaChat-2',
            http_client=http_client,
        )
        try:
            with mock.patch('src.adapters.gigachat.time.sleep'):
                result = client.complete_json('system', 'user', {'type': 'object'})
            self.assertEqual(result, {'action': 'unknown'})
            self.assertEqual((auth_calls, completion_calls), (2, 3))
        finally:
            http_client.close()


if __name__ == '__main__':
    unittest.main()
