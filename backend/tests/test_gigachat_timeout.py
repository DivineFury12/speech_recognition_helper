"""Bound retries when the external model stalls."""

import time
import unittest

import httpx

from src.adapters.gigachat import GigaChatAPIError, GigaChatClient


class GigaChatTimeoutTests(unittest.TestCase):
    def test_read_timeout_is_not_retried(self):
        completions = 0

        def handle(request):
            nonlocal completions
            if request.url.path == '/api/v2/oauth':
                return httpx.Response(200, json={
                    'access_token': 'test-token', 'expires_at': time.time() + 1800,
                })
            completions += 1
            raise httpx.ReadTimeout('model did not respond')

        http_client = httpx.Client(transport=httpx.MockTransport(handle))
        client = GigaChatClient('test-key', 'GIGACHAT_API_PERS', 'GigaChat-2',
                                http_client=http_client)
        try:
            with self.assertRaises(GigaChatAPIError):
                client.complete_json('system', 'user', {'type': 'object'})
            self.assertEqual(completions, 1)
        finally:
            http_client.close()
