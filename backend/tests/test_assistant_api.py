import io
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from app import create_app
from src.api.transcribe.router import get_transcription_service


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def handle_message(self, text, timezone, conversation_id):
        self.calls.append((text, timezone, conversation_id))
        return {
            'conversation_id': conversation_id or uuid4(),
            'turn_id': uuid4(),
            'user_text': text,
            'assistant_text': 'Тестовый ответ',
            'status': 'no_action',
            'intent': 'unknown',
            'entries': [],
        }


class FakeTranscription:
    def transribe(self, source, filename):
        assert filename == 'voice.webm'
        assert source.read() == b'audio'
        return 'Напомни завтра'


class FakeConversations:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id

    def list_messages(self, conversation_id):
        return [] if conversation_id == self.conversation_id else None


class AssistantApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_routes_report_unimplemented_without_business_service(self):
        response = self.client.post('/v1/assistant/messages', json={
            'text': 'Что у меня сегодня?', 'timezone': 'Europe/Moscow',
        })
        self.assertEqual(response.status_code, 501)
        self.assertIn('не подключена', response.json()['detail'])

    def test_text_route_passes_request_to_shared_service(self):
        assistant = FakeAssistant()
        self.app.state.assistant_service = assistant
        conversation_id = uuid4()

        response = self.client.post('/v1/assistant/messages', json={
            'text': '  Что у меня сегодня?  ',
            'timezone': 'Europe/Moscow',
            'conversation_id': str(conversation_id),
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(assistant.calls, [
            ('Что у меня сегодня?', 'Europe/Moscow', conversation_id),
        ])
        self.assertEqual(response.json()['user_text'], 'Что у меня сегодня?')

    def test_audio_route_uses_asr_then_same_service(self):
        assistant = FakeAssistant()
        self.app.state.assistant_service = assistant
        self.app.dependency_overrides[get_transcription_service] = FakeTranscription

        response = self.client.post(
            '/v1/assistant/audio',
            data={'timezone': 'Europe/Moscow'},
            files={'file': ('voice.webm', io.BytesIO(b'audio'), 'audio/webm')},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(assistant.calls, [('Напомни завтра', 'Europe/Moscow', None)])
        self.assertEqual(response.json()['user_text'], 'Напомни завтра')

    def test_blank_text_is_rejected(self):
        response = self.client.post('/v1/assistant/messages', json={
            'text': '   ', 'timezone': 'Europe/Moscow',
        })
        self.assertEqual(response.status_code, 422)

    def test_saved_conversation_history_route(self):
        conversation_id = uuid4()
        self.app.state.conversations = FakeConversations(conversation_id)
        response = self.client.get(f'/v1/conversations/{conversation_id}/messages')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            'conversation_id': str(conversation_id), 'messages': [],
        })
        self.assertEqual(
            self.client.get(f'/v1/conversations/{uuid4()}/messages').status_code, 404,
        )


if __name__ == '__main__':
    unittest.main()
