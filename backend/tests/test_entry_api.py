import datetime
import io
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.entries.router import router
from src.api.assistant.router import router as assistant_router
from src.api.transcribe.router import get_transcription_service
from src.schemas.assistant import EntryKind
from src.schemas.assistant_command import AssistantCommand
from src.services.assistant import AssistantService
from src.services.entries import EntryService
from test_entry_flow import FakeConversations, FakeExtractor, FakeRepository


class EntryApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRepository()
        self.service = EntryService(self.repository)
        app = FastAPI()
        app.state.entries_service = self.service
        app.include_router(router)
        self.client = TestClient(app)
        self.entry = self.service.create(
            kind=EntryKind.task, title='Купить билеты',
            date=datetime.date(2026, 10, 15), time=None, timezone='Europe/Moscow',
        )

    def test_get_patch_and_delete_entry(self):
        path = f'/v1/entries/{self.entry.id}'
        self.assertEqual(self.client.get(path).json()['title'], 'Купить билеты')
        changed = self.client.patch(path, json={
            'title': 'Купить билеты на поезд', 'status': 'completed',
        })
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()['status'], 'completed')
        self.assertEqual(changed.json()['title'], 'Купить билеты на поезд')
        self.assertIsNotNone(changed.json()['updated_at'])
        self.assertEqual(self.client.delete(path).status_code, 204)
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.delete(path).status_code, 404)

    def test_patch_rejects_empty_title_and_invalid_time(self):
        path = f'/v1/entries/{self.entry.id}'
        self.assertEqual(self.client.patch(path, json={}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={'title': '   '}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={'time': '25:00'}).status_code, 422)
        self.assertEqual(self.client.get(path).json()['title'], 'Купить билеты')

    def test_meeting_duration_can_be_changed_via_patch(self):
        meeting = self.service.create(
            kind=EntryKind.meeting, title='Планёрка',
            date=datetime.date(2026, 10, 20), time='10:00', timezone='Europe/Moscow',
        )
        path = f'/v1/entries/{meeting.id}'
        self.assertEqual(meeting.duration_minutes, 60)
        changed = self.client.patch(path, json={'time': '11:00', 'duration_minutes': 30})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()['duration_minutes'], 30)
        self.assertEqual(self.client.patch(path, json={'duration_minutes': None}).status_code, 422)
        self.assertEqual(self.client.patch(path, json={'duration_minutes': 1441}).status_code, 422)
        self.assertEqual(self.client.patch(
            f'/v1/entries/{self.entry.id}', json={'duration_minutes': 30},
        ).status_code, 422)

    def test_audio_route_completes_task_through_shared_assistant(self):
        class FakeTranscription:
            def transribe(self, source, filename):
                self.last_audio = source.read()
                return 'Отметь задачу купить билеты выполненной'

        app = FastAPI()
        app.state.assistant_service = AssistantService(
            FakeExtractor(AssistantCommand.model_validate({
                'action': 'complete_entry', 'kind': 'task',
                'target_query': 'купить билеты',
            })), self.service, FakeConversations(),
        )
        app.include_router(assistant_router)
        transcription = FakeTranscription()
        app.dependency_overrides[get_transcription_service] = lambda: transcription
        response = TestClient(app).post(
            '/v1/assistant/audio', data={'timezone': 'Europe/Moscow'},
            files={'file': ('voice.webm', io.BytesIO(b'audio'), 'audio/webm')},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['intent'], 'complete_entry')
        self.assertEqual(response.json()['entries'][0]['status'], 'completed')
        self.assertEqual(transcription.last_audio, b'audio')


if __name__ == '__main__':
    unittest.main()
