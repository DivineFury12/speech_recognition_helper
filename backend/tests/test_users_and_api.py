"""Separate users in PostgreSQL, recurrence, undo restore and the HTTP security layer."""

import datetime
import unittest
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import delete

from src.adapters.conversation_repository import (SqlAlchemyConversationRepository,
                                                  SqlAlchemyUserSettingsRepository)
from src.adapters.entry_repository import SqlAlchemyEntryRepository
from src.api.auth import Lockout, Sessions
from src.db import create_session_factory
from src.db_config import get_database_url
from src.models.entry import EntryRow
from src.schemas.assistant import EntryKind, Recurrence
from src.user_context import acting_as


class UserIsolationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database_url = get_database_url()
        if database_url is None:
            raise unittest.SkipTest('Database not configured')
        cls.sessions, cls.engine = create_session_factory(database_url)
        cls.repository = SqlAlchemyEntryRepository(cls.sessions)
        cls.conversations = SqlAlchemyConversationRepository(cls.sessions)
        cls.settings = SqlAlchemyUserSettingsRepository(cls.sessions)

    @classmethod
    def tearDownClass(cls):
        with cls.sessions.begin() as session:
            session.execute(delete(EntryRow).where(EntryRow.title.like('Изоляция %')))
        cls.engine.dispose()

    def test_users_do_not_see_each_other(self):
        alice, bob = f'alice-{uuid.uuid4()}', f'bob-{uuid.uuid4()}'
        with acting_as(alice):
            entry = self.repository.create(kind=EntryKind.task, title='Изоляция отчёт',
                                           date=datetime.date(2026, 11, 2), time=None,
                                           timezone='Europe/Moscow', embedding=None, embedding_model=None)
        with acting_as(bob):
            self.assertIsNone(self.repository.get(entry.id))
            self.assertEqual(self.repository.search_text('Изоляция отчёт'), [])
            self.assertFalse(self.repository.delete(entry.id))
            self.assertIsNone(self.repository.update(entry.id, {'title': 'чужое'}, embedding=None,
                                                     embedding_model=None))
        with acting_as(alice):
            self.assertEqual(self.repository.get(entry.id).title, 'Изоляция отчёт')

    def test_recurring_entry_expands_and_restore_returns_same_id(self):
        user = f'rec-{uuid.uuid4()}'
        with acting_as(user):
            entry = self.repository.create(kind=EntryKind.meeting, title='Изоляция планёрка',
                                           date=datetime.date(2026, 11, 2), time='10:00',
                                           timezone='Europe/Moscow', embedding=None, embedding_model=None,
                                           duration_minutes=30, recurrence=Recurrence.weekly)
            days = [e.date for e in self.repository.list_period(datetime.date(2026, 11, 1),
                                                                datetime.date(2026, 11, 30))]
            self.assertEqual(days, [datetime.date(2026, 11, d) for d in (2, 9, 16, 23, 30)])
            self.assertTrue(self.repository.delete(entry.id))
            restored = self.repository.restore(entry)
            self.assertEqual(restored.id, entry.id)
            self.assertEqual(restored.recurrence, Recurrence.weekly)

    def test_feed_version_and_undo_record(self):
        user = f'feed-{uuid.uuid4()}'
        self.assertEqual(self.settings.feed_version(user), 0)
        self.assertEqual(self.settings.bump_feed_version(user), 1)
        with acting_as(user):
            cid = uuid.uuid4()
            self.conversations.set_undo(cid, {'op': 'create'})
            self.assertEqual(self.conversations.get_undo(cid), {'op': 'create'})
        with acting_as('someone-else'):
            self.assertIsNone(self.conversations.get_undo(cid))
            self.assertTrue(self.conversations.is_foreign(cid))


class SecurityLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import create_app
        from src.settings import Settings
        import pathlib
        import tempfile
        from unittest import mock

        cls.tmp = tempfile.TemporaryDirectory()
        settings = Settings(
            model_name='test', device='cpu', cache_dir=pathlib.Path(cls.tmp.name),
            ffmpeg_timeout_seconds=10, gigachat_auth_key=None, gigachat_scope='GIGACHAT_API_PERS',
            gigachat_model='GigaChat-2', gigachat_ca_bundle=None, database_url=get_database_url(),
            voice_enabled=False, tts_enabled=False, data_dir=pathlib.Path(cls.tmp.name),
            access_code='секрет-123',
        )
        cls.patch = mock.patch('app.src.adapters.local_embeddings.create_embedding_model', return_value=None)
        cls.patch.start()
        cls.app = create_app(settings)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def test_access_code_lockout_headers_and_csrf(self):
        with TestClient(self.app, headers={'X-Requested-With': 'assistant'}) as client:
            info = client.get('/v1/session')
            self.assertEqual(info.headers['x-frame-options'], 'DENY')
            self.assertIn("frame-ancestors 'none'", info.headers['content-security-policy'])
            self.assertTrue(info.json()['access_code_required'])
            self.assertEqual(client.get('/v1/entries/undated').status_code, 401)
            for _ in range(5):
                self.assertEqual(client.post('/v1/session/unlock', json={'code': 'нет'}).status_code, 401)
            self.assertEqual(client.post('/v1/session/unlock', json={'code': 'секрет-123'}).status_code, 429)
            self.app.state.unlock_lockout = Lockout(5, 900)
            self.assertEqual(client.post('/v1/session/unlock', json={'code': 'секрет-123'}).status_code, 204)
            no_header = client.post('/v1/calendar/rotate', headers={'X-Requested-With': ''})
            self.assertEqual(no_header.status_code, 403)
            too_big = client.post('/v1/assistant/messages', content=b'x',
                                  headers={'Content-Length': str(50 * 1024 * 1024)})
            self.assertEqual(too_big.status_code, 413)
            self.assertEqual(client.post('/v1/assistant/messages',
                                         json={'text': 'а' * 1001, 'timezone': 'Europe/Moscow'}).status_code, 422)
            names = [s['name'] for s in client.get('/v1/status').json()['services']]
            self.assertEqual(names[0], 'Понимание команд')

    def test_calendar_link_rotation(self):
        if self.app.state.sessions is None or get_database_url() is None:
            self.skipTest('Database not configured')
        with TestClient(self.app, headers={'X-Requested-With': 'assistant'}) as client:
            client.post('/v1/session/unlock', json={'code': 'секрет-123'})
            old = client.get('/v1/calendar').json()['url']
            new = client.post('/v1/calendar/rotate').json()['url']
            self.assertNotEqual(old, new)
            path = lambda url: '/' + url.split('://', 1)[1].split('/', 1)[1]
            with TestClient(self.app) as anonymous:
                self.assertEqual(anonymous.get(path(old)).status_code, 404)
                feed = anonymous.get(path(new))
                self.assertEqual(feed.status_code, 200)
                self.assertIn('BEGIN:VCALENDAR', feed.text)

    def test_feed_token_signature(self):
        sessions = Sessions(b'k' * 32)
        token = sessions.feed_token('user-1', 3)
        self.assertEqual(sessions.feed_user(token), ('user-1', 3))
        self.assertIsNone(sessions.feed_user('user-1.3.forged'))
