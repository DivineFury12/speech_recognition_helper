import unittest
import uuid

from sqlalchemy import delete, select

from src.adapters.assistant_request_repository import (
    RequestIdConflictError, SqlAlchemyAssistantRequestRepository,
)
from src.adapters.conversation_repository import SqlAlchemyConversationRepository
from src.adapters.entry_repository import SqlAlchemyEntryRepository
from src.db import create_session_factory
from src.db_config import get_database_url
from src.models.assistant_request import AssistantRequestRow
from src.models.conversation import ConversationRow
from src.models.entry import EntryRow
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, EntryKind


class AssistantRequestRepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database_url = get_database_url()
        if database_url is None:
            raise unittest.SkipTest('Database not configured')
        cls.sessions, cls.engine = create_session_factory(database_url)
        cls.requests = SqlAlchemyAssistantRequestRepository(cls.sessions)
        cls.entries = SqlAlchemyEntryRepository(cls.sessions)
        cls.conversations = SqlAlchemyConversationRepository(cls.sessions)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_retry_replays_one_turn_and_conflicting_payload_is_rejected(self):
        request_id, conversation_id = uuid.uuid4(), uuid.uuid4()
        title = f'Retry {uuid.uuid4()}'
        calls = 0
        entry_id = None

        def execute():
            nonlocal calls, entry_id
            calls += 1
            entry, created = self.entries.create_once(
                kind=EntryKind.task, title=title, date=None, time=None,
                timezone='Europe/Moscow', embedding=None, embedding_model=None,
            )
            self.assertTrue(created)
            entry_id = entry.id
            turn = AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text=title, assistant_text='Сохранено.',
                status=AssistantStatus.completed, intent=AssistantIntent.create_entry,
                entries=[entry],
            )
            self.conversations.save_turn(turn, None)
            return turn

        try:
            first = self.requests.execute_once(request_id, 'same', conversation_id, execute)
            second = self.requests.execute_once(request_id, 'same', conversation_id, execute)
            self.assertEqual(first, second)
            self.assertEqual(first.request_id, request_id)
            self.assertEqual(calls, 1)
            self.assertEqual(len(self.conversations.list_messages(conversation_id)), 2)
            with self.assertRaises(RequestIdConflictError):
                self.requests.execute_once(request_id, 'different', conversation_id, execute)
        finally:
            with self.sessions.begin() as session:
                session.execute(delete(AssistantRequestRow).where(AssistantRequestRow.id == request_id))
                session.execute(delete(ConversationRow).where(ConversationRow.id == conversation_id))
                if entry_id is not None:
                    session.execute(delete(EntryRow).where(EntryRow.id == entry_id))

    def test_failed_request_rolls_back_entry(self):
        request_id = uuid.uuid4()
        title = f'Rollback {uuid.uuid4()}'

        def execute():
            self.entries.create_once(
                kind=EntryKind.task, title=title, date=None, time=None,
                timezone='Europe/Moscow', embedding=None, embedding_model=None,
            )
            raise RuntimeError('simulated failure')

        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            self.requests.execute_once(request_id, 'same', None, execute)
        with self.sessions() as session:
            self.assertIsNone(session.get(AssistantRequestRow, request_id))
            self.assertIsNone(session.scalar(select(EntryRow.id).where(EntryRow.title == title)))
