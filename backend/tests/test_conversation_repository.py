import unittest
import uuid

from sqlalchemy import delete

from src.adapters.conversation_repository import SqlAlchemyConversationRepository
from src.db import create_session_factory
from src.db_config import get_database_url
from src.models.conversation import ConversationRow
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn
from src.schemas.assistant_command import AssistantCommand


class ConversationRepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database_url = get_database_url()
        if database_url is None:
            raise unittest.SkipTest('Database not configured')
        cls.sessions, cls.engine = create_session_factory(database_url)
        cls.repository = SqlAlchemyConversationRepository(cls.sessions)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_pending_command_and_ordered_history_survive_new_session(self):
        conversation_id = uuid.uuid4()
        pending = AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'reminder', 'title': 'Тренировка',
            'date': '2026-10-16', 'date_phrase': 'в пятницу',
        })
        try:
            first = AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text='В пятницу напомни про тренировку',
                assistant_text='На какое время поставить запись?',
                status=AssistantStatus.needs_clarification,
                intent=AssistantIntent.create_entry, entries=[],
            )
            self.repository.save_turn(first, pending)
            state = self.repository.get(conversation_id)
            self.assertEqual(state.pending_command.date, pending.date)
            self.assertEqual([m.role for m in state.messages], ['user', 'assistant'])

            second = AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text='В 18:00', assistant_text='Сохранено: Тренировка.',
                status=AssistantStatus.completed,
                intent=AssistantIntent.create_entry, entries=[],
            )
            self.repository.save_turn(second, None)
            self.assertIsNone(self.repository.get(conversation_id).pending_command)
            messages = self.repository.list_messages(conversation_id)
            self.assertEqual(
                [m.text for m in messages],
                [first.user_text, first.assistant_text, second.user_text, second.assistant_text],
            )
        finally:
            with self.sessions.begin() as session:
                session.execute(delete(ConversationRow).where(ConversationRow.id == conversation_id))


if __name__ == '__main__':
    unittest.main()
