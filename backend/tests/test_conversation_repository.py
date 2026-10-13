import datetime
import unittest
import uuid

from sqlalchemy import delete

from src.adapters.conversation_repository import SqlAlchemyConversationRepository
from src.db import create_session_factory
from src.db_config import get_database_url
from src.models.conversation import ConversationRow
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, Entry, EntryKind, EntryStatus
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import PendingStep


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
            'pending_step': PendingStep.fields,
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
            self.assertEqual(state.pending_command.pending_step, PendingStep.fields)
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

    def test_selected_meeting_survives_clarification_round_trip(self):
        conversation_id = uuid.uuid4()
        entry_id = uuid.uuid4()
        entry = Entry(
            id=entry_id, kind=EntryKind.meeting, title='Встреча с Иваном',
            date=datetime.date(2026, 10, 16), time='16:00', timezone='Europe/Moscow',
            status=EntryStatus.pending, created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        pending = AssistantCommand(
            action=AssistantIntent.update_entry, kind=EntryKind.meeting,
            selected_entry_id=entry_id, move_intent=True,
        )
        try:
            self.repository.save_turn(AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text='Создай встречу', assistant_text='Сохранено.',
                status=AssistantStatus.completed, intent=AssistantIntent.create_entry,
                entries=[entry],
            ), None)
            self.repository.save_turn(AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text='А перенеси её', assistant_text='На какую дату или время?',
                status=AssistantStatus.needs_clarification,
                intent=AssistantIntent.update_entry, entries=[entry],
            ), pending)
            state = self.repository.get(conversation_id)
            self.assertEqual(state.last_entry_id, entry_id)
            self.assertEqual(state.pending_command.selected_entry_id, entry_id)
            self.assertTrue(state.pending_command.move_intent)
        finally:
            with self.sessions.begin() as session:
                session.execute(delete(ConversationRow).where(ConversationRow.id == conversation_id))


if __name__ == '__main__':
    unittest.main()
