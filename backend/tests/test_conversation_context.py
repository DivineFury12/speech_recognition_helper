import unittest
import datetime
import uuid
from unittest.mock import patch

from src.ports.conversations import ConversationState
from src.schemas.assistant import ConversationMessage
from src.services.assistant import AssistantService
from src.services.conversation_context import select_conversation_context
from src.services.command_extractor import TurnKind, TurnRoute
from src.schemas.assistant import AssistantIntent


class ConversationContextTests(unittest.TestCase):
    def test_retrieves_older_topic_and_keeps_latest_question(self):
        history = []
        for index in range(10):
            topic = 'собеседование' if index == 1 else f'отпуск {index}'
            history.extend([
                {'role': 'user', 'text': f'Расскажи про {topic}'},
                {'role': 'assistant', 'text': f'Запомнил {topic}'},
            ])
        selected = select_conversation_context(history, 'Когда собеседование?')
        self.assertIn(history[2], selected)
        self.assertIn(history[3], selected)
        self.assertEqual(selected[-1], history[-1])
        self.assertNotIn(history[0], selected)
        self.assertLessEqual(len(selected), 12)
        self.assertEqual(len(history), 20)

    def test_inflected_topic_matches_and_order_is_preserved(self):
        history = []
        for index in range(8):
            topic = 'встреча с Максимом' if index == 0 else f'уроки {index}'
            history.extend([
                {'role': 'user', 'text': topic},
                {'role': 'assistant', 'text': f'Записано: {topic}'},
            ])
        selected = select_conversation_context(history, 'Что было на встрече с Максимом?')
        self.assertEqual(selected[:2], history[:2])
        positions = [history.index(message) for message in selected]
        self.assertEqual(positions, sorted(positions))

    def test_short_reply_keeps_recent_turns_without_unrelated_old_ones(self):
        history = [{'role': 'user' if index % 2 == 0 else 'assistant',
                    'text': f'Реплика {index}'} for index in range(20)]
        selected = select_conversation_context(history, 'да')
        self.assertEqual(selected, history[-6:])

    def test_bounds_large_messages_without_modifying_stored_history(self):
        history = [{'role': 'user', 'text': 'а' * 3000},
                   {'role': 'assistant', 'text': 'б' * 3000}]
        selected = select_conversation_context(history, 'следующий вопрос')
        self.assertEqual([len(message['text']) for message in selected], [1000, 1000])
        self.assertEqual(len(history[0]['text']), 3000)
        self.assertEqual(len(history[1]['text']), 3000)

    def test_budget_does_not_include_half_an_old_turn(self):
        history = []
        for index in range(7):
            topic = 'собеседование' if index == 0 else f'поездка {index}'
            history.extend([
                {'role': 'user', 'text': topic + ' а' * 500},
                {'role': 'assistant', 'text': str(index) + 'б' * 999},
            ])
        selected = select_conversation_context(history, 'собеседование')
        self.assertNotIn(history[0], selected)
        self.assertNotIn(history[1], selected)
        self.assertEqual(selected[-6:], [
            {'role': message['role'], 'text': message['text'][:1000]}
            for message in history[-6:]
        ])

    def test_saved_summary_is_always_sent_before_recent_turns(self):
        history = [{'role': 'summary', 'text': 'Пользователь обсуждает поездку в Казань.'}]
        history.extend({'role': 'user' if index % 2 == 0 else 'assistant',
                        'text': f'Сообщение {index}'} for index in range(8))
        selected = select_conversation_context(history, 'Когда вылет?')
        self.assertEqual(selected[0], history[0])
        self.assertEqual(selected[-1], history[-1])
        self.assertEqual(len([item for item in selected if item['role'] == 'summary']), 1)

    def test_old_turns_are_summarized_for_one_conversation(self):
        class Summarizer:
            def summarize_history(self, previous, messages):
                self.seen = (previous, messages)
                return 'Обсуждали поездку в Казань.'

        class Store:
            def save_summary(self, conversation_id, summary, count):
                self.saved = (conversation_id, summary, count)

        messages = [ConversationMessage(
            id=uuid.uuid4(), role='user' if index % 2 == 0 else 'assistant',
            text=f'Реплика {index}', created_at=datetime.datetime.now(datetime.timezone.utc),
        ) for index in range(18)]
        state = ConversationState(id=uuid.uuid4(), pending_command=None, messages=messages)
        extractor, store = Summarizer(), Store()
        compacted = AssistantService(extractor, conversations=store)._refresh_summary(state)
        self.assertEqual(len(extractor.seen[1]), 12)
        self.assertEqual(store.saved, (state.id, 'Обсуждали поездку в Казань.', 12))
        self.assertEqual(compacted.model_history()[0]['role'], 'summary')
        self.assertEqual(len(compacted.model_history()), 7)
        self.assertEqual(state.summarized_message_count, 0)
        more = [ConversationMessage(
            id=uuid.uuid4(), role='user' if index % 2 == 0 else 'assistant',
            text=f'Новая реплика {index}', created_at=datetime.datetime.now(datetime.timezone.utc),
        ) for index in range(12)]
        continued = AssistantService(extractor, conversations=store)._refresh_summary(
            ConversationState(id=state.id, pending_command=None, messages=messages + more,
                              summary=compacted.summary,
                              summarized_message_count=compacted.summarized_message_count),
        )
        self.assertEqual(extractor.seen[0], 'Обсуждали поездку в Казань.')
        self.assertEqual(continued.summarized_message_count, 24)
        fresh = ConversationState(id=uuid.uuid4(), pending_command=None, messages=[])
        self.assertEqual(fresh.model_history(), [])

    def test_route_receives_refreshed_summary_before_command_processing(self):
        messages = [ConversationMessage(
            id=uuid.uuid4(), role='user' if index % 2 == 0 else 'assistant',
            text=f'Реплика {index}', created_at=datetime.datetime.now(datetime.timezone.utc),
        ) for index in range(18)]
        state = ConversationState(id=uuid.uuid4(), pending_command=None, messages=messages)

        class Store:
            def get(self, conversation_id):
                return state

            def save_summary(self, conversation_id, summary, count):
                self.saved = (summary, count)

        class Extractor:
            def summarize_history(self, previous, chunk):
                return 'Обсуждали поездку.'

            def classify_turn_detailed(self, text, timezone, today, *, pending, history):
                self.history = history
                return TurnRoute(kind=TurnKind.command)

        extractor, store = Extractor(), Store()
        service = AssistantService(extractor, conversations=store)
        sentinel = object()
        with patch.object(service, '_handle_at', return_value=sentinel):
            self.assertIs(service._handle_many('Продолжим', 'Europe/Moscow', state.id,
                                               datetime.date(2026, 10, 29)), sentinel)
        self.assertEqual(store.saved, ('Обсуждали поездку.', 12))
        self.assertEqual(extractor.history[0], {'role': 'summary', 'text': 'Обсуждали поездку.'})
        self.assertNotIn('Реплика 0', str(extractor.history))


if __name__ == '__main__':
    unittest.main()
