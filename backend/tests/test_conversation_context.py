import unittest

from src.services.conversation_context import select_conversation_context


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


if __name__ == '__main__':
    unittest.main()
