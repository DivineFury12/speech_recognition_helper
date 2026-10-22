"""Phrasing reviewers like to try: spoken time, slips of the tongue, questions, bulk changes."""

import datetime
import unittest

from src.schemas.assistant import AssistantStatus, EntryKind, EntryStatus, Recurrence
from src.services import assistant as assistant_module
from src.services.guards import find_until
from src.services.spoken import impossible_date, normalize_request, relative_minutes, spoken_time
from tests.test_our_features import NOW, TODAY, Chat, command

TOMORROW = TODAY + datetime.timedelta(days=1)


def setUpModule():
    global TOKEN
    TOKEN = assistant_module._now.set(NOW)


def tearDownModule():
    assistant_module._now.reset(TOKEN)


def seed(chat, kind, title, date=None, time=None):
    return chat.repository.create(kind=kind, title=title, date=date, time=time, timezone='Europe/Moscow',
                                  embedding=None, embedding_model=None)


class SpokenTimeTests(unittest.TestCase):
    def test_clock_phrases(self):
        for text, expected in [
            ('в полвосьмого вечера', '19:30'), ('без пятнадцати девять утра', '08:45'),
            ('в четверть третьего дня', '14:15'), ('в половине шестого вечера', '17:30'),
            ('без четверти двенадцать', '11:45'), ('полвторого', '13:30'), ('в 15:00', None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(spoken_time(text), expected)

    def test_relative_minutes(self):
        for text, expected in [('через 2 часа 30 минут', 150), ('через полтора часа', 90),
                               ('через 15 минут', 15), ('через час', 60), ('через полчаса', 30), ('в 15:00', None)]:
            with self.subTest(text=text):
                self.assertEqual(relative_minutes(text), expected)

    def test_normalization(self):
        for text, expected in [
            ('Напомни в 10, ой нет, в 11 позвонить', 'Напомни в 11 позвонить'),
            ('пастав задачу на завтр', 'поставь задачу на завтра'),
            ('блин, поставь эту встречу завтра', 'поставь встречу завтра'),
            ('прмвеи завтра стирка в 15 00', 'завтра стирка в 15:00'),
            ('Привет, напомни в 9 30 позвонить', 'напомни в 9:30 позвонить'),
            ('нет, не эту, а ту что в 15:00', 'ту что в 15:00'),
            ('привези завтра посылку', 'привези завтра посылку'),
            ('перенеси на 12 03 2027', 'перенеси на 12 03 2027'),
        ]:
            with self.subTest(text=text):
                self.assertEqual(normalize_request(text), expected)

    def test_impossible_dates_and_until(self):
        self.assertEqual(impossible_date('встреча 31 февраля', 2026), 'В феврале нет 31-го числа')
        self.assertIsNone(impossible_date('встреча 28 февраля', 2026))
        today = datetime.date(2026, 10, 22)
        self.assertEqual(find_until('каждый вторник до конца декабря', today), datetime.date(2026, 12, 31))
        self.assertEqual(find_until('до декабря', today), datetime.date(2026, 11, 30))
        self.assertEqual(find_until('до 5 ноября', today), datetime.date(2026, 11, 5))


class QuestionTests(unittest.TestCase):
    def test_off_topic_and_impossible_date_create_nothing(self):
        chat = Chat([])
        self.assertIn('не помогу', chat('Какая погода завтра?').assistant_text)
        self.assertIn('нет 31-го', chat('Поставь встречу 31 февраля в 10:00').assistant_text)
        self.assertEqual(chat.repository.items, [])

    def test_labelled_task_is_not_off_topic(self):
        chat = Chat([])
        turn = chat('Задача: узнать курс доллара')
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual(chat.repository.items[0].title, 'Узнать курс доллара')
        self.assertEqual(chat.repository.items[0].kind, EntryKind.task)

    def test_count_next_and_free(self):
        chat = Chat([])
        seed(chat, EntryKind.meeting, 'Встреча А', TOMORROW, '10:00')
        seed(chat, EntryKind.meeting, 'Встреча Б', TOMORROW, '12:00')
        self.assertIn('2 встречи', chat('Сколько у меня встреч завтра?').assistant_text)
        self.assertIn('Встреча А', chat('Когда у меня следующая встреча?').assistant_text)
        self.assertIn('Нет', chat('Я свободен завтра в 10:30?').assistant_text)
        self.assertIn('свободно', chat('Я свободен завтра в 15:00?').assistant_text)
        self.assertEqual(len(chat.repository.items), 2)

    def test_weekday_contradiction(self):
        chat = Chat([])
        turn = chat('Поставь встречу с Олегом завтра в понедельник в 10:00')
        self.assertIn('суббота', turn.assistant_text)
        self.assertEqual(chat.repository.items, [])


class BulkTests(unittest.TestCase):
    def test_mark_all_done_skips_meetings_and_undoes_at_once(self):
        chat = Chat([])
        seed(chat, EntryKind.task, 'Задача 1', TODAY)
        seed(chat, EntryKind.task, 'Задача 2', TODAY)
        seed(chat, EntryKind.meeting, 'Планёрка', TODAY, '15:00')
        turn = chat('Отметь все дела на сегодня выполненными')
        self.assertIn('2 записи', turn.assistant_text)
        self.assertIn('Встречи не трогал', turn.assistant_text)
        chat('Отмени последнее')
        self.assertTrue(all(e.status == EntryStatus.pending for e in chat.repository.items))

    def test_move_all_and_undo(self):
        chat = Chat([])
        seed(chat, EntryKind.meeting, 'Планёрка', TOMORROW, '10:00')
        seed(chat, EntryKind.meeting, 'Встреча с Петром', TOMORROW, '15:00')
        monday = TODAY + datetime.timedelta(days=(7 - TODAY.weekday()) % 7 or 7)
        turn = chat('Перенеси все встречи с завтра на понедельник')
        self.assertIn('Перенёс 2', turn.assistant_text)
        self.assertEqual({e.date for e in chat.repository.items}, {monday})
        self.assertIn('2 записи', chat('Отмени последнее').assistant_text)
        self.assertEqual({e.date for e in chat.repository.items}, {TOMORROW})

    def test_undo_walks_back_step_by_step(self):
        tomorrow = TOMORROW.isoformat()
        chat = Chat([
            command(action='create_entry', kind='task', title='купить хлеб', date=tomorrow,
                    date_phrase='завтра'),
            command(action='create_entry', kind='task', title='купить молоко', date=tomorrow,
                    date_phrase='завтра'),
        ])
        chat('Поставь задачу купить хлеб завтра')
        chat('Поставь задачу купить молоко завтра')
        chat('Отмени последнее')
        chat('Отмени последнее')
        self.assertEqual(chat.repository.items, [])


class FollowUpTests(unittest.TestCase):
    def test_rename_and_series_on_last_entry(self):
        tomorrow = TOMORROW.isoformat()
        chat = Chat([command(action='create_entry', kind='task', title='купить хлеб', date=tomorrow,
                             date_phrase='завтра')])
        chat('Поставь задачу купить хлеб завтра')
        self.assertEqual(chat('переименуй её в купить батон').entries[0].title, 'Купить батон')
        turn = chat('сделай его на каждый день')
        self.assertIn('каждый день', turn.assistant_text)
        self.assertEqual(chat.repository.items[0].recurrence, Recurrence.daily)

    def test_same_for_another_person(self):
        tomorrow = TOMORROW.isoformat()
        chat = Chat([command(action='create_entry', kind='reminder', title='позвонить Петру Сергеевичу',
                             date=tomorrow, time='10:00', date_phrase='завтра', time_phrase='в 10:00')])
        chat('Напомни завтра в 10:00 позвонить Петру Сергеевичу')
        chat('а Ивану в 11:00?')
        self.assertEqual([(e.title, e.time) for e in chat.repository.items],
                         [('Позвонить Петру Сергеевичу', '10:00'), ('Позвонить Ивану', '11:00')])

    def test_new_record_is_not_a_silent_move_of_the_last_one(self):
        tomorrow = TOMORROW.isoformat()
        chat = Chat([
            command(action='create_entry', kind='task', title='распитие напитков', date=TODAY.isoformat(),
                    time='18:00', date_phrase='сегодня', time_phrase='в 18:00'),
            command(action='update_entry', date=tomorrow, time='15:00', date_phrase='завтра',
                    time_phrase='в 15:00'),
        ])
        chat('Поставь задачу распитие напитков сегодня в 18:00')
        turn = chat('прмвеи завтра стирка в 15 00')
        self.assertEqual(turn.status, AssistantStatus.completed)
        titles = {e.title: (e.date, e.time) for e in chat.repository.items}
        self.assertEqual(titles['Распитие напитков'], (TODAY, '18:00'))
        self.assertEqual(titles['Стирка'], (TOMORROW, '15:00'))

    def test_new_request_while_yes_no_question_is_open(self):
        from src.services.assistant import AssistantService
        for text, expected in [('завтра стирка в 15:00', True), ('поставь задачу купить хлеб', True),
                               ('на завтра в 15:00', False), ('да', False), ('нет, на 16:00', False),
                               ('ну на вчера', False), ('Встреча с Иваном', False)]:
            with self.subTest(text=text):
                self.assertEqual(AssistantService._is_new_request(text), expected)

    def test_pick_by_time_after_negation(self):
        chat = Chat([command(action='delete_entry', kind='meeting')])
        seed(chat, EntryKind.meeting, 'Встреча с Петром', TOMORROW, '10:00')
        seed(chat, EntryKind.meeting, 'Встреча с Иваном', TOMORROW, '15:00')
        chat('Удали встречу')
        chat('нет, не эту, а ту что в 15:00')
        chat('да')
        self.assertEqual([e.title for e in chat.repository.items], ['Встреча с Петром'])


if __name__ == '__main__':
    unittest.main()
