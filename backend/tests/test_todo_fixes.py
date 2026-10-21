"""Cases from the team's TODO: several records per phrase, bulk filters, dates and times."""

import datetime
import unittest

from src.schemas.assistant import AssistantStatus, EntryKind, EntryStatus
from src.services import assistant as assistant_module
from src.services.batch import bulk_filter, expand_creations
from src.services.temporal import _in_text, ambiguous_time_phrase, parse_date, parse_period
from tests.test_our_features import NOW, TODAY, TZ, Chat, command

WEDNESDAY = datetime.date(2026, 10, 21)


def setUpModule():
    global TOKEN
    TOKEN = assistant_module._now.set(NOW)


def tearDownModule():
    assistant_module._now.reset(TOKEN)


class ParsingTests(unittest.TestCase):
    def test_several_records_in_one_phrase(self):
        self.assertEqual(len(expand_creations('Сделай на послезавтра напоминания сходить в магазин на 10:00 и 18:00')), 2)
        two = expand_creations('Сделай на послезавтра 2 напоминания сходить в магазин на 10:00 и вынести мусор в 18:00')
        self.assertEqual(len(two), 2)
        self.assertIn('вынести мусор', two[1])
        self.assertNotIn(' 2 ', two[0])
        three = expand_creations('сделай напоминание вынести мусор на завтра 10:00, послезавтра 10:00 и через неделю в 10:00')
        self.assertEqual(len(three), 3)
        self.assertIn('послезавтра', three[1])
        for single in ('Поставь встречу с Петром и Иваном завтра в 15:00', 'поставь встречу с 10:00 до 12:00 завтра',
                       'поставь напоминание на завтра 16:00 qwer сдвинутое на 3 символа по шифру Цезаря'):
            self.assertIsNone(expand_creations(single), single)

    def test_bulk_filters(self):
        done = bulk_filter('удали все выполненные задачи', WEDNESDAY, '12:00')
        self.assertEqual((done['kinds'], done['status']), ({EntryKind.task}, EntryStatus.completed))
        after = bulk_filter('удали все задачи после 22 окт.', WEDNESDAY, '12:00')
        self.assertEqual(after['date_from'], datetime.date(2026, 10, 23))
        self.assertTrue(bulk_filter('удали все прошедшие напоминания', WEDNESDAY, '12:00')['past'])
        topic = bulk_filter('Удали напоминания связанные с программированием', WEDNESDAY, '12:00')
        self.assertEqual(topic['topic'], 'программированием')
        months = bulk_filter('удали встречи с октября по ноябрь', WEDNESDAY, '12:00')
        self.assertEqual((months['date_from'], months['date_to']), (datetime.date(2026, 10, 1), datetime.date(2026, 11, 30)))
        self.assertIsNone(bulk_filter('удали встречу с Иваном', WEDNESDAY, '12:00'))

    def test_dates_and_times(self):
        self.assertIsNone(ambiguous_time_phrase('поставь напоминание на завтра 16:00 qwer сдвинутое на 3 символа'))
        self.assertIsNotNone(ambiguous_time_phrase('поставь встречу на завтра в 3'))
        self.assertEqual(parse_date('через месяц', WEDNESDAY), datetime.date(2026, 11, 21))
        self.assertEqual(parse_period('во второй половине этой недели', WEDNESDAY),
                         (datetime.date(2026, 10, 22), datetime.date(2026, 10, 25)))
        self.assertFalse(_in_text('завтра', 'вынести мусор послезавтра 10:00'))
        self.assertTrue(_in_text('послезавтра', 'вынести мусор послезавтра 10:00'))


class DialogueTests(unittest.TestCase):
    def test_undated_and_bare_cancel(self):
        chat = Chat([command(action='create_entry', kind='reminder', title='отчислится')])
        created = chat('поставь бессрочное напоминание отчислится')
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].kind, EntryKind.task)
        self.assertIsNone(created.entries[0].date)
        self.assertIn('нечего отменять', chat('Отмена').assistant_text)

    def test_seventeen_hours_is_a_time_and_same_as_copies(self):
        tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
        chat = Chat([
            command(action='create_entry', kind='meeting', title='встреча с Максимом', date=tomorrow, time='10:00',
                    date_phrase='завтра', time_phrase='10:00'),
            command(action='create_entry', kind='meeting', title='встреча с Максимом', date=tomorrow, time='17:00',
                    time_phrase='на 17 часов', duration_minutes=1020, duration_phrase='на 17 часов'),
        ])
        chat('Встреча, 10:00, завтра, Встреча с Максимом')
        later = chat('Сделай ещё одну встречу с Максимом, но только на 17 часов')
        self.assertEqual(later.entries[0].time, '17:00')
        self.assertEqual(later.entries[0].duration_minutes, 60)
        copy = chat('Сделай такую же на 12:00')
        self.assertEqual((copy.entries[0].title, copy.entries[0].time), ('Встреча с Максимом', '12:00'))

    def test_complete_phrase_and_bulk_delete_by_status(self):
        chat = Chat([command(action='unknown')] * 3)
        reminder = chat.entries.create(kind=EntryKind.reminder, title='Изучить SQL',
                                       date=TODAY + datetime.timedelta(days=2), time='20:00', timezone=TZ)
        done = chat('Поставь изучить sql как выполнено')
        self.assertEqual(chat.repository.get(reminder.id).status, EntryStatus.completed)
        self.assertEqual(done.status, AssistantStatus.completed)
        asked = chat('удали все выполненные напоминания')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertIn('Удалить 1 запись', asked.assistant_text)
        chat('да')
        self.assertIsNone(chat.repository.get(reminder.id))


class ContextTests(unittest.TestCase):
    def test_pronoun_keeps_the_record_even_when_model_copies_new_date_into_target(self):
        tomorrow = (TODAY + datetime.timedelta(days=1)).isoformat()
        week = (TODAY + datetime.timedelta(days=7)).isoformat()
        chat = Chat([
            command(action='create_entry', kind='reminder', title='проверить маяк', date=week, time='18:00',
                    date_phrase='через неделю', time_phrase='в 6 вечера'),
            command(action='update_entry', action_phrase='перенеси', date=tomorrow, date_phrase='завтра',
                    time='17:00', time_phrase='в 5 вечера', target_date=tomorrow, target_date_phrase='завтра',
                    target_time='17:00', target_time_phrase='в 5 вечера', signals_verified=True),
        ])
        chat('Напомни проверить маяк через неделю в 6 вечера')
        moved = chat('А перенеси его на завтра в 5 вечера')
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual((moved.entries[0].date.isoformat(), moved.entries[0].time), (tomorrow, '17:00'))
