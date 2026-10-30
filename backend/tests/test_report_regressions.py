import datetime
import unittest
from unittest import mock

from src.adapters.rules_model import command_from_rules
from src.schemas.assistant import AssistantIntent, EntryKind
from src.schemas.assistant_command import AssistantCommand, DateRelation
from src.services.assistant import AssistantService
from src.services.command_extractor import ExternalCommandExtractor
from src.services.entries import EntryService
from src.services.guards import restore_numbers
from src.services.temporal import verify_creation_fields


class ReportRegressionsTest(unittest.TestCase):
    def test_absolute_past_date_is_not_repeated(self):
        note = AssistantService._past_note(datetime.date(2024, 1, 7), '14:45')
        self.assertIn('2024', note)
        self.assertIn('уже прошла', note)
        self.assertNotIn(' — это ', note)

    def test_standalone_clock_time_uses_next_occurrence_without_changing_type(self):
        command = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.task,
            title='Отправить отчёт', time='11:15',
        )
        with mock.patch.object(AssistantService, '_now', return_value=datetime.datetime(2026, 10, 29, 10)):
            today = AssistantService._anchor_standalone_time(command)
        with mock.patch.object(AssistantService, '_now', return_value=datetime.datetime(2026, 10, 29, 12)):
            tomorrow = AssistantService._anchor_standalone_time(command)
        self.assertEqual(today.date, datetime.date(2026, 10, 29))
        self.assertEqual(tomorrow.date, datetime.date(2026, 10, 30))
        self.assertEqual(today.kind, EntryKind.task)
        self.assertEqual(tomorrow.kind, EntryKind.task)
        self.assertNotIn('дату', AssistantService._missing(today))

    def test_standalone_time_does_not_resolve_an_ambiguous_period(self):
        command = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.task,
            title='Подготовить отчёт', time='11:15',
            date_relation=DateRelation.period,
        )
        self.assertIsNone(AssistantService._anchor_standalone_time(command).date)
        self.assertIn('дату', AssistantService._missing(command))

    def test_offline_fallback_does_not_turn_timed_work_into_reminder(self):
        command = command_from_rules('Подготовить протокол к 19:30', datetime.date(2026, 10, 29))
        self.assertEqual(command['action'], 'create_entry')
        self.assertEqual(command['kind'], 'task')
        self.assertEqual(command['time'], '19:30')

    def test_clock_expression_cannot_justify_model_proposed_date(self):
        command = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.task,
            title='Подготовить протокол', date=datetime.date(2026, 10, 29),
            date_phrase='к 19:30', date_relation=DateRelation.exact,
            time='19:30', time_phrase='к 19:30',
        )
        verified = verify_creation_fields(
            command, 'Подготовить протокол к 19:30', datetime.date(2026, 10, 29),
        )
        self.assertIsNone(verified.date)
        self.assertIsNone(verified.date_unverified)
        self.assertEqual(verified.time, '19:30')

    def test_entry_service_rejects_time_without_date(self):
        service = EntryService(object())
        kwargs = dict(kind=EntryKind.task, title='Отправить отчёт',
                      date=None, time='11:15', timezone='Europe/Moscow')
        with self.assertRaisesRegex(ValueError, 'Время требует даты'):
            service.create(**kwargs)
        with self.assertRaisesRegex(ValueError, 'Время требует даты'):
            service.create_once(**kwargs)

    def test_schedule_year_is_not_restored_into_title(self):
        self.assertEqual(
            restore_numbers('Про презентации Сергею',
                            'Напомни 9 января 2027 в 20:15 про презентации Сергею'),
            'Про презентации Сергею',
        )
        self.assertEqual(
            restore_numbers('Подготовить отчёт за 2027 год',
                            'Подготовить отчёт за 2027 год 9 января 2028'),
            'Подготовить отчёт за 2027 год',
        )

    def test_number_in_name_is_still_restored(self):
        self.assertEqual(
            restore_numbers('Встреча с Иваном', 'Поставь встречу с Иваном 159516'),
            'Встреча с Иваном 159516',
        )

    def test_independent_review_can_recover_new_task_from_wrong_search(self):
        class Model:
            def __init__(self):
                self.responses = [
                    {'action': 'search_entries', 'query': 'аналитика'},
                    {
                        'action': 'create_entry', 'kind': 'task',
                        'title': 'Обновить материалы по аналитике',
                        'date': '2027-09-15', 'date_phrase': '2027-09-15',
                        'time': '11:30', 'time_phrase': 'в 11:30',
                    },
                ]

            def complete_json(self, **_):
                return self.responses.pop(0)

        model = Model()
        command = ExternalCommandExtractor(model, intent_review=True).extract(
            'Обновить материалы по аналитике 2027-09-15 в 11:30',
            'Europe/Moscow', datetime.date(2026, 10, 29),
        )
        self.assertEqual(command.action, AssistantIntent.create_entry)
        self.assertEqual(command.title, 'Обновить материалы по аналитике')
        self.assertEqual(command.date, datetime.date(2027, 9, 15))
        self.assertEqual(command.time, '11:30')
        self.assertEqual(model.responses, [])


if __name__ == '__main__':
    unittest.main()
