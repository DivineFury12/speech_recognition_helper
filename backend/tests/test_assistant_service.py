import datetime
import unittest
import uuid
from zoneinfo import ZoneInfo

import pydantic

from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, ConversationMessage, Entry, EntryKind, EntryStatus
from src.schemas.assistant_command import AssistantCommand
from src.services.assistant import AssistantService, InvalidTimezoneError
from src.services.command_extractor import ExternalCommandExtractor, InvalidModelResponseError
from src.services.temporal import parse_date, parse_time


class FakeExtractor:
    def __init__(self, command):
        self.command = command
        self.calls = []

    def extract(self, text, timezone, today, previous=None, history=None):
        self.calls.append((text, timezone, today, previous, history))
        if isinstance(self.command, list):
            return self.command.pop(0)
        return self.command


class FakeConversations:
    def __init__(self):
        self.states = {}

    def get(self, conversation_id):
        return self.states.get(conversation_id)

    def save_turn(self, turn, pending_command):
        previous = self.states.get(turn.conversation_id)
        messages = list(previous.messages) if previous else []
        for role, text in (('user', turn.user_text), ('assistant', turn.assistant_text)):
            messages.append(ConversationMessage(
                id=uuid.uuid4(), role=role, text=text,
                created_at=datetime.datetime.now(datetime.timezone.utc),
            ))
        self.states[turn.conversation_id] = ConversationState(
            id=turn.conversation_id, pending_command=pending_command, messages=messages,
        )

    def list_messages(self, conversation_id):
        state = self.get(conversation_id)
        return state.messages if state else None


class FakeEntries:
    def __init__(self):
        self.created = []

    def create(self, *, kind, title, date, time, timezone):
        self.created.append((kind, title, date, time, timezone))
        return Entry(
            id=uuid.uuid4(), kind=kind, title=title, date=date, time=time,
            timezone=timezone, status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def create_once(self, *, kind, title, date, time, timezone):
        return self.create(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
        ), True


class FakeModelClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append((system_prompt, user_prompt, json_schema))
        if isinstance(self.payload, list):
            return self.payload.pop(0)
        return self.payload


class AssistantServiceTests(unittest.TestCase):
    def test_command_schema_accepts_expected_fields(self):
        command = AssistantCommand.model_validate({
            'action': 'create_entry',
            'kind': 'meeting',
            'title': '  Встреча с Иваном  ',
            'date': '2026-10-12',
            'time': '15:30',
        })
        self.assertEqual(command.title, 'Встреча с Иваном')
        self.assertEqual(command.date, datetime.date(2026, 10, 12))
        self.assertEqual(
            set(AssistantCommand.provider_schema()['required']),
            set(AssistantCommand.model_fields),
        )

    def test_command_schema_rejects_invalid_result(self):
        for payload in (
            {'action': 'delete_everything'},
            {'action': 'create_entry', 'kind': 'meeting', 'time': '25:99'},
            {'action': 'search_entries', 'query': 'тема', 'extra': 'unexpected'},
            {'action': 'list_entries', 'date_from': '2026-10-12', 'date_to': '2026-10-10'},
        ):
            with self.subTest(payload=payload), self.assertRaises(pydantic.ValidationError):
                AssistantCommand.model_validate(payload)

    def test_service_returns_no_action_until_storage_is_connected(self):
        command = AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'reminder',
            'title': 'Позвонить Ивану', 'date': '2026-10-12', 'time': '10:00',
        })
        extractor = FakeExtractor(command)
        service = AssistantService(extractor)
        conversation_id = uuid.uuid4()

        turn = service.handle_message('Напомни позвонить Ивану', 'Europe/Moscow', conversation_id)

        self.assertEqual(turn.conversation_id, conversation_id)
        self.assertEqual(turn.intent, AssistantIntent.create_entry)
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertEqual(turn.entries, [])
        self.assertIn('не выполнено', turn.assistant_text)
        self.assertEqual(extractor.calls[0][:2], ('Напомни позвонить Ивану', 'Europe/Moscow'))
        self.assertIsInstance(extractor.calls[0][2], datetime.date)

    def test_service_reports_missing_fields_without_claiming_completion(self):
        extractor = FakeExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'С Иваном',
        }))
        turn = AssistantService(extractor).handle_message('Встреча с Иваном', 'Europe/Moscow', None)
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertIn('дату', turn.assistant_text)
        self.assertIn('время', turn.assistant_text)

    def test_clarification_uses_saved_command_and_executes_once(self):
        extractor = FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Сделать тренировку', 'date': '2026-10-16',
            }),
            AssistantCommand.model_validate({
                'action': 'create_entry', 'time': '18:00', 'time_phrase': 'в 18:00',
            }),
        ])
        entries = FakeEntries()
        conversations = FakeConversations()
        service = AssistantService(extractor, entries, conversations)

        first = service.handle_message(
            'В пятницу напомни сделать тренировку', 'Europe/Moscow', None,
        )
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        self.assertEqual(entries.created, [])
        self.assertIsNotNone(conversations.get(first.conversation_id).pending_command)

        second = service.handle_message('В 18:00', 'Europe/Moscow', first.conversation_id)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(len(entries.created), 1)
        self.assertEqual(entries.created[0][3], '18:00')
        self.assertIsNone(conversations.get(first.conversation_id).pending_command)
        self.assertEqual(len(conversations.list_messages(first.conversation_id)), 4)
        self.assertIsNotNone(extractor.calls[1][3])

    def test_pending_command_can_be_cancelled(self):
        extractor = FakeExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
        }))
        entries = FakeEntries()
        conversations = FakeConversations()
        service = AssistantService(extractor, entries, conversations)
        first = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        cancelled = service.handle_message('Отмена', 'Europe/Moscow', first.conversation_id)
        self.assertEqual(cancelled.status, AssistantStatus.no_action)
        self.assertEqual(len(extractor.calls), 1)
        self.assertIsNone(conversations.get(first.conversation_id).pending_command)
        self.assertEqual(entries.created, [])

    def test_service_rejects_unknown_timezone(self):
        extractor = FakeExtractor(AssistantCommand(action='unknown'))
        with self.assertRaises(InvalidTimezoneError):
            AssistantService(extractor).handle_message('Привет', 'Not/A_Zone', None)
        self.assertEqual(extractor.calls, [])

    def test_external_extractor_sends_context_and_validates_result(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'task', 'title': 'Купить билеты',
        })
        command = ExternalCommandExtractor(client).extract(
            'Создай задачу купить билеты', 'Europe/Moscow', datetime.date(2026, 10, 10),
        )
        self.assertEqual(command.title, 'Купить билеты')
        self.assertIn('Создай задачу купить билеты', client.calls[0][1])
        self.assertIn('2026-10-10', client.calls[0][1])
        self.assertIn('properties', client.calls[0][2])

    def test_model_date_uses_source_phrase_and_ignores_invented_time(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder',
            'title': 'Сделать тренировку', 'date': '2026-10-13',
            'date_phrase': 'в пятницу', 'time': '18:00', 'time_phrase': None,
        })
        command = ExternalCommandExtractor(client).extract(
            'В пятницу напомни сделать тренировку',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertEqual(command.date, datetime.date(2026, 10, 16))
        self.assertIsNone(command.time)

    def test_explicit_weekday_is_recognized_when_model_omits_date_phrase(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder',
            'title': 'про тренировку', 'date': None, 'date_phrase': None,
            'time': None, 'time_phrase': None,
        })
        command = ExternalCommandExtractor(client).extract(
            'Напомни в пятницу про тренировку',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertEqual(command.date, datetime.date(2026, 10, 16))
        self.assertIsNone(command.time)

    def test_model_cannot_invent_date_for_undated_task(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'task', 'title': 'Разобраться с уроками',
            'date': '2026-10-11', 'time': '18:00',
        })
        command = ExternalCommandExtractor(client).extract(
            'Создай задачу разобраться с уроками',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertIsNone(command.date)
        self.assertIsNone(command.time)

    def test_model_time_is_recomputed_from_user_phrase(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
            'date': '2026-10-12', 'date_phrase': 'завтра',
            'time': '00:00', 'time_phrase': 'в 16:00',
        })
        command = ExternalCommandExtractor(client).extract(
            'Создай встречу с Иваном завтра в 16:00',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertEqual(command.date, datetime.date(2026, 10, 12))
        self.assertEqual(command.time, '16:00')

    def test_explicit_time_survives_missing_model_phrase(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder',
            'title': 'Сварить гречку', 'date': None, 'time': None,
            'date_phrase': None, 'time_phrase': None,
        })
        command = ExternalCommandExtractor(client).extract(
            'Создай напоминание на завтра на 16:00 сварить гречку',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertEqual(command.title, 'Сварить гречку')
        self.assertEqual(command.date, datetime.date(2026, 10, 12))
        self.assertEqual(command.time, '16:00')

    def test_spoken_time_does_not_turn_date_or_duration_into_clock_time(self):
        self.assertEqual(parse_time('Создай напоминание на пять утра'), '05:00')
        self.assertEqual(parse_time('На 4 часа вечера, пожалуйста'), '16:00')
        self.assertEqual(parse_time('3 ноября на 6 утра'), '06:00')
        self.assertEqual(parse_time('В четверг в 11 встреча'), '11:00')
        self.assertIsNone(parse_time('на 3 ноября'))
        self.assertIsNone(parse_time('через два дня'))

    def test_relative_week_and_date_arithmetic(self):
        today = datetime.date(2026, 10, 11)
        for phrase, expected in (
            ('через неделю', datetime.date(2026, 10, 18)),
            ('через две недели', datetime.date(2026, 10, 25)),
            ('через 7 дней', datetime.date(2026, 10, 18)),
            ('на текущую дату + 7 дней', datetime.date(2026, 10, 18)),
            ('сегодня плюс семь дней', datetime.date(2026, 10, 18)),
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_date(phrase, today), expected)

    def test_week_later_reminder_keeps_title_and_evening_time(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder',
            'title': 'посмотреть фильм', 'date': None, 'date_phrase': None,
            'time': '18:00', 'time_phrase': 'в 6 вечера',
        })
        command = ExternalCommandExtractor(client).extract(
            'поставь напоменание посмотреть фильм через неделю в 6 вечера',
            'Europe/Moscow', datetime.date(2026, 10, 11),
        )
        self.assertEqual(command.title, 'посмотреть фильм')
        self.assertEqual(command.date, datetime.date(2026, 10, 18))
        self.assertEqual(command.time, '18:00')

    def test_date_only_reply_completes_pending_reminder_even_if_model_says_unknown(self):
        expected_date = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date() + datetime.timedelta(days=7)
        client = FakeModelClient([
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'посмотреть фильм', 'date': None,
                'time': '18:00', 'time_phrase': 'в 6 вечера',
            },
            {'action': 'unknown', 'title': 'текущую дату'},
        ])
        entries = FakeEntries()
        service = AssistantService(
            ExternalCommandExtractor(client), entries, FakeConversations(),
        )
        first = service.handle_message(
            'Поставь напоминание посмотреть фильм в 6 вечера',
            'Europe/Moscow', None,
        )
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        self.assertIn('дату', first.assistant_text)

        second = service.handle_message(
            'на текущую дату + 7 дней', 'Europe/Moscow', first.conversation_id,
        )
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][1:4], (
            'посмотреть фильм', expected_date, '18:00',
        ))

    def test_missing_title_is_not_filled_from_model_hallucination(self):
        client = FakeModelClient([
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Письмо другу', 'time': '05:00',
                'time_phrase': 'на пять утра',
            },
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Письмо другу', 'date': '2026-11-03',
                'date_phrase': '3 ноября', 'time': '06:00',
                'time_phrase': 'на 6 утра',
            },
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Написать письмо другу',
            },
        ])
        entries = FakeEntries()
        service = AssistantService(
            ExternalCommandExtractor(client), entries, FakeConversations(),
        )
        first = service.handle_message(
            'Создай напоминание на пять утра', 'Europe/Moscow', None,
        )
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        self.assertIn('назвать', first.assistant_text)
        self.assertEqual(entries.created, [])

        second = service.handle_message(
            '3 ноября на 6 утра', 'Europe/Moscow', first.conversation_id,
        )
        self.assertEqual(second.status, AssistantStatus.needs_clarification)
        self.assertIn('назвать', second.assistant_text)
        self.assertEqual(entries.created, [])

        third = service.handle_message(
            'Напомни написать письмо другу', 'Europe/Moscow', first.conversation_id,
        )
        self.assertEqual(third.status, AssistantStatus.completed)
        self.assertEqual(len(entries.created), 1)
        self.assertEqual(entries.created[0][1:4], (
            'Написать письмо другу', datetime.date(2026, 11, 3), '06:00',
        ))

    def test_external_extractor_rejects_bad_model_json(self):
        client = FakeModelClient({'action': 'delete_everything'})
        with self.assertRaises(InvalidModelResponseError):
            ExternalCommandExtractor(client).extract(
                'Удалить всё', 'Europe/Moscow', datetime.date(2026, 10, 10),
            )


if __name__ == '__main__':
    unittest.main()
