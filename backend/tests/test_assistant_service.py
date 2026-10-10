import datetime
import unittest
import uuid

import pydantic

from src.schemas.assistant import AssistantIntent, AssistantStatus
from src.schemas.assistant_command import AssistantCommand
from src.services.assistant import AssistantService, InvalidTimezoneError
from src.services.command_extractor import ExternalCommandExtractor, InvalidModelResponseError


class FakeExtractor:
    def __init__(self, command):
        self.command = command
        self.calls = []

    def extract(self, text, timezone, today):
        self.calls.append((text, timezone, today))
        return self.command


class FakeModelClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append((system_prompt, user_prompt, json_schema))
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

    def test_external_extractor_rejects_bad_model_json(self):
        client = FakeModelClient({'action': 'delete_everything'})
        with self.assertRaises(InvalidModelResponseError):
            ExternalCommandExtractor(client).extract(
                'Удалить всё', 'Europe/Moscow', datetime.date(2026, 10, 10),
            )


if __name__ == '__main__':
    unittest.main()
