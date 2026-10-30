"""Without GigaChat the assistant still handles the task's phrases with rules."""

import datetime
import unittest

from src.adapters.rules_model import FallbackModel, RulesModel
from src.schemas.assistant import AssistantStatus, EntryKind
from src.services import assistant as assistant_module
from src.services.assistant import AssistantService
from src.services.command_extractor import ExternalCommandExtractor
from src.services.entries import EntryService
from tests.test_our_features import NOW, TODAY, TZ, Conversations, RecurringRepository


class Broken:
    def complete_json(self, *args, **kwargs):
        raise RuntimeError('нет сети')

    def close(self):
        pass


def setUpModule():
    global TOKEN
    TOKEN = assistant_module._now.set(NOW)


def tearDownModule():
    assistant_module._now.reset(TOKEN)


class RulesTests(unittest.TestCase):
    def test_rules_only_command_is_not_marked_as_model_verified(self):
        command = ExternalCommandExtractor(RulesModel()).extract(
            'Создай задачу купить хлеб', TZ, TODAY,
        )
        self.assertFalse(command.signals_verified)

    def test_later_success_does_not_verify_fallback_intent(self):
        class FailsFirst:
            calls = 0

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('нет сети')
                return {name: None for name in json_schema['properties']}

            def close(self):
                pass

        primary = FailsFirst()
        model = FallbackModel(primary)
        command = ExternalCommandExtractor(model, temporal_retry=True).extract(
            'Создай задачу купить хлеб', TZ, TODAY,
        )
        self.assertEqual(primary.calls, 2)
        self.assertTrue(model.healthy)
        self.assertFalse(command.signals_verified)

    def chat(self, model=None):
        self.repository = RecurringRepository()
        service = AssistantService(ExternalCommandExtractor(model or RulesModel()),
                                   EntryService(self.repository), Conversations())
        state = {'cid': None}

        def say(text):
            turn = service.handle_message(text, TZ, state['cid'])
            state['cid'] = turn.conversation_id
            return turn
        return say

    def test_task_phrases(self):
        say = self.chat()
        reminder = say('Напомни мне завтра в 10 утра позвонить Ивану')
        self.assertEqual(reminder.status, AssistantStatus.completed)
        self.assertEqual(reminder.entries[0].kind, EntryKind.reminder)
        self.assertEqual(reminder.entries[0].title, 'Позвонить Ивану')
        self.assertEqual(reminder.entries[0].date, TODAY + datetime.timedelta(days=1))
        self.assertEqual(reminder.entries[0].time, '10:00')
        meeting = say('Добавь встречу с командой завтра в 15:00')
        self.assertEqual(meeting.entries[0].kind, EntryKind.meeting)
        listed = say('Что у меня завтра?')
        self.assertEqual(len(listed.entries), 2)
        found = say('Найди встречу с командой')
        self.assertEqual(found.entries[0].id, meeting.entries[0].id)
        asked = say('Удали встречу с командой')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertEqual(say('да').status, AssistantStatus.completed)

    def test_clarification_without_model(self):
        say = self.chat()
        asked = say('Поставь встречу с Иваном завтра')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        done = say('в 16:00')
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(done.entries[0].time, '16:00')

    def test_offline_new_task_interrupts_pending_reminder(self):
        say = self.chat()
        asked = say('Напомни позвонить Ивану')
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        created = say('Мне нужно купить молоко')
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].kind, EntryKind.task)
        self.assertEqual(created.entries[0].title, 'Купить молоко')

    def test_falls_back_when_gigachat_fails(self):
        model = FallbackModel(Broken())
        say = self.chat(model)
        turn = say('Напомни завтра в 9:00 купить хлеб')
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertFalse(model.healthy)
