import datetime
import json
import unittest
import uuid
from zoneinfo import ZoneInfo

import pydantic

from src.ports.conversations import ConversationState
from src.schemas.assistant import (AssistantIntent, AssistantStatus, AssistantTurn,
                                   ConversationMessage, Entry, EntryKind, EntryStatus)
from src.schemas.assistant_command import AssistantCommand, BulkDeleteRequest
from src.schemas.dialogue import (DialogueTransition, PendingReply, PendingStep,
                                  ReplyAct, transition_for_reply)
from src.services import assistant as assistant_module
from src.services.assistant import AssistantService, InvalidTimezoneError
from src.services.command_extractor import ExternalCommandExtractor, InvalidModelResponseError
from src.services.guards import requests_bulk_mutation
from src.services.temporal import (ambiguous_time_options, parse_date, parse_duration,
                                    parse_period, parse_target_date, parse_time,
                                   replacement_schedule_reply)


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

    def create(self, *, kind, title, date, time, timezone, duration_minutes=None):
        self.created.append((kind, title, date, time, timezone))
        return Entry(
            id=uuid.uuid4(), kind=kind, title=title, date=date, time=time,
            timezone=timezone, duration_minutes=duration_minutes, status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

    def create_once(self, *, kind, title, date, time, timezone, duration_minutes=None):
        return self.create(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            duration_minutes=duration_minutes,
        ), True

    def list_period(self, date_from, date_to, *, kind=None):
        return []

    def find_candidates(self, query, *, kind=None):
        return [], False


class FakeModelClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, json_schema):
        self.calls.append((system_prompt, user_prompt, json_schema))
        if 'from_phrase' in json_schema.get('properties', {}) and 'scope' not in json_schema['properties']:
            return {'date_from': None, 'date_to': None,
                    'from_phrase': None, 'to_phrase': None}
        if isinstance(self.payload, list):
            return self.payload.pop(0)
        return self.payload


class AssistantServiceTests(unittest.TestCase):
    def test_vague_schedule_constraints_survive_missing_model_phrases(self):
        command = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting',
            'title': 'встретиться с коллегой', 'date': None, 'date_phrase': None,
            'time': None, 'time_phrase': None,
        })).extract('планировал с коллегой на этой недели встретиться вечером когда нибудь',
                    'Europe/Moscow', datetime.date(2026, 10, 18))
        self.assertEqual(command.date_window_start, datetime.date(2026, 10, 12))
        self.assertEqual(command.date_window_end, datetime.date(2026, 10, 18))
        self.assertEqual(command.preferred_daypart, 'evening')

    def test_schedule_clarifications_keep_period_and_daypart_constraints(self):
        today = datetime.date(2026, 10, 18)
        first_text = 'планировал с коллегой на этой неделе встретиться вечером когда нибудь'
        first_command = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting',
            'title': 'встретиться с коллегой', 'date': None,
            'date_phrase': 'на этой неделе', 'date_relation': 'period',
            'time': None, 'time_phrase': 'вечером когда нибудь',
            'time_relation': 'ambiguous',
        })).extract(first_text, 'Europe/Moscow', today)
        self.assertEqual(first_command.date_window_end, datetime.date(2026, 10, 18))
        self.assertEqual(first_command.preferred_daypart, 'evening')
        self.assertEqual(parse_time('давай с утра на 7'), '07:00')
        self.assertIsNone(ambiguous_time_options('давай с утра на 7'))
        self.assertEqual(replacement_schedule_reply('давай с утра на 7', today),
                         (None, '07:00', None))
        entries = FakeEntries()
        conversations = FakeConversations()
        service = AssistantService(FakeExtractor([
            first_command,
            AssistantCommand(action=AssistantIntent.create_entry,
                             date=datetime.date(2026, 10, 26)),
            AssistantCommand(action=AssistantIntent.create_entry, time='07:00'),
        ]), entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message(first_text, 'Europe/Moscow', None)
            second = service.handle_message('26 числа', 'Europe/Moscow', first.conversation_id)
            third = service.handle_message('давай с утра на 7', 'Europe/Moscow', first.conversation_id)
            self.assertEqual(entries.created, [])
            self.assertIn('26.10', third.assistant_text)
            self.assertIn('07:00', third.assistant_text)
            self.assertIn('вечер', third.assistant_text)
            self.assertNotIn('07:00 или 19:00', third.assistant_text)
            fourth = service.handle_message('оставь 26 числа в 7 утра',
                                            'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        self.assertEqual(second.status, AssistantStatus.needs_clarification)
        self.assertEqual(third.status, AssistantStatus.needs_clarification)
        self.assertEqual(fourth.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][2:4], (datetime.date(2026, 10, 26), '07:00'))

    def test_explicit_weekday_overrides_wrong_model_date_without_confirmation(self):
        today = datetime.date(2026, 10, 18)
        text = 'Поставь встречу с командой в пятницу в 15:00'
        command = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'встречу с командой',
            'date': '2026-10-20', 'date_phrase': 'в пятницу', 'date_relation': 'exact',
            'time': '15:00', 'time_phrase': 'в 15:00', 'time_relation': 'exact',
        })).extract(text, 'Europe/Moscow', today)
        self.assertEqual(command.date, datetime.date(2026, 10, 23))
        self.assertIsNone(command.date_unverified)
        entries = FakeEntries()
        service = AssistantService(FakeExtractor(command), entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            turn = service.handle_message(text, 'Europe/Moscow', None)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][2:4], (datetime.date(2026, 10, 23), '15:00'))

    def test_schedule_clarification_accepts_a_corrected_date(self):
        conversations = FakeConversations()
        entries = FakeEntries()
        service = AssistantService(FakeExtractor(AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.meeting,
            title='Встреча с командой', date=datetime.date(2026, 10, 20),
            date_unverified=True, time='15:00',
        )), entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('Поставь встречу с командой во вторник в 15:00',
                                           'Europe/Moscow', None)
            self.assertNotIn('Ответьте «да» или «нет»', first.assistant_text)
            second = service.handle_message('Нет, в субботу', 'Europe/Moscow',
                                            first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][2:4], (datetime.date(2026, 10, 24), '15:00'))

    def test_yesterday_is_grounded_before_asking_for_missing_fields(self):
        today = datetime.date(2026, 10, 18)
        command = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder', 'title': None,
            'date': '2026-10-17', 'date_phrase': 'на вчера',
            'date_relation': 'exact',
        })).extract('на вчера оставь напоминание', 'Europe/Moscow', today)
        self.assertEqual(command.date, datetime.date(2026, 10, 17))
        self.assertIsNone(command.date_unverified)
        conversations = FakeConversations()
        service = AssistantService(FakeExtractor(command), FakeEntries(), conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            turn = service.handle_message('на вчера оставь напоминание',
                                          'Europe/Moscow', None)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.assistant_text, 'Как назвать запись?')
        self.assertEqual(conversations.get(turn.conversation_id).pending_command.date,
                         datetime.date(2026, 10, 17))

    def test_repeating_proposed_date_keeps_it_when_confirmation_is_pending(self):
        conversations = FakeConversations()
        conversation_id = uuid.uuid4()
        pending = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.reminder,
            date=datetime.date(2026, 10, 17), date_unverified=True,
            confirm='schedule',
        )
        conversations.save_turn(AssistantTurn(
            conversation_id=conversation_id, turn_id=uuid.uuid4(),
            user_text='на вчера оставь напоминание',
            assistant_text='Я понял: дата — 17 октября. Верно?',
            status=AssistantStatus.needs_clarification,
            intent=AssistantIntent.create_entry, entries=[],
        ), pending)
        service = AssistantService(FakeExtractor(pending), FakeEntries(), conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            turn = service.handle_message('ну на вчера', 'Europe/Moscow',
                                          conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.assistant_text, 'Как назвать запись?')
        saved = conversations.get(conversation_id).pending_command
        self.assertEqual(saved.date, datetime.date(2026, 10, 17))
        self.assertIsNone(saved.date_unverified)

    def test_repeating_only_date_does_not_confirm_unverified_time(self):
        conversations = FakeConversations()
        conversation_id = uuid.uuid4()
        pending = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.reminder,
            title='Позвонить другу', date=datetime.date(2026, 10, 17),
            time='19:00', date_unverified=True, time_unverified=True,
            confirm='schedule',
        )
        conversations.save_turn(AssistantTurn(
            conversation_id=conversation_id, turn_id=uuid.uuid4(),
            user_text='Напомни позвонить другу вчера вечером',
            assistant_text='Я понял: дата — 17 октября, время — 19:00. Верно?',
            status=AssistantStatus.needs_clarification,
            intent=AssistantIntent.create_entry, entries=[],
        ), pending)
        entries = FakeEntries()
        service = AssistantService(FakeExtractor(pending), entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            turn = service.handle_message('ну на вчера', 'Europe/Moscow',
                                          conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.status, AssistantStatus.needs_clarification)
        self.assertEqual(entries.created, [])
        self.assertTrue(conversations.get(conversation_id).pending_command.time_unverified)

    def test_natural_confirmation_uses_dialogue_context(self):
        class ContextualExtractor(FakeExtractor):
            def __init__(self, command):
                super().__init__(command)
                self.confirmation = None

            def classify_confirmation(self, text, proposal, *, history=None, pending=None):
                self.confirmation = (text, proposal, history, pending)
                return True

        extractor = ContextualExtractor(AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.reminder,
            title='Позвонить другу', date=datetime.date(2026, 10, 20),
            time='19:00', date_unverified=True,
        ))
        entries = FakeEntries()
        service = AssistantService(extractor, entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('Напомни позвонить другу 20 числа в 19:00',
                                           'Europe/Moscow', None)
            second = service.handle_message('естественно', 'Europe/Moscow',
                                            first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(len(entries.created), 1)
        self.assertEqual(extractor.confirmation[0], 'естественно')
        self.assertIn('20 октября', extractor.confirmation[1])
        self.assertEqual([message['role'] for message in extractor.confirmation[2]],
                         ['user', 'assistant'])
        self.assertEqual(extractor.confirmation[3].confirm, 'schedule')

    def test_false_new_command_classification_keeps_pending_meeting(self):
        class WrongNewCommand(FakeExtractor):
            def classify_pending_reply(self, *args, **kwargs):
                return PendingReply(act=ReplyAct.new_command)

        extractor = WrongNewCommand([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Встреча с другом', signals_verified=True),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='он сможет только 20 числа после шести',
                             signals_verified=True),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             date=datetime.date(2026, 10, 20), signals_verified=True),
        ])
        entries = FakeEntries()
        service = AssistantService(extractor, entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('хочу встретиться с другом на этой неделе',
                                           'Europe/Moscow', None)
            second = service.handle_message('он сможет только 20 числа после шести',
                                            'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.needs_clarification)
        self.assertEqual(entries.created, [])

    def test_clarification_is_not_a_new_task_without_action_evidence(self):
        class UnclearExtractor(FakeExtractor):
            def classify_pending_reply(self, *args, **kwargs):
                return PendingReply(act=ReplyAct.unclear)

        extractor = UnclearExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Встреча с другом', signals_verified=True),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='он сможет только 20 числа после шести',
                             date=datetime.date(2026, 10, 20), signals_verified=True),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             date=datetime.date(2026, 10, 20), date_phrase='20 числа',
                             time_phrase='после шести', signals_verified=True),
        ])
        entries = FakeEntries()
        service = AssistantService(extractor, entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('хочу встретиться с другом на этой неделе',
                                           'Europe/Moscow', None)
            second = service.handle_message('он сможет только 20 числа после шести',
                                            'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.needs_clarification,
                         second.assistant_text)
        self.assertIn('время', second.assistant_text)
        self.assertEqual(entries.created, [])

    def test_focused_schedule_retry_fills_only_grounded_missing_fields(self):
        client = FakeModelClient([
            {'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить другу',
             'date': None, 'date_phrase': None, 'time': None, 'time_phrase': None},
            {'date': '2026-10-25', 'date_start': 3, 'date_end': 4,
             'date_relation': 'exact', 'time': '19:30',
             'time_start': 5, 'time_end': 8, 'time_relation': 'exact'},
        ])
        command = ExternalCommandExtractor(client, temporal_retry=True).extract(
            'Напомни позвонить другу двадцать пятого в половине восьмого вечера',
            'Europe/Moscow', datetime.date(2026, 10, 18),
        )
        self.assertEqual(command.date, datetime.date(2026, 10, 25))
        self.assertEqual(command.time, '19:30')
        self.assertTrue(command.date_unverified)
        self.assertTrue(command.time_unverified)
        self.assertEqual(len(client.calls), 2)

    def test_unfamiliar_clarification_reaches_structured_extractor(self):
        class UnclearModelExtractor(FakeExtractor):
            def classify_pending_reply(self, *args, **kwargs):
                return PendingReply(act=ReplyAct.unclear)

            def extract(self, text, timezone, today, previous=None, history=None):
                self.calls.append((text, timezone, today, previous, history))
                if previous is not None:
                    return AssistantCommand(
                        action=AssistantIntent.create_entry,
                        date=datetime.date(2026, 10, 27),
                        date_phrase='в последний вторник октября', date_unverified=True,
                        signals_verified=True,
                    )
                if 'последний вторник октября' in text:
                    return AssistantCommand(action=AssistantIntent.unknown,
                                            signals_verified=True)
                return AssistantCommand(action=AssistantIntent.create_entry,
                                        kind=EntryKind.reminder, title='Позвонить другу',
                                        signals_verified=True)

        extractor = UnclearModelExtractor(None)
        conversations = FakeConversations()
        service = AssistantService(extractor, FakeEntries(), conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('Напомни позвонить другу', 'Europe/Moscow', None)
            second = service.handle_message('в последний вторник октября', 'Europe/Moscow',
                                            first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.assistant_text, 'На какое время поставить запись?')
        pending = conversations.get(first.conversation_id).pending_command
        self.assertEqual(pending.date, datetime.date(2026, 10, 27))
        self.assertTrue(pending.date_unverified)
        self.assertIsNone(pending.confirm)

    def test_unfamiliar_date_uses_model_evidence_but_requires_confirmation(self):
        today = datetime.date(2026, 10, 18)
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить другу',
            'date': '2026-10-25', 'date_phrase': 'двадцать пятого',
            'date_relation': 'exact',
            'time': '19:30', 'time_phrase': 'в половине восьмого вечера',
            'time_relation': 'exact',
        })
        command = ExternalCommandExtractor(client).extract(
            'Напомни позвонить другу двадцать пятого в половине восьмого вечера',
            'Europe/Moscow', today,
        )
        self.assertTrue(command.date_unverified)
        self.assertTrue(command.time_unverified)
        entries = FakeEntries()
        conversations = FakeConversations()
        service = AssistantService(FakeExtractor(command), entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message(
                'Напомни позвонить другу двадцать пятого в половине восьмого вечера',
                'Europe/Moscow', None,
            )
            self.assertEqual(entries.created, [])
            self.assertIn('25 октября', first.assistant_text)
            self.assertIn('19:30', first.assistant_text)
            second = service.handle_message('да', 'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][2:4], (datetime.date(2026, 10, 25), '19:30'))

    def test_model_boundary_or_ungrounded_schedule_cannot_create_exact_time(self):
        today = datetime.date(2026, 10, 18)
        boundary = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с другом',
            'date': '2026-10-20', 'date_phrase': '20 числа',
            'time': '18:00', 'time_phrase': 'после шести', 'time_relation': 'exact',
        })).extract('Встреча с другом 20 числа после шести', 'Europe/Moscow', today)
        self.assertEqual(boundary.date, datetime.date(2026, 10, 20))
        self.assertIsNone(boundary.time)
        interval = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с другом',
            'date': '2026-10-18', 'date_phrase': 'на этой неделе',
            'date_relation': 'exact',
        })).extract('Встреча с другом на этой неделе', 'Europe/Moscow', today)
        self.assertIsNone(interval.date)
        invented = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с другом',
            'date': '2026-10-25', 'date_phrase': 'двадцать пятого',
            'date_relation': 'exact',
            'time': '19:30', 'time_phrase': 'в половине восьмого вечера',
            'time_relation': 'exact',
        })).extract('Создай встречу с другом', 'Europe/Moscow', today)
        self.assertIsNone(invented.date)
        self.assertIsNone(invented.time)

    def test_natural_date_disagreement_uses_verified_calendar_date(self):
        today = datetime.date(2026, 10, 18)
        command = ExternalCommandExtractor(FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с другом',
            'date': '2026-10-20', 'date_phrase': 'в последний вторник октября',
            'date_relation': 'exact',
        })).extract('Встреча с другом в последний вторник октября',
                    'Europe/Moscow', today)
        self.assertEqual(parse_date('в последний вторник октября', today),
                         datetime.date(2026, 10, 27))
        self.assertEqual(parse_date('в первую среду ноября', today),
                         datetime.date(2026, 11, 4))
        self.assertEqual(command.date, datetime.date(2026, 10, 27))
        self.assertIsNone(command.date_unverified)

    def test_deadline_reply_keeps_date_and_asks_for_exact_reminder_time(self):
        today = datetime.date(2026, 10, 18)
        reply = 'время до 10:00 любое, 18.10 надо'
        self.assertIsNone(parse_time(reply))
        self.assertEqual(replacement_schedule_reply(reply, today), (today, None, None))

        class UnclearReplyExtractor(FakeExtractor):
            def classify_pending_reply(self, *args, **kwargs):
                return PendingReply(act=ReplyAct.unclear)

        extractor = UnclearReplyExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.reminder,
                             title='прогулки с собакой'),
            AssistantCommand(action=AssistantIntent.unknown),
        ])
        conversations = FakeConversations()
        service = AssistantService(extractor, FakeEntries(), conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 8, 0))
        try:
            first = service.handle_message('создай напоминание прогулки с собакой',
                                           'Europe/Moscow', None)
            second = service.handle_message(reply, 'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.needs_clarification)
        self.assertIn('10:00', second.assistant_text)
        self.assertIn('Во сколько', second.assistant_text)
        pending = conversations.get(first.conversation_id).pending_command
        self.assertEqual(pending.date, today)
        self.assertIsNone(pending.time)

    def test_day_of_month_and_time_boundary_continue_meeting_clarification(self):
        today = datetime.date(2026, 10, 18)
        reply = 'он сможет только 20 числа после шести'
        self.assertEqual(parse_date(reply, today), datetime.date(2026, 10, 20))
        self.assertIsNone(parse_time(reply))
        self.assertEqual(replacement_schedule_reply(reply, today),
                         (datetime.date(2026, 10, 20), None, None))
        self.assertEqual(parse_date('31 числа', datetime.date(2026, 2, 1)),
                         datetime.date(2026, 3, 31))

        class UnclearReplyExtractor(FakeExtractor):
            def classify_pending_reply(self, *args, **kwargs):
                return PendingReply(act=ReplyAct.unclear)

        extractor = UnclearReplyExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Встреча с другом'),
            AssistantCommand(action=AssistantIntent.unknown),
        ])
        conversations = FakeConversations()
        entries = FakeEntries()
        service = AssistantService(extractor, entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('хочу встретиться с другом на этой неделе',
                                           'Europe/Moscow', None)
            second = service.handle_message(reply, 'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(second.status, AssistantStatus.needs_clarification)
        self.assertIn('20 октября', second.assistant_text)
        self.assertIn('после шести', second.assistant_text)
        self.assertIn('Во сколько', second.assistant_text)
        self.assertEqual(entries.created, [])
        pending = conversations.get(first.conversation_id).pending_command
        self.assertEqual(pending.date, datetime.date(2026, 10, 20))
        self.assertIsNone(pending.time)

    def test_bulk_delete_previews_frozen_period_and_requires_confirmation(self):
        class StoredEntries(FakeEntries):
            def __init__(self):
                super().__init__()
                self.items = [Entry(
                    id=uuid.uuid4(), kind=EntryKind.meeting, title=title,
                    date=datetime.date(2026, 10, day), time='18:00',
                    timezone='Europe/Moscow', status=EntryStatus.pending,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                ) for day, title in ((20, 'С Иваном'), (22, 'С Максимом'), (30, 'Позже'))]

            def list_period(self, date_from, date_to, *, kind=None):
                return [entry for entry in self.items if date_from <= entry.date <= date_to
                        and (kind is None or kind == entry.kind)]

            def delete(self, entry_id):
                before = len(self.items)
                self.items = [entry for entry in self.items if entry.id != entry_id]
                return len(self.items) < before

        entries = StoredEntries()
        service = AssistantService(FakeExtractor(AssistantCommand(action=AssistantIntent.unknown)),
                                   entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            first = service.handle_message('удали все встречи за следующую неделю',
                                           'Europe/Moscow', None)
            self.assertEqual(first.status, AssistantStatus.needs_clarification)
            self.assertEqual(len(first.entries), 2)
            self.assertEqual(len(entries.items), 3)
            cancelled = service.handle_message('нет', 'Europe/Moscow', first.conversation_id)
            self.assertEqual(cancelled.status, AssistantStatus.no_action)
            self.assertEqual(len(entries.items), 3)
            second = service.handle_message('удали все встречи за следующую неделю',
                                            'Europe/Moscow', first.conversation_id)
            confirmed = service.handle_message('подтверждаю', 'Europe/Moscow', second.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(confirmed.status, AssistantStatus.completed)
        self.assertEqual(len(entries.items), 1)
        self.assertEqual(entries.items[0].title, 'Позже')

    def test_period_scoped_generic_delete_is_bulk_without_the_word_all(self):
        today = datetime.date(2026, 10, 18)
        for text in ('удали события за этот месяц', 'отмени встречи за следующую неделю',
                     'удали события которые в пятницу'):
            with self.subTest(text=text):
                self.assertTrue(requests_bulk_mutation(
                    text, has_period=parse_period(text, today) is not None,
                ))
        for text in ('удали встречу с Иваном за этот месяц',
                     'удали все встречи с Иваном за этот месяц'):
            with self.subTest(text=text):
                self.assertFalse(requests_bulk_mutation(
                    text, has_period=parse_period(text, today) is not None,
                ))
        entries = FakeEntries()
        entries.list_period = lambda start, end, *, kind=None: [Entry(
            id=uuid.uuid4(), kind=EntryKind.meeting, title='Встреча с командой',
            date=datetime.date(2026, 10, 23), time='15:00',
            timezone='Europe/Moscow', status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )]
        service = AssistantService(FakeExtractor(AssistantCommand(action=AssistantIntent.unknown)),
                                   entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 18, 12, 0))
        try:
            turn = service.handle_message('удали события за этот месяц', 'Europe/Moscow', None)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.status, AssistantStatus.needs_clarification)
        self.assertIn('Удалить 1 запись за период', turn.assistant_text)

    def test_bulk_delete_can_narrow_friday_entries_by_names_before_confirmation(self):
        class StoredEntries(FakeEntries):
            def __init__(self):
                super().__init__()
                self.items = [Entry(
                    id=uuid.uuid4(), kind=kind, title=title,
                    date=datetime.date(2026, 10, 23), time='18:00',
                    timezone='Europe/Moscow', status=EntryStatus.pending,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                ) for kind, title in (
                    (EntryKind.meeting, 'Встреча с коллегой'),
                    (EntryKind.reminder, 'Прогулка с собакой'),
                    (EntryKind.task, 'Подготовить отчёт'),
                )]

            def list_period(self, date_from, date_to, *, kind=None):
                return [entry for entry in self.items if date_from <= entry.date <= date_to
                        and (kind is None or entry.kind == kind)]

            def get(self, entry_id):
                return next((entry for entry in self.items if entry.id == entry_id), None)

            def delete(self, entry_id):
                before = len(self.items)
                self.items = [entry for entry in self.items if entry.id != entry_id]
                return len(self.items) < before

        class SelectingExtractor(FakeExtractor):
            def classify_bulk_selection(self, reply, candidates):
                self.calls.append(('selection', reply))
                if reply == 'встречу с коллегой и прогулку с собакой':
                    return 'select', candidates[:2]
                if reply == 'согласен удалить эти записи':
                    return 'confirm_all', []
                return 'unclear', []

        entries = StoredEntries()
        conversations = FakeConversations()
        extractor = SelectingExtractor(AssistantCommand(action=AssistantIntent.unknown))
        service = AssistantService(extractor, entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 19, 12, 0))
        try:
            first = service.handle_message('удали события которые в пятницу',
                                           'Europe/Moscow', None)
            self.assertEqual(first.status, AssistantStatus.needs_clarification)
            self.assertEqual(len(first.entries), 3)
            self.assertEqual(len(entries.items), 3)
            narrowed = service.handle_message('встречу с коллегой и прогулку с собакой',
                                              'Europe/Moscow', first.conversation_id)
            self.assertEqual(narrowed.status, AssistantStatus.needs_clarification)
            self.assertEqual(len(narrowed.entries), 2)
            self.assertEqual(len(entries.items), 3)
            confirmed = service.handle_message('согласен удалить эти записи',
                                               'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(confirmed.status, AssistantStatus.completed)
        self.assertEqual([entry.title for entry in entries.items], ['Подготовить отчёт'])
        self.assertEqual(extractor.calls, [
            ('selection', 'встречу с коллегой и прогулку с собакой'),
            ('selection', 'согласен удалить эти записи'),
        ])

    def test_bulk_selection_classifier_only_returns_existing_candidates(self):
        entries = [Entry(
            id=uuid.uuid4(), kind=EntryKind.meeting, title=title,
            date=datetime.date(2026, 10, 23), time='18:00',
            timezone='Europe/Moscow', status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        ) for title in ('Встреча с коллегой', 'Прогулка с собакой')]
        client = FakeModelClient({'mode': 'select', 'selected_numbers': [2, 1, 2]})
        mode, selected = ExternalCommandExtractor(client).classify_bulk_selection(
            'встречу с коллегой и прогулку с собакой', entries,
        )
        self.assertEqual(mode, 'select')
        self.assertEqual(selected, entries)
        client.payload = {'mode': 'confirm_all', 'selected_numbers': []}
        self.assertEqual(ExternalCommandExtractor(client).classify_bulk_selection(
            'согласен удалить все перечисленные записи', entries,
        ), ('confirm_all', []))
        client.payload = {'mode': 'select', 'selected_numbers': [3]}
        with self.assertRaises(InvalidModelResponseError):
            ExternalCommandExtractor(client).classify_bulk_selection('третью', entries)

    def test_bulk_delete_uses_model_period_for_typo_and_dash_ranges(self):
        class StoredEntries(FakeEntries):
            def __init__(self):
                super().__init__()
                self.items = [Entry(
                    id=uuid.uuid4(), kind=EntryKind.task, title=title,
                    date=datetime.date(2026, month, day), time=None,
                    timezone='Europe/Moscow', status=EntryStatus.pending,
                    created_at=datetime.datetime.now(datetime.timezone.utc),
                ) for month, day, title in (
                    (9, 5, 'Первое сентябрьское'), (9, 10, 'Второе сентябрьское'),
                    (10, 11, 'Первое октябрьское'), (10, 15, 'Второе октябрьское'),
                )]

            def list_period(self, date_from, date_to, *, kind=None):
                return [entry for entry in self.items if date_from <= entry.date <= date_to]

        class PeriodExtractor(FakeExtractor):
            def __init__(self, payload):
                super().__init__(AssistantCommand(action=AssistantIntent.unknown))
                self.model = ExternalCommandExtractor(FakeModelClient(payload))

            def classify_bulk_delete(self, text, today):
                return self.model.classify_bulk_delete(text, today)

        cases = (
            ('удали события с 5 о 10 сентября', {
                'scope': 'bulk', 'date_from': '2026-09-05', 'date_to': '2026-09-10',
                'from_phrase': '5', 'to_phrase': '10 сентября',
            }, 'сентябрьское'),
            ('удали события с 11.10 -15.10', {
                'scope': 'bulk', 'date_from': '2026-10-11', 'date_to': '2026-10-15',
                'from_phrase': '11.10', 'to_phrase': '15.10',
            }, 'октябрьское'),
        )
        token = assistant_module._now.set(datetime.datetime(2026, 10, 19, 12, 0))
        try:
            for text, payload, title_suffix in cases:
                with self.subTest(text=text):
                    entries = StoredEntries()
                    service = AssistantService(PeriodExtractor(payload), entries, FakeConversations())
                    turn = service.handle_message(text, 'Europe/Moscow', None)
                    self.assertEqual(turn.status, AssistantStatus.needs_clarification)
                    self.assertEqual(len(turn.entries), 2)
                    self.assertTrue(all(title_suffix in entry.title for entry in turn.entries))
                    self.assertEqual(len(entries.items), 4)
        finally:
            assistant_module._now.reset(token)

    def test_bulk_period_rejects_collapsed_or_ungrounded_model_dates(self):
        text = 'удали события с 11.10 -15.10'
        payload = {
            'scope': 'bulk', 'date_from': '2027-10-11', 'date_to': '2027-10-11',
            'from_phrase': '11.10', 'to_phrase': '15.10',
        }
        client = FakeModelClient(payload)
        with self.assertRaises(InvalidModelResponseError):
            ExternalCommandExtractor(client).classify_bulk_delete(
                text, datetime.date(2026, 10, 19),
            )
        client.payload = {**payload, 'date_to': '2027-10-15', 'to_phrase': 'несуществующая дата'}
        with self.assertRaises(InvalidModelResponseError):
            ExternalCommandExtractor(client).classify_bulk_delete(
                text, datetime.date(2026, 10, 19),
            )
        self.assertEqual(set(BulkDeleteRequest.provider_schema()['required']),
                         set(BulkDeleteRequest.model_fields))

    def test_list_and_search_resolve_unfamiliar_ranges_from_grounded_model_evidence(self):
        class PeriodClient:
            def complete_json(self, system_prompt, user_prompt, json_schema):
                text = json.loads(user_prompt)['text']
                if 'from_phrase' in json_schema['properties']:
                    if '11.10' in text:
                        return {
                            'date_from': '2026-10-11', 'date_to': '2026-10-15',
                            'from_phrase': '11.10', 'to_phrase': '15.10',
                        }
                    return {
                        'date_from': '2026-09-05', 'date_to': '2026-09-10',
                        'from_phrase': '5', 'to_phrase': '10 сентября',
                    }
                return {
                    'action': 'search_entries' if 'здоровье' in text else 'list_entries',
                    'query': 'здоровье' if 'здоровье' in text else None,
                    'date_from': '2026-10-11' if '11.10' in text else '2026-09-05',
                    'date_to': '2026-10-15' if '11.10' in text else '2026-09-10',
                }

        extractor = ExternalCommandExtractor(PeriodClient())
        today = datetime.date(2026, 10, 19)
        listed = extractor.extract('Покажи события с 5 о 10 сентября',
                                   'Europe/Moscow', today)
        self.assertEqual(listed.action, AssistantIntent.list_entries)
        self.assertEqual((listed.date_from, listed.date_to), (
            datetime.date(2026, 9, 5), datetime.date(2026, 9, 10),
        ))
        searched = extractor.extract('Найди события о здоровье с 11.10 -15.10',
                                     'Europe/Moscow', today)
        self.assertEqual(searched.action, AssistantIntent.search_entries)
        self.assertEqual(searched.query, 'здоровье')
        self.assertEqual((searched.date_from, searched.date_to), (
            datetime.date(2026, 10, 11), datetime.date(2026, 10, 15),
        ))

    def test_list_period_clarification_keeps_model_verified_range(self):
        from src.services.command_interpreter import CommandInterpreter

        class PeriodClient:
            def complete_json(self, system_prompt, user_prompt, json_schema):
                if 'from_phrase' in json_schema['properties']:
                    return {
                        'date_from': '2026-10-11', 'date_to': '2026-10-15',
                        'from_phrase': '11.10', 'to_phrase': '15.10',
                    }
                return {
                    'action': 'unknown', 'date_from': '2026-10-11',
                    'date_to': '2026-10-15',
                }

        previous = AssistantCommand(action=AssistantIntent.list_entries)
        text = 'с 11.10 -15.10'
        today = datetime.date(2026, 10, 19)
        update = ExternalCommandExtractor(PeriodClient()).extract(
            text, 'Europe/Moscow', today, previous=previous,
        )
        continued = CommandInterpreter._continuation_fields(previous, update, text, today)
        self.assertEqual((continued.date_from, continued.date_to), (
            datetime.date(2026, 10, 11), datetime.date(2026, 10, 15),
        ))

    def test_multiple_explicit_creations_use_the_same_entry_flow(self):
        entries = FakeEntries()
        extractor = FakeExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='купить молоко'),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='позвонить Ивану'),
        ])
        service = AssistantService(extractor, entries, FakeConversations())
        turn = service.handle_message('Создай задачу купить молоко и создай задачу позвонить Ивану',
                                      'Europe/Moscow', None)
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual([entry.title for entry in turn.entries],
                         ['Купить молоко', 'Позвонить Ивану'])
        self.assertEqual(len(extractor.calls), 2)

    def test_model_split_accepts_only_excerpts_and_carries_creation_action(self):
        text = 'Создай собеседование на среду и созвон с коллегой в четверг'
        client = FakeModelClient({'parts': [
            {'text': 'Создай собеседование на среду', 'inherit_action': False},
            {'text': 'созвон с коллегой в четверг', 'inherit_action': True},
        ]})
        extractor = ExternalCommandExtractor(client)
        self.assertEqual(extractor.split_create_requests(text), [
            'Создай собеседование на среду', 'Создай созвон с коллегой в четверг',
        ])
        client.payload = {'parts': [
            {'text': 'Создай собеседование на среду', 'inherit_action': False},
            {'text': 'созвон с другом в четверг', 'inherit_action': True},
        ]}
        self.assertEqual(extractor.split_create_requests(text), [text])

    def test_model_split_keeps_independent_second_action_after_greeting(self):
        text = ('Привет поставь мне созвон с коллегой на субботу в 9 утра, '
                'а в воскресенье в 8 вечера надо покормить черепаху')
        client = FakeModelClient({'parts': [
            {'text': 'Привет поставь мне созвон с коллегой на субботу в 9 утра',
             'inherit_action': False},
            {'text': 'в воскресенье в 8 вечера надо покормить черепаху',
             'inherit_action': False},
        ]})
        extractor = ExternalCommandExtractor(client)
        self.assertEqual(extractor.split_create_requests(text), [
            'Привет поставь мне созвон с коллегой на субботу в 9 утра',
            'в воскресенье в 8 вечера надо покормить черепаху',
        ])
        client.payload['parts'][0]['text'] = 'поставь мне созвон с коллегой на субботу в 9 утра'
        self.assertEqual(extractor.split_create_requests(text)[0],
                         'Привет поставь мне созвон с коллегой на субботу в 9 утра')

        class SplitExtractor(FakeExtractor):
            def split_create_requests(self, request):
                return extractor.split_create_requests(request)

        commands = SplitExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Созвон с коллегой', date=datetime.date(2026, 10, 24),
                             time='09:00'),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='покормить черепаху', date=datetime.date(2026, 10, 25),
                             time='20:00'),
        ])
        entries = FakeEntries()
        service = AssistantService(commands, entries, FakeConversations())
        token = assistant_module._now.set(datetime.datetime(2026, 10, 19, 12, 0))
        try:
            turn = service.handle_message(text, 'Europe/Moscow', None)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual([entry.title for entry in turn.entries],
                         ['Созвон с коллегой', 'Покормить черепаху'])
        self.assertEqual(len(turn.responses), 2)
        self.assertEqual([response.entries[0].kind for response in turn.responses],
                         [EntryKind.meeting, EntryKind.task])
        self.assertEqual([call[0] for call in commands.calls], [
            'Привет поставь мне созвон с коллегой на субботу в 9 утра',
            'в воскресенье в 8 вечера надо покормить черепаху',
        ])

    def test_batch_creation_resumes_after_each_clarification(self):
        class SplitExtractor(FakeExtractor):
            def split_create_requests(self, text):
                return ['Создай собеседование на среду',
                        'Создай созвон с коллегой в четверг']

        extractor = SplitExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Собеседование', date=datetime.date(2026, 10, 21)),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             time='16:00'),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Созвон с коллегой', date=datetime.date(2026, 10, 22)),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             time='15:00'),
        ])
        entries = FakeEntries()
        conversations = FakeConversations()
        service = AssistantService(extractor, entries, conversations)
        token = assistant_module._now.set(datetime.datetime(2026, 10, 19, 12, 0))
        try:
            first = service.handle_message(
                'Создай собеседование на среду и созвон с коллегой в четверг',
                'Europe/Moscow', None,
            )
            pending = conversations.get(first.conversation_id).pending_command
            self.assertEqual(pending.remaining_requests,
                             ['Создай созвон с коллегой в четверг'])
            self.assertEqual(
                AssistantCommand.model_validate(pending.model_dump(mode='json')).remaining_requests,
                pending.remaining_requests,
            )
            second = service.handle_message('на 16', 'Europe/Moscow', first.conversation_id)
            self.assertEqual(second.status, AssistantStatus.needs_clarification)
            self.assertEqual(len(entries.created), 1)
            self.assertIn('Собеседование', second.assistant_text)
            self.assertIn('время', second.assistant_text)
            self.assertEqual([response.status for response in second.responses],
                             [AssistantStatus.completed, AssistantStatus.needs_clarification])
            self.assertEqual([len(response.entries) for response in second.responses], [1, 0])
            third = service.handle_message('в 15', 'Europe/Moscow', first.conversation_id)
        finally:
            assistant_module._now.reset(token)
        self.assertEqual(third.status, AssistantStatus.completed)
        self.assertEqual([row[1] for row in entries.created],
                         ['Собеседование', 'Созвон с коллегой'])
        self.assertIsNone(conversations.get(first.conversation_id).pending_command)

    def test_time_reply_cannot_replace_pending_date_with_model_guess(self):
        previous = AssistantCommand(
            action=AssistantIntent.create_entry, kind=EntryKind.meeting,
            title='Собеседование', date=datetime.date(2026, 10, 21),
        )
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting',
            'date': '2026-10-25', 'date_phrase': 'на 16', 'date_relation': 'exact',
            'time': '16:00', 'time_phrase': '16', 'time_relation': 'exact',
        })
        command = ExternalCommandExtractor(client).extract(
            'на 16', 'Europe/Moscow', datetime.date(2026, 10, 19), previous=previous,
        )
        self.assertIsNone(command.date)
        self.assertIsNone(command.date_unverified)
        self.assertEqual(command.time, '16:00')

    def test_new_command_abandons_queued_batch(self):
        class SplitExtractor(FakeExtractor):
            def split_create_requests(self, text):
                return ['Создай собеседование на среду',
                        'Создай созвон с коллегой в четверг']

        extractor = SplitExtractor([
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.meeting,
                             title='Собеседование', date=datetime.date(2026, 10, 21)),
            AssistantCommand(action=AssistantIntent.create_entry, kind=EntryKind.task,
                             title='купить молоко'),
        ])
        entries = FakeEntries()
        service = AssistantService(extractor, entries, FakeConversations())
        first = service.handle_message(
            'Создай собеседование на среду и созвон с коллегой в четверг',
            'Europe/Moscow', None,
        )
        second = service.handle_message('Создай задачу купить молоко',
                                        'Europe/Moscow', first.conversation_id)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual([row[1] for row in entries.created], ['Купить молоко'])
        self.assertNotIn('созвон', second.assistant_text.casefold())

    def test_dialogue_transition_is_explicit_for_each_pending_step(self):
        self.assertEqual(transition_for_reply(PendingStep.fields, ReplyAct.select),
                         DialogueTransition.repeat_question)
        self.assertEqual(transition_for_reply(PendingStep.time_choice, ReplyAct.select),
                         DialogueTransition.continue_command)
        self.assertEqual(transition_for_reply(PendingStep.candidate, ReplyAct.reject),
                         DialogueTransition.continue_command)
        self.assertEqual(transition_for_reply(PendingStep.update_details, ReplyAct.new_command),
                         DialogueTransition.new_command)
        self.assertEqual(transition_for_reply(PendingStep.candidate, ReplyAct.cancel),
                         DialogueTransition.cancel)

    def test_pending_reply_classifier_requires_unambiguous_selection(self):
        client = FakeModelClient([
            {'act': 'select', 'choice': 2},
            {'act': 'select', 'choice': None},
            {'act': 'confirm', 'choice': 1},
        ])
        extractor = ExternalCommandExtractor(client)
        replies = [extractor.classify_pending_reply(
            'вторую', step=PendingStep.candidate, action=AssistantIntent.delete_entry,
            question='Какую запись?', candidates=[{'number': 1}, {'number': 2}],
        ) for _ in range(3)]
        self.assertEqual(replies[0], PendingReply(act=ReplyAct.select, choice=2))
        self.assertEqual(replies[1].act, ReplyAct.unclear)
        self.assertEqual(replies[2].act, ReplyAct.unclear)
        self.assertEqual(set(client.calls[0][2]['required']),
                         {'act', 'choice', 'schedule_role'})

    def test_confirmation_classifier_uses_small_validated_result(self):
        client = FakeModelClient([{'decision': 'yes'}, {'decision': 'no'},
                                  {'decision': 'unclear'}, {'decision': ['yes']}])
        extractor = ExternalCommandExtractor(client)
        self.assertIs(extractor.classify_confirmation('это именно нужная', 'Купить билеты'), True)
        self.assertIs(extractor.classify_confirmation('не та', 'Купить билеты'), False)
        self.assertIsNone(extractor.classify_confirmation('может быть', 'Купить билеты'))
        self.assertIsNone(extractor.classify_confirmation('сложно сказать', 'Купить билеты'))
        self.assertEqual(client.calls[0][2]['properties']['decision']['enum'],
                         ['yes', 'no', 'unclear'])

    def test_command_extractor_sends_selected_conversation_history(self):
        client = FakeModelClient({'action': 'unknown'})
        history = []
        for index in range(8):
            topic = 'собеседование' if index == 1 else f'поездка {index}'
            history.extend([
                {'role': 'user', 'text': f'Обсудим {topic}'},
                {'role': 'assistant', 'text': f'Запомнил: {topic}'},
            ])
        ExternalCommandExtractor(client).extract(
            'Что с собеседованием?', 'Europe/Moscow', datetime.date(2026, 10, 18),
            history=history,
        )
        selected = json.loads(client.calls[0][1])['history']
        self.assertIn(history[2], selected)
        self.assertEqual(selected[-2:], history[-2:])
        self.assertNotIn(history[0], selected)
        self.assertLessEqual(len(selected), 12)

    def test_clarification_and_confirmation_use_selected_history(self):
        history = []
        for index in range(9):
            topic = 'Собеседование' if index == 1 else f'поездка {index}'
            history.extend([
                {'role': 'user', 'text': topic},
                {'role': 'assistant', 'text': f'Записано: {topic}'},
            ])
        client = FakeModelClient([
            {'act': 'continue_command', 'choice': None, 'schedule_role': 'none'},
            {'decision': 'yes'},
        ])
        extractor = ExternalCommandExtractor(client)
        pending = AssistantCommand(action=AssistantIntent.create_entry,
                                   kind=EntryKind.meeting, title='Собеседование')
        extractor.classify_pending_reply(
            'на завтра', step=PendingStep.fields, action=AssistantIntent.create_entry,
            question='На какую дату?', pending=pending, history=history,
        )
        extractor.classify_confirmation(
            'подтверждаю', 'Собеседование', pending=pending, history=history,
        )
        for _, user_prompt, _ in client.calls:
            selected = json.loads(user_prompt)['history']
            self.assertIn(history[2], selected)
            self.assertEqual(selected[-2:], history[-2:])
            self.assertLessEqual(len(selected), 12)

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
            set(AssistantCommand.model_fields) - {
                'candidate_ids', 'remaining_requests', 'selected_entry_id', 'move_intent', 'pending_step',
                'signals_verified', 'date_unverified', 'time_unverified',
                'date_window_start', 'date_window_end', 'preferred_daypart',
                'recurrence', 'confirm', 'confirm_entry_id', 'forced', 'shift_minutes',
                'remind_minutes', 'recurrence_until', 'priority',
            },
        )

    def test_mutation_extractor_verifies_old_and_new_dates(self):
        client = FakeModelClient({
            'action': 'update_entry', 'kind': 'meeting',
            'target_query': 'встречу с Иваном',
            'target_date': '2026-10-15', 'target_date_phrase': 'с четверга',
            'date': '2026-10-16', 'date_phrase': 'на пятницу',
            'time': '18:00', 'time_phrase': None,
        })
        command = ExternalCommandExtractor(client).extract(
            'Перенеси встречу с Иваном с четверга на пятницу',
            'Europe/Moscow', datetime.date(2026, 10, 14),
        )
        self.assertEqual(command.target_date, datetime.date(2026, 10, 15))
        self.assertEqual(command.date, datetime.date(2026, 10, 16))
        self.assertIsNone(command.time)
        for server_field in ('candidate_ids', 'selected_entry_id', 'move_intent',
                             'pending_step', 'signals_verified'):
            self.assertNotIn(server_field, client.calls[0][2]['properties'])

    def test_command_signals_must_appear_in_latest_message(self):
        client = FakeModelClient({
            'action': 'update_entry', 'action_phrase': 'перенеси',
            'target_reference_phrase': 'её', 'update_mode': 'move',
        })
        command = ExternalCommandExtractor(client).extract(
            'А перенеси её на завтра', 'Europe/Moscow', datetime.date(2026, 10, 15),
        )
        self.assertTrue(command.signals_verified)
        self.assertEqual(command.action_phrase, 'перенеси')
        self.assertEqual(command.target_reference_phrase, 'её')
        self.assertEqual(command.update_mode.value, 'move')

        client.payload = {
            'action': 'update_entry', 'action_phrase': 'удали',
            'target_reference_phrase': 'её', 'update_mode': 'move',
        }
        command = ExternalCommandExtractor(client).extract(
            'Перенеси встречу с Иваном', 'Europe/Moscow', datetime.date(2026, 10, 15),
        )
        self.assertIsNone(command.action_phrase)
        self.assertIsNone(command.target_reference_phrase)
        self.assertIsNone(command.update_mode)

    def test_named_record_is_not_treated_as_pronoun_reference(self):
        client = FakeModelClient({
            'action': 'update_entry', 'action_phrase': 'Перенеси',
            'target_reference_phrase': 'встречу с Аркадием',
            'target_query': None, 'kind': 'meeting', 'update_mode': 'move',
            'date': '2026-10-27', 'date_phrase': '27 октября',
            'time': '17:00', 'time_phrase': '17:00',
        })
        command = ExternalCommandExtractor(client).extract(
            'Перенеси встречу с Аркадием на 27 октября в 17:00',
            'Europe/Moscow', datetime.date(2026, 10, 18),
        )
        self.assertEqual(command.target_query, 'встречу с Аркадием')
        self.assertIsNone(command.target_reference_phrase)
        self.assertEqual(command.date, datetime.date(2026, 10, 27))
        self.assertEqual(command.time, '17:00')

    def test_mutation_ignores_unrelated_provider_fields(self):
        client = FakeModelClient({
            'action': 'update_entry', 'kind': 'meeting',
            'target_query': None, 'date_from': '2026-10-20',
            'date_to': '2026-10-20', 'query': 'встреча с Иваном',
        })
        command = ExternalCommandExtractor(client).extract(
            'А перенеси её', 'Europe/Moscow', datetime.date(2026, 10, 14),
        )
        self.assertEqual(command.action, AssistantIntent.update_entry)
        self.assertIsNone(command.query)
        self.assertIsNone(command.date_from)
        self.assertIsNone(command.date_to)
        self.assertIsNone(command.target_query)

    def test_mutation_ignores_kind_invented_by_model(self):
        client = FakeModelClient({
            'action': 'delete_entry', 'kind': 'task',
            'target_query': 'просмотр фильма',
        })
        command = ExternalCommandExtractor(client).extract(
            'Привет, можешь удалить запись на просмотр фильма?',
            'Europe/Moscow', datetime.date(2026, 10, 15),
        )
        self.assertIsNone(command.kind)
        self.assertEqual(command.target_query, 'просмотр фильма')

    def test_list_and_search_ignore_unrelated_model_fields(self):
        today = datetime.date(2026, 10, 14)
        client = FakeModelClient([
            {
                'action': 'list_entries', 'kind': 'meeting',
                'date_from': '2025-01-01', 'date_to': '2025-01-03',
                'title': 'Встреча', 'time': '12:00', 'query': 'встречи',
            },
            {
                'action': 'search_entries', 'kind': 'task',
                'query': 'проверочный проект', 'title': 'Проект',
                'date': '2026-10-15', 'date_from': '2026-10-01',
            },
        ])
        extractor = ExternalCommandExtractor(client)
        listed = extractor.extract('Покажи встречи на этой неделе', 'Europe/Moscow', today)
        self.assertEqual((listed.date_from, listed.date_to), (
            datetime.date(2026, 10, 12), datetime.date(2026, 10, 18),
        ))
        self.assertIsNone(listed.title)
        found = extractor.extract('Найди задачу про проверочный проект', 'Europe/Moscow', today)
        self.assertEqual(found.query, 'проверочный проект')
        self.assertIsNone(found.date)

    def test_vague_list_period_is_not_invented(self):
        today = datetime.date(2026, 10, 14)
        client = FakeModelClient({
            'action': 'list_entries', 'date_from': '2026-10-20',
            'date_to': '2026-10-14',
        })
        command = ExternalCommandExtractor(client).extract(
            'Покажи мои дела', 'Europe/Moscow', today,
        )
        self.assertIsNone(command.date_from)
        self.assertIsNone(command.date_to)
        self.assertEqual(parse_period('Что у меня завтра?', today), (
            datetime.date(2026, 10, 15), datetime.date(2026, 10, 15),
        ))
        self.assertEqual(parse_period('Задачи за прошлую неделю', today), (
            datetime.date(2026, 10, 5), datetime.date(2026, 10, 11),
        ))
        self.assertEqual(parse_period('Планы на неделю', today), (
            today, datetime.date(2026, 10, 20),
        ))
        self.assertEqual(parse_period('Что было с 10 по 15 октября?', today), (
            datetime.date(2026, 10, 10), datetime.date(2026, 10, 15),
        ))
        self.assertEqual(parse_period('Покажи задачи за 11 октября', today), (
            datetime.date(2026, 10, 11), datetime.date(2026, 10, 11),
        ))

    def test_periods_from_manual_scenarios(self):
        today = datetime.date(2026, 10, 17)
        expected = {
            'Какие есть напоминания на октябрь?': (datetime.date(2026, 10, 1),
                                                    datetime.date(2026, 10, 31)),
            'за окт.': (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)),
            'за 11  октября': (datetime.date(2026, 10, 11), datetime.date(2026, 10, 11)),
            'с 01.10 по 31.10': (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)),
            'за осень': (datetime.date(2026, 9, 1), datetime.date(2026, 11, 30)),
            'в этом месяце': (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)),
            'на следующий месяц': (datetime.date(2026, 11, 1), datetime.date(2026, 11, 30)),
            'какие встречи есть на ближайшую неделю': (today, datetime.date(2026, 10, 23)),
            'события во второй половине следующей недели':
                (datetime.date(2026, 10, 22), datetime.date(2026, 10, 25)),
            'события за следующую неделю':
                (datetime.date(2026, 10, 19), datetime.date(2026, 10, 25)),
        }
        for phrase, period in expected.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_period(phrase, today), period)

    def test_model_cannot_invent_period_or_topic(self):
        today = datetime.date(2026, 10, 17)
        client = FakeModelClient([
            {'action': 'search_entries', 'query': 'здоровье',
             'date_from': '2027-10-01', 'date_to': '2027-10-31'},
            {'action': 'list_entries', 'query': 'здоровье',
             'date_from': '2027-10-01', 'date_to': '2027-10-31'},
            {'action': 'search_entries', 'query': 'несуществующее',
             'date_from': '2027-10-01', 'date_to': '2027-10-31'},
        ])
        extractor = ExternalCommandExtractor(client)
        for action in (AssistantIntent.search_entries, AssistantIntent.list_entries):
            command = extractor.extract(
                'Выведи события, связанные со здоровьем, за октябрь', 'Europe/Moscow', today,
            )
            self.assertEqual(command.action, action)
            self.assertEqual(command.query, 'здоровье')
            self.assertEqual((command.date_from, command.date_to),
                             (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)))
        invalid = extractor.extract('Какие напоминания на октябрь?', 'Europe/Moscow', today)
        self.assertIsNone(invalid.query)
        self.assertEqual(invalid.kind, EntryKind.reminder)

    def test_kind_and_month_are_not_a_search_topic(self):
        client = FakeModelClient({
            'action': 'search_entries', 'kind': 'reminder',
            'query': 'напоминания на октябрь',
        })
        command = ExternalCommandExtractor(client).extract(
            'Какие есть напоминания на октябрь?', 'Europe/Moscow',
            datetime.date(2026, 10, 17),
        )
        self.assertIsNone(command.query)
        self.assertEqual(command.kind, EntryKind.reminder)
        self.assertEqual((command.date_from, command.date_to),
                         (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)))
        client.payload = {'action': 'search_entries', 'query': 'осень'}
        command = ExternalCommandExtractor(client).extract(
            'Покажи события за осень', 'Europe/Moscow', datetime.date(2026, 10, 17),
        )
        self.assertIsNone(command.query)
        self.assertEqual((command.date_from, command.date_to),
                         (datetime.date(2026, 9, 1), datetime.date(2026, 11, 30)))
        client.payload = {'action': 'search_entries', 'query': 'планируемые мероприятия'}
        command = ExternalCommandExtractor(client).extract(
            'Что у нас в этом месяце планируется?', 'Europe/Moscow',
            datetime.date(2026, 10, 18),
        )
        self.assertIsNone(command.query)
        self.assertEqual((command.date_from, command.date_to),
                         (datetime.date(2026, 10, 1), datetime.date(2026, 10, 31)))

    def test_search_relevance_selection_is_bounded_to_candidates(self):
        entries = [Entry(
            id=uuid.uuid4(), kind=EntryKind.task, title=title, date=None,
            time=None, timezone='Europe/Moscow', status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        ) for title in ('Поход к врачу', 'Встреча с Максимом', 'Просмотр фильмов')]
        client = FakeModelClient({'relevant_numbers': [1]})
        extractor = ExternalCommandExtractor(client)
        self.assertEqual(extractor.classify_relevant_entries('здоровье', entries), entries[:1])
        client.payload = {'relevant_numbers': [4]}
        with self.assertRaises(InvalidModelResponseError):
            extractor.classify_relevant_entries('здоровье', entries)

    def test_copy_reference_must_be_in_current_message(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': None,
            'time': '17:00', 'time_phrase': 'на 17 часов',
            'source_reference_phrase': 'с данными из прошлого сообщения',
        })
        extractor = ExternalCommandExtractor(client)
        command = extractor.extract(
            'Сделай ещё одну встречу с данными из прошлого сообщения, но только на 17 часов',
            'Europe/Moscow', datetime.date(2026, 10, 17),
        )
        self.assertEqual(command.source_reference_phrase, 'с данными из прошлого сообщения')
        self.assertEqual(command.time, '17:00')
        client.payload = {
            'action': 'create_entry', 'kind': 'meeting',
            'source_reference_phrase': 'из прошлого сообщения',
        }
        ungrounded = extractor.extract(
            'Создай встречу с Петром завтра', 'Europe/Moscow', datetime.date(2026, 10, 17),
        )
        self.assertIsNone(ungrounded.source_reference_phrase)

    def test_meeting_duration_is_read_from_user_words(self):
        today = datetime.date(2026, 10, 14)
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
            'date': '2026-10-15', 'time': '10:00', 'duration_minutes': 180,
            'duration_phrase': None,
        })
        command = ExternalCommandExtractor(client).extract(
            'Создай встречу с Иваном завтра в 10:00 на 30 минут',
            'Europe/Moscow', today,
        )
        self.assertEqual(command.duration_minutes, 30)
        self.assertIsNone(parse_duration('Создай встречу завтра в 10:00'))
        self.assertEqual(parse_duration('Создай встречу на полтора часа'), 90)
        self.assertEqual(parse_duration('Создай встречу на час и 30 минут'), 90)
        self.assertEqual(parse_duration('Перенеси её на 12:00 и сократи до 45 минут', for_update=True), 45)
        self.assertEqual(parse_duration('Сократи её до 30 минут', for_update=True), 30)
        self.assertIsNone(parse_duration('Перенеси её на час', for_update=True))

    def test_clock_time_is_not_also_meeting_duration(self):
        client = FakeModelClient({
            'action': 'create_entry', 'kind': 'meeting',
            'time': '17:00', 'time_phrase': 'на 17 часов',
            'duration_minutes': 1020, 'duration_phrase': None,
        })
        extractor = ExternalCommandExtractor(client)
        command = extractor.extract(
            'Сделай ещё одну встречу на 17 часов', 'Europe/Moscow',
            datetime.date(2026, 10, 17),
        )
        self.assertEqual(command.time, '17:00')
        self.assertIsNone(command.duration_minutes)
        client.payload = {
            'action': 'create_entry', 'kind': 'meeting',
            'time': '17:00', 'time_phrase': 'на 17 часов',
        }
        command = extractor.extract(
            'Сделай встречу на 17 часов длительностью два часа',
            'Europe/Moscow', datetime.date(2026, 10, 17),
        )
        self.assertEqual(command.duration_minutes, 120)

    def test_old_record_date_can_be_in_previous_week(self):
        today = datetime.date(2026, 10, 14)
        self.assertEqual(parse_target_date('вчера', today), datetime.date(2026, 10, 13))
        self.assertEqual(parse_target_date('в прошлую пятницу', today), datetime.date(2026, 10, 9))

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
        token = assistant_module._now.set(datetime.datetime(2026, 10, 15, 12, 0))
        try:
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
        finally:
            assistant_module._now.reset(token)

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

    def test_new_create_request_does_not_inherit_pending_schedule(self):
        extractor = FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить Ивану',
                'date': '2026-10-16',
            }),
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'Купить хлеб',
            }),
        ])
        entries = FakeEntries()
        service = AssistantService(extractor, entries, FakeConversations())
        first = service.handle_message('Напомни в пятницу позвонить Ивану',
                                       'Europe/Moscow', None)
        second = service.handle_message('Создай задачу купить хлеб',
                                        'Europe/Moscow', first.conversation_id)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual(entries.created[0][1:4], ('Купить хлеб', None, None))
        self.assertIsNone(extractor.calls[1][3])

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
        self.assertIsNone(parse_time('В четверг в 11 встреча'))
        self.assertEqual(ambiguous_time_options('в 11'), ('11:00', '23:00'))
        self.assertIsNone(parse_time('на 3 ноября'))
        self.assertIsNone(parse_time('на 15 минут'))
        self.assertIsNone(parse_time('через два дня'))

    def test_relative_week_and_date_arithmetic(self):
        today = datetime.date(2026, 10, 11)
        for phrase, expected in (
            ('через неделю', datetime.date(2026, 10, 18)),
            ('через две недели', datetime.date(2026, 10, 25)),
            ('через 7 дней', datetime.date(2026, 10, 18)),
            ('спустя неделю', datetime.date(2026, 10, 18)),
            ('через пару дней', datetime.date(2026, 10, 13)),
            ('через пару недель', datetime.date(2026, 10, 25)),
            ('на текущую дату + 7 дней', datetime.date(2026, 10, 18)),
            ('сегодня плюс семь дней', datetime.date(2026, 10, 18)),
        ):
            with self.subTest(phrase=phrase):
                self.assertEqual(parse_date(phrase, today), expected)

    def test_schedule_reply_accepts_relative_date_but_rejects_new_request(self):
        today = datetime.date(2026, 10, 11)
        self.assertEqual(replacement_schedule_reply('спустя неделю в 17:00', today),
                         (datetime.date(2026, 10, 18), '17:00', None))
        self.assertEqual(replacement_schedule_reply('через пару дней в 18:00', today),
                         (datetime.date(2026, 10, 13), '18:00', None))
        self.assertIsNone(replacement_schedule_reply('Что у меня завтра?', today))
        self.assertIsNone(replacement_schedule_reply('Создай встречу завтра в 16:00', today))

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
            {'act': 'continue_command', 'choice': None},
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
            'Посмотреть фильм', expected_date, '18:00',
        ))

    def test_missing_title_is_not_filled_from_model_hallucination(self):
        client = FakeModelClient([
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Письмо другу', 'time': '05:00',
                'time_phrase': 'на пять утра',
            },
            {'act': 'continue_command', 'choice': None},
            {
                'action': 'create_entry', 'kind': 'reminder',
                'title': 'Письмо другу', 'date': '2026-11-03',
                'date_phrase': '3 ноября', 'time': '06:00',
                'time_phrase': 'на 6 утра',
            },
            {'act': 'continue_command', 'choice': None},
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


PINNED_NOW = None


def setUpModule():
    """Pin «now» to 08:00 today so same-day times in fixtures are never in the past."""
    global PINNED_NOW
    import datetime as _dt
    from src.services import assistant as _assistant
    PINNED_NOW = _assistant._now.set(_dt.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0))


def tearDownModule():
    from src.services import assistant as _assistant
    _assistant._now.reset(PINNED_NOW)
