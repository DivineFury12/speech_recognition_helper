import datetime
import unittest
import uuid
from zoneinfo import ZoneInfo

import pydantic

from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, ConversationMessage, Entry, EntryKind, EntryStatus
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import (DialogueTransition, PendingReply, PendingStep,
                                  ReplyAct, transition_for_reply)
from src.services import assistant as assistant_module
from src.services.assistant import AssistantService, InvalidTimezoneError
from src.services.command_extractor import ExternalCommandExtractor, InvalidModelResponseError
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
                'candidate_ids', 'selected_entry_id', 'move_intent', 'pending_step',
                'signals_verified', 'recurrence', 'confirm', 'confirm_entry_id', 'forced', 'shift_minutes',
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
