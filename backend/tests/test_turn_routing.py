"""The external turn route must precede legacy semantic shortcuts."""

import datetime
import json
import unittest
import uuid

from src.adapters.rules_model import FallbackModel
from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, EntryKind
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import PendingReply, ReplyAct
from src.services import assistant as assistant_module
from src.services.assistant import AssistantService
from src.services.command_extractor import (ExternalCommandExtractor,
                                            InvalidModelResponseError, SpecialMutation,
                                            TurnDecision, TurnKind, TurnRoute)
from src.services.entries import EntryService
from tests.test_our_features import Conversations, NOW, RecurringRepository, TODAY, TZ


class RoutedExtractor:
    def __init__(self, route):
        self.route = route
        self.calls = []

    def classify_turn(self, text, timezone, today, *, pending=None, history=None):
        self.calls.append(('route', text, len(history or [])))
        if isinstance(self.route, Exception):
            raise self.route
        return self.route.get(text, TurnKind.command) if isinstance(self.route, dict) else self.route

    def split_requests(self, text):
        self.calls.append(('split', text, 0))
        return ['создай задачу А', 'создай задачу Б']

    def extract(self, *, text, timezone, today, previous=None, history=None):
        self.calls.append(('extract', text, len(history or [])))
        return AssistantCommand(action=AssistantIntent.unknown)


class TurnRoutingTests(unittest.TestCase):
    def setUp(self):
        self.token = assistant_module._now.set(NOW)

    def tearDown(self):
        assistant_module._now.reset(self.token)

    def test_unified_turn_extracts_action_with_one_model_call(self):
        class Primary:
            calls = 0

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls += 1
                self.assert_schema = set(json_schema['properties'])
                return {'kind': 'command', 'reply': None, 'command': {
                    'action': 'create_entry', 'kind': 'task',
                    'title': 'Проверить сроки по проекту',
                }}

        primary = Primary()
        extractor = ExternalCommandExtractor(primary, semantic_routing=True)
        decision = extractor.interpret_turn('Проверить сроки по проекту', TZ, TODAY)
        self.assertEqual(decision.kind, TurnKind.command)
        self.assertEqual(decision.command.action, AssistantIntent.create_entry)
        self.assertEqual(decision.command.kind, EntryKind.task)
        self.assertEqual(primary.calls, 1)
        self.assertEqual(primary.assert_schema, {'kind', 'command', 'reply'})

    def test_unified_pending_reply_extracts_fields_with_one_model_call(self):
        class Primary:
            calls = 0

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls += 1
                return {'kind': 'pending_reply',
                        'reply': {'act': 'continue_command', 'choice': None,
                                  'schedule_role': 'none'},
                        'command': {'action': 'create_entry', 'title': 'Купить хлеб'}}

        primary = Primary()
        extractor = ExternalCommandExtractor(primary, semantic_routing=True)
        decision = extractor.interpret_turn(
            'Купить хлеб', TZ, TODAY,
            pending=AssistantCommand(action=AssistantIntent.create_entry,
                                     kind=EntryKind.task),
        )
        self.assertEqual(decision.reply.act, ReplyAct.continue_command)
        self.assertEqual(decision.command.title, 'Купить хлеб')
        self.assertEqual(primary.calls, 1)

    def test_unified_search_does_not_repeat_intent_review(self):
        class Primary:
            calls = 0

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls += 1
                return {'kind': 'command', 'reply': None, 'command': {
                    'action': 'search_entries', 'query': 'проект',
                    'date_from': TODAY.isoformat(), 'date_to': TODAY.isoformat(),
                }}

        primary = Primary()
        decision = ExternalCommandExtractor(
            primary, intent_review=True, semantic_routing=True,
        ).interpret_turn('Покажи записи про проект сегодня', TZ, TODAY)
        self.assertEqual(decision.command.action, AssistantIntent.search_entries)
        self.assertEqual(primary.calls, 1)

    def test_model_command_prevents_lexical_undo(self):
        extractor = RoutedExtractor(TurnKind.command)
        turn = AssistantService(extractor, conversations=Conversations()).handle_message(
            'Отмени последнее', TZ, None,
        )
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertEqual([call[0] for call in extractor.calls], ['route', 'extract'])

    def test_model_undo_does_not_require_fixed_wording(self):
        extractor = RoutedExtractor(TurnKind.undo)
        turn = AssistantService(extractor, conversations=Conversations()).handle_message(
            'Верни сделанное только что', TZ, None,
        )
        self.assertIn('Отменять нечего', turn.assistant_text)
        self.assertEqual([call[0] for call in extractor.calls], ['route'])

    def test_wrong_undo_route_never_mutates_without_confirmation(self):
        repository = RecurringRepository()
        entry = repository.create(
            kind=EntryKind.task, title='Подготовить отчёт', date=TODAY,
            time=None, timezone=TZ, embedding=None, embedding_model='test',
        )
        conversations = Conversations()
        conversation_id = uuid.uuid4()
        conversations.set_undo(conversation_id, {'op': 'create', 'entry': entry.model_dump(mode='json')})
        extractor = RoutedExtractor({'Удали встречу': TurnKind.undo,
                                     'нет': TurnKind.pending_reply})
        service = AssistantService(extractor, EntryService(repository), conversations)
        preview = service.handle_message('Удали встречу', TZ, conversation_id)
        self.assertEqual(preview.status, AssistantStatus.needs_clarification)
        self.assertIsNotNone(repository.get(entry.id))
        declined = service.handle_message('нет', TZ, conversation_id)
        self.assertEqual(declined.status, AssistantStatus.no_action)
        self.assertIsNotNone(repository.get(entry.id))

    def test_undo_confirmation_rejects_changed_last_operation(self):
        repository = RecurringRepository()
        entry = repository.create(kind=EntryKind.task, title='Подготовить отчёт',
                                  date=TODAY, time=None, timezone=TZ,
                                  embedding=None, embedding_model='test')
        conversations = Conversations()
        conversation_id = uuid.uuid4()
        conversations.set_undo(conversation_id, {'op': 'create',
                                                 'entry': entry.model_dump(mode='json')})
        extractor = RoutedExtractor({'Отмени последнее': TurnKind.undo,
                                     'да': TurnKind.pending_reply})
        service = AssistantService(extractor, EntryService(repository), conversations)
        self.assertEqual(service.handle_message('Отмени последнее', TZ, conversation_id).status,
                         AssistantStatus.needs_clarification)
        conversations.set_undo(conversation_id, {'op': 'update',
                                                 'entry': entry.model_dump(mode='json')})
        turn = service.handle_message('да', TZ, conversation_id)
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertIn('изменилось', turn.assistant_text)
        self.assertIsNotNone(repository.get(entry.id))

    def test_bad_route_does_not_mutate_or_use_rules(self):
        extractor = RoutedExtractor(ValueError('bad route'))
        turn = AssistantService(extractor, conversations=Conversations()).handle_message(
            'Отмени последнее', TZ, None,
        )
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertIn('не изменены', turn.assistant_text)
        self.assertEqual([call[0] for call in extractor.calls], ['route'])

    def test_command_route_still_requires_delete_confirmation(self):
        class InconsistentExtractor(RoutedExtractor):
            def classify_turn_detailed(self, text, timezone, today, *, pending=None, history=None):
                return TurnRoute(kind=TurnKind.command)

            def extract(self, **kwargs):
                return AssistantCommand(action=AssistantIntent.delete_entry,
                                        target_query='отчёт')

        repository = RecurringRepository()
        entry = repository.create(kind=EntryKind.task, title='Подготовить отчёт',
                                  date=TODAY, time=None, timezone=TZ,
                                  embedding=None, embedding_model='test')
        service = AssistantService(InconsistentExtractor(TurnKind.command),
                                   EntryService(repository), Conversations())
        turn = service.handle_message('Удали отчёт', TZ, None)
        self.assertEqual(turn.status, AssistantStatus.needs_clarification)
        self.assertIn('Удалить', turn.assistant_text)
        self.assertIsNotNone(repository.get(entry.id))

    def test_route_receives_saved_context(self):
        extractor = RoutedExtractor(TurnKind.command)
        service = AssistantService(extractor, conversations=Conversations())
        first = service.handle_message('Привет', TZ, None)
        service.handle_message('А теперь покажи планы', TZ, first.conversation_id)
        self.assertIn(('route', 'А теперь покажи планы', 2), extractor.calls)

    def test_distant_reference_receives_persisted_entry_anchor(self):
        class CaptureExtractor(RoutedExtractor):
            history = None

            def classify_turn(self, text, timezone, today, *, pending=None, history=None):
                self.history = history
                return TurnKind.command

        repository = RecurringRepository()
        entry = repository.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                  date=TODAY, time='15:00', timezone=TZ,
                                  embedding=None, embedding_model='test')
        conversations = Conversations()
        conversation_id = uuid.uuid4()
        conversations.states[conversation_id] = ConversationState(
            id=conversation_id, pending_command=None, messages=[], last_entry_id=entry.id,
            summary='Ранее обсуждали встречу с Иваном.',
        )
        extractor = CaptureExtractor(TurnKind.command)
        AssistantService(extractor, EntryService(repository), conversations).handle_message(
            'А перенеси её на завтра', TZ, conversation_id,
        )
        self.assertEqual(extractor.history[0]['role'], 'summary')
        self.assertIn('Встреча с Иваном', extractor.history[-1]['text'])

    def test_pending_new_command_uses_model_route(self):
        extractor = RoutedExtractor(TurnKind.command)
        conversations = Conversations()
        conversation_id = uuid.uuid4()
        conversations.states[conversation_id] = ConversationState(
            id=conversation_id,
            pending_command=AssistantCommand(action=AssistantIntent.create_entry),
            messages=[],
        )
        turn = AssistantService(extractor, conversations=conversations).handle_message(
            'А теперь покажи планы', TZ, conversation_id,
        )
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertIsNone(conversations.get(conversation_id).pending_command)
        self.assertEqual([call[0] for call in extractor.calls], ['route', 'extract'])

    def test_model_batch_route_splits_before_individual_extraction(self):
        text = 'Создай задачу А и задачу Б'
        extractor = RoutedExtractor({text: TurnKind.batch})
        AssistantService(extractor, conversations=Conversations()).handle_message(
            text, TZ, None,
        )
        self.assertEqual([call[0] for call in extractor.calls],
                         ['route', 'split', 'route', 'extract'])

    def test_missed_batch_route_asks_model_to_check_independent_clauses(self):
        text = 'Создай задачу А и создай задачу Б'
        extractor = RoutedExtractor(TurnKind.command)
        AssistantService(extractor, conversations=Conversations()).handle_message(
            text, TZ, None,
        )
        self.assertEqual([call[0] for call in extractor.calls],
                         ['route', 'split', 'route', 'extract'])

    def test_unsplittable_model_batch_does_not_execute_first_part(self):
        text = 'Создай задачу А и задачу Б'
        extractor = RoutedExtractor({text: TurnKind.batch})
        extractor.split_requests = lambda value: [value]
        turn = AssistantService(extractor, conversations=Conversations()).handle_message(
            text, TZ, None,
        )
        self.assertEqual(turn.status, AssistantStatus.no_action)
        self.assertIn('не изменены', turn.assistant_text)
        self.assertEqual([call[0] for call in extractor.calls], ['route'])

    def test_generic_model_split_handles_distinct_actions(self):
        class Primary:
            def complete_json(self, system_prompt, user_prompt, json_schema):
                return {'parts': [
                    {'start_token': 0, 'end_token': 3, 'inherit_action': False},
                    {'start_token': 5, 'end_token': 8, 'inherit_action': False},
                ]}

            def close(self):
                pass

        extractor = ExternalCommandExtractor(FallbackModel(Primary()), semantic_routing=True)
        self.assertEqual(extractor.split_requests(
            'Удали встречу с Иваном и отметь задачу про отчёт',
        ), ['Удали встречу с Иваном', 'отметь задачу про отчёт'])

    def test_classifier_uses_fallback_only_after_primary_failure(self):
        class Primary:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls += 1
                return {'kind': 'command'}

            def close(self):
                pass

        primary = Primary()
        extractor = ExternalCommandExtractor(FallbackModel(primary), semantic_routing=True)
        route = extractor.classify_turn('Отмени встречу', TZ, TODAY)
        self.assertEqual(route, TurnKind.command)
        self.assertEqual(primary.calls, 1)

    def test_model_route_receives_pending_queue_and_chat_context(self):
        class Primary:
            request = None

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.request = json.loads(user_prompt)
                return {'kind': 'pending_reply'}

            def close(self):
                pass

        primary = Primary()
        extractor = ExternalCommandExtractor(FallbackModel(primary), semantic_routing=True)
        route = extractor.classify_turn(
            'в 18:00', TZ, TODAY,
            pending=AssistantCommand(action=AssistantIntent.create_entry,
                                     remaining_requests=['создай задачу купить хлеб']),
            history=[{'role': 'user', 'text': 'Создай напоминание и задачу'},
                     {'role': 'assistant', 'text': 'На какое время поставить запись?'}],
        )
        self.assertEqual(route, TurnKind.pending_reply)
        self.assertEqual(primary.request['pending_command']['remaining_requests'],
                         ['создай задачу купить хлеб'])
        self.assertEqual(len(primary.request['history']), 2)

    def test_special_change_uses_model_fields_after_route(self):
        class Primary:
            def __init__(self):
                self.calls = []

            def complete_json(self, system_prompt, user_prompt, json_schema):
                self.calls.append(set(json_schema['properties']))
                if set(json_schema['properties']) == {'kind', 'command', 'reply'}:
                    return {'kind': 'set_priority', 'command': None, 'reply': None}
                return {
                    'operation': 'set_priority',
                    'target_phrase': None,
                    'reference_phrase': 'этой записи',
                    'value_phrase': 'Повышай важность',
                    'priority': 'high',
                    'lead_minutes': None,
                }

            def close(self):
                pass

        repository = RecurringRepository()
        entry = repository.create(
            kind=EntryKind.task, title='Подготовить отчёт', date=TODAY,
            time=None, timezone=TZ, embedding=None, embedding_model='test',
        )
        conversations = Conversations()
        conversation_id = uuid.uuid4()
        conversations.states[conversation_id] = ConversationState(
            id=conversation_id, pending_command=None, messages=[], last_entry_id=entry.id,
        )
        primary = Primary()
        extractor = ExternalCommandExtractor(FallbackModel(primary), semantic_routing=True)
        turn = AssistantService(extractor, EntryService(repository), conversations).handle_message(
            'Повышай важность этой записи', TZ, conversation_id,
        )
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual(getattr(repository.get(entry.id).priority, 'value',
                                 repository.get(entry.id).priority), 'high')
        self.assertEqual(primary.calls[0], {'kind', 'command', 'reply'})
        self.assertEqual(primary.calls[1], {
            'operation', 'target_phrase', 'reference_phrase', 'value_phrase',
            'priority', 'lead_minutes',
        })

    def test_special_priority_asks_which_matching_record(self):
        class Extractor(RoutedExtractor):
            def __init__(self):
                super().__init__({'вторую': TurnKind.pending_reply})

            def classify_turn(self, text, timezone, today, *, pending=None, history=None):
                if text == 'вторую':
                    raise ValueError('invalid route')
                return TurnKind.command

            def extract(self, **kwargs):
                return AssistantCommand(action=AssistantIntent.update_entry,
                                        target_query='отчёт')

            def extract_special(self, text, kind, *, history=None):
                return SpecialMutation(operation='set_priority', target_phrase='отчёт',
                                       value_phrase='важной', priority='high')

            def classify_pending_reply(self, text, **kwargs):
                return PendingReply(act=ReplyAct.select, choice=2)

        repository = RecurringRepository()
        first = repository.create(kind=EntryKind.task, title='Отчёт по проекту',
                                  date=TODAY, time=None, timezone=TZ,
                                  embedding=None, embedding_model='test')
        second = repository.create(kind=EntryKind.task, title='Отчёт по бюджету',
                                   date=TODAY, time=None, timezone=TZ,
                                   embedding=None, embedding_model='test')
        service = AssistantService(Extractor(), EntryService(repository), Conversations())
        choice = service.handle_message('Сделай важной задачу про отчёт', TZ, None)
        self.assertEqual(choice.status, AssistantStatus.needs_clarification)
        self.assertNotEqual(getattr(repository.get(first.id).priority, 'value',
                                    repository.get(first.id).priority), 'high')
        done = service.handle_message('вторую', TZ, choice.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(getattr(repository.get(second.id).priority, 'value',
                                 repository.get(second.id).priority), 'high')

    def test_special_target_accepts_inflection_but_rejects_invented_topic(self):
        class Primary:
            phrase = 'задача про отчёт'

            def complete_json(self, system_prompt, user_prompt, json_schema):
                return {'operation': 'set_priority', 'target_phrase': self.phrase,
                        'reference_phrase': None, 'value_phrase': 'важной',
                        'priority': 'high', 'lead_minutes': None}

            def close(self):
                pass

        primary = Primary()
        extractor = ExternalCommandExtractor(FallbackModel(primary), semantic_routing=True)
        details = extractor.extract_special('Сделай задачу про отчёт важной', None)
        self.assertEqual(details.target_phrase, 'задача про отчёт')
        primary.phrase = 'задача про бюджет'
        with self.assertRaises(InvalidModelResponseError):
            extractor.extract_special('Сделай задачу про отчёт важной', None)

    def test_pending_answer_resumes_model_split_queue(self):
        whole = 'Создай напоминание погулять завтра и задачу купить хлеб завтра'

        class SequenceExtractor:
            def __init__(self):
                self.commands = [
                    AssistantCommand(action=AssistantIntent.create_entry,
                                     kind=EntryKind.reminder, title='погулять',
                                     date=TODAY + datetime.timedelta(days=1)),
                    AssistantCommand(action=AssistantIntent.create_entry,
                                     time='18:00', time_phrase='в 18:00'),
                    AssistantCommand(action=AssistantIntent.create_entry,
                                     kind=EntryKind.task, title='купить хлеб',
                                     date=TODAY + datetime.timedelta(days=1)),
                ]

            def classify_turn(self, text, timezone, today, *, pending=None, history=None):
                return (TurnKind.batch if text == whole else
                        TurnKind.pending_reply if text == 'в 18:00' else TurnKind.command)

            def split_requests(self, text):
                return ['Создай напоминание погулять завтра',
                        'создай задачу купить хлеб завтра']

            def classify_pending_reply(self, text, **kwargs):
                return PendingReply(act=ReplyAct.continue_command)

            def extract(self, **kwargs):
                return self.commands.pop(0)

        repository = RecurringRepository()
        service = AssistantService(SequenceExtractor(), EntryService(repository), Conversations())
        first = service.handle_message(whole, TZ, None)
        self.assertEqual(first.status, AssistantStatus.needs_clarification)
        second = service.handle_message('в 18:00', TZ, first.conversation_id)
        self.assertEqual(second.status, AssistantStatus.completed)
        self.assertEqual({entry.title.casefold() for entry in repository.items},
                         {'погулять', 'купить хлеб'})


if __name__ == '__main__':
    unittest.main()
