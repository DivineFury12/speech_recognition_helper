import datetime
import math
import unittest
import uuid
from zoneinfo import ZoneInfo

from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, Entry, EntryKind, EntryStatus
from src.schemas.assistant_command import AssistantCommand
from src.schemas.dialogue import PendingReply, PendingStep, ReplyAct, ScheduleRole
from src.services.assistant import AssistantService
from src.services.dialogue_interpreter import DialogueInterpreter
from src.services.entries import EntryService


class FakeRepository:
    def __init__(self):
        self.items = []
        self.last_embedding = None
        self.embeddings = {}

    def create(self, *, kind, title, date, time, timezone, embedding, embedding_model,
               duration_minutes=None):
        self.last_embedding = embedding
        entry = Entry(
            id=uuid.uuid4(), kind=kind, title=title, date=date, time=time,
            timezone=timezone, duration_minutes=duration_minutes, status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.items.append(entry)
        if embedding is not None:
            self.embeddings[(entry.id, embedding_model)] = embedding
        return entry

    def get(self, entry_id):
        return next((entry for entry in self.items if entry.id == entry_id), None)

    def update(self, entry_id, changes, *, embedding, embedding_model):
        entry = self.get(entry_id)
        if entry is None:
            return None
        updated = entry.model_copy(update={
            **changes, 'updated_at': datetime.datetime.now(datetime.timezone.utc),
        })
        self.items[self.items.index(entry)] = updated
        if 'title' in changes:
            self.embeddings = {key: value for key, value in self.embeddings.items()
                               if key[0] != entry_id}
            if embedding is not None:
                self.embeddings[(entry_id, embedding_model)] = embedding
        return updated

    def delete(self, entry_id):
        entry = self.get(entry_id)
        if entry is None:
            return False
        self.items.remove(entry)
        return True

    def create_once(self, *, kind, title, date, time, timezone, embedding, embedding_model,
                    duration_minutes=None):
        for entry in self.items:
            if (
                entry.kind == kind and entry.title.casefold() == title.casefold()
                and entry.date == date and entry.time == time
                and entry.timezone == timezone and entry.status == EntryStatus.pending
            ):
                return entry, False
        return self.create(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            embedding=embedding, embedding_model=embedding_model,
            duration_minutes=duration_minutes,
        ), True

    def list_period(self, date_from, date_to, *, kind=None, status=None):
        return [entry for entry in self.items if entry.date is not None
                and date_from <= entry.date <= date_to
                and (kind is None or entry.kind == kind)
                and (status is None or entry.status == status)]

    def search_text(self, query, *, kind=None, limit=5):
        return [entry for entry in self.items if query.lower() in entry.title.lower()
                and (kind is None or entry.kind == kind)][:limit]

    def search_similar(self, embedding, *, embedding_model, kind=None, limit=5,
                       min_similarity=0.0):
        scored = []
        for entry in self.items:
            stored = self.embeddings.get((entry.id, embedding_model))
            if stored is None or (kind is not None and entry.kind != kind):
                continue
            similarity = sum(a * b for a, b in zip(embedding, stored)) / (
                math.sqrt(sum(a * a for a in embedding))
                * math.sqrt(sum(b * b for b in stored))
            )
            if similarity >= min_similarity:
                scored.append((similarity, entry))
        return [entry for _, entry in sorted(scored, key=lambda item: -item[0])[:limit]]

    def list_unindexed(self, embedding_model, *, limit=100):
        return [(entry.id, entry.title) for entry in self.items
                if (entry.id, embedding_model) not in self.embeddings][:limit]

    def save_embedding(self, entry_id, embedding_model, embedding):
        self.embeddings[(entry_id, embedding_model)] = embedding


class FakeEmbedder:
    def __init__(self, name='e5', dimensions=384):
        self.name = name
        self.dimensions = dimensions

    def embed_document(self, text):
        return [0.1] * self.dimensions

    def embed_query(self, text):
        return [0.1] * self.dimensions


class SearchTestEmbedder:
    name = 'intfloat/multilingual-e5-small'
    dimensions = 2

    def embed_document(self, text):
        return [1.0, 0.0] if 'фильм' in text else [0.0, 1.0]

    def embed_query(self, text):
        return [0.854, 0.520] if text == 'кино' else [0.0, 1.0]


class FakeExtractor:
    def __init__(self, command):
        self.command = command

    def extract(self, text, timezone, today, previous=None, history=None):
        if isinstance(self.command, list):
            return self.command.pop(0)
        return self.command


class ReplyExtractor(FakeExtractor):
    def __init__(self, command, replies):
        super().__init__(command)
        self.replies = list(replies)
        self.reply_context = []

    def classify_pending_reply(self, text, *, step, action, question, candidates=None,
                               pending=None):
        self.reply_context.append((text, step, action, question, candidates))
        return self.replies.pop(0)


class FakeConversations:
    def __init__(self):
        self.states = {}

    def get(self, conversation_id):
        return self.states.get(conversation_id)

    def save_turn(self, turn, pending_command):
        previous = self.states.get(turn.conversation_id)
        last_entry_id = previous.last_entry_id if previous else None
        if turn.status == AssistantStatus.completed:
            if turn.intent == AssistantIntent.delete_entry:
                last_entry_id = None
            elif len(turn.entries) == 1:
                last_entry_id = turn.entries[0].id
            elif turn.intent in (AssistantIntent.list_entries, AssistantIntent.search_entries):
                last_entry_id = None
        self.states[turn.conversation_id] = ConversationState(
            id=turn.conversation_id, pending_command=pending_command,
            messages=[], last_entry_id=last_entry_id,
        )


class EntryFlowTests(unittest.TestCase):
    def test_schedule_clarification_cannot_mutate_another_entry(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        entries = EntryService(repository)
        unrelated = entries.create(
            kind=EntryKind.reminder, title='Проверить маяк',
            date=today + datetime.timedelta(days=1), time='16:00',
            timezone='Europe/Moscow',
        )
        extractor = ReplyExtractor([
            AssistantCommand(action=AssistantIntent.create_entry,
                             kind=EntryKind.meeting, title='Встреча с Иваном'),
            AssistantCommand(action=AssistantIntent.update_entry,
                             date=today + datetime.timedelta(days=1), time='16:00',
                             target_date=today + datetime.timedelta(days=1),
                             target_time='16:00'),
        ], [PendingReply(act=ReplyAct.continue_command)])
        service = AssistantService(extractor, entries, FakeConversations())

        question = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        reply = service.handle_message('Завтра в 16:00', 'Europe/Moscow',
                                       question.conversation_id)
        self.assertEqual(reply.status, AssistantStatus.completed)
        self.assertEqual(reply.intent, AssistantIntent.create_entry)
        self.assertEqual(reply.entries[0].kind, EntryKind.meeting)
        self.assertEqual(reply.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(reply.entries[0].time, '16:00')
        self.assertEqual(repository.get(unrelated.id), unrelated)

    def test_unclear_schedule_reply_moves_only_selected_meeting(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        entries = EntryService(repository)
        first = entries.create(
            kind=EntryKind.meeting, title='Встреча с Иваном',
            date=today + datetime.timedelta(days=1), time='11:00',
            timezone='Europe/Moscow',
        )
        second = entries.create(
            kind=EntryKind.meeting, title='Встреча с Иваном',
            date=today + datetime.timedelta(days=2), time='16:00',
            timezone='Europe/Moscow',
        )
        extractor = ReplyExtractor([
            AssistantCommand(action=AssistantIntent.update_entry,
                             kind=EntryKind.meeting, target_query='Встреча с Иваном'),
            AssistantCommand(action=AssistantIntent.update_entry,
                             date=today + datetime.timedelta(days=7)),
        ], [PendingReply(act=ReplyAct.select, choice=2),
            PendingReply(act=ReplyAct.unclear)])
        service = AssistantService(extractor, entries, FakeConversations())

        question = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        chosen = service.handle_message('вторую', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(chosen.status, AssistantStatus.needs_clarification)
        moved = service.handle_message('через неделю в 17:00', 'Europe/Moscow',
                                       question.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, second.id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=7))
        self.assertEqual(moved.entries[0].time, '17:00')
        self.assertEqual(repository.get(first.id).time, '11:00')

    def test_new_task_does_not_inherit_pending_reminder(self):
        repository = FakeRepository()
        conversations = FakeConversations()
        extractor = ReplyExtractor(
            AssistantCommand(action=AssistantIntent.create_entry,
                             kind=EntryKind.reminder, title='Позвонить Ивану'),
            [PendingReply(act=ReplyAct.continue_command)],
        )
        service = AssistantService(extractor, EntryService(repository), conversations)
        question = service.handle_message('Напомни позвонить Ивану', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)

        created = service.handle_message('Мне нужно купить молоко', 'Europe/Moscow',
                                         question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].kind, EntryKind.task)
        self.assertEqual(created.entries[0].title, 'Купить молоко')
        self.assertIsNone(created.entries[0].date)
        self.assertIsNone(conversations.get(question.conversation_id).pending_command)

    def test_unclear_schedule_reply_completes_meeting_creation(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        extractor = ReplyExtractor([
            AssistantCommand(action=AssistantIntent.create_entry,
                             kind=EntryKind.meeting, title='Встреча с Иваном'),
            AssistantCommand(action=AssistantIntent.create_entry),
        ], [PendingReply(act=ReplyAct.unclear)])
        service = AssistantService(extractor, EntryService(FakeRepository()),
                                   FakeConversations())
        question = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        created = service.handle_message('Завтра в 16:00', 'Europe/Moscow',
                                         question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].kind, EntryKind.meeting)
        self.assertEqual(created.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(created.entries[0].time, '16:00')

    def test_time_choice_state_resolves_evening_without_reextracting(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить Ивану',
            'date': (today + datetime.timedelta(days=1)).isoformat(),
            'time_phrase': 'в пять',
        }), [PendingReply(act=ReplyAct.select, choice=2)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Напомни завтра в пять позвонить Ивану',
                                          'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(conversations.get(question.conversation_id).pending_command.pending_step,
                         PendingStep.time_choice)
        done = service.handle_message('вечером', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(done.entries[0].time, '17:00')
        self.assertEqual(done.entries[0].date, today + datetime.timedelta(days=1))

    def test_time_choice_can_update_date_with_period_reply(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить Ивану',
            'date': today.isoformat(), 'time_phrase': 'в пять',
        }), [PendingReply(act=ReplyAct.continue_command)])
        service = AssistantService(extractor, EntryService(FakeRepository()), conversations)
        question = service.handle_message('Напомни сегодня в пять позвонить Ивану',
                                          'Europe/Moscow', None)
        done = service.handle_message('лучше завтра вечером', 'Europe/Moscow',
                                      question.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(done.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(done.entries[0].time, '17:00')

    def test_invalid_model_time_choice_cannot_create_record(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить Ивану',
            'date': today.isoformat(), 'time_phrase': 'в пять',
        }), [PendingReply(act=ReplyAct.select, choice=3)])
        service = AssistantService(extractor, EntryService(repository), conversations)
        question = service.handle_message('Напомни сегодня в пять позвонить Ивану',
                                          'Europe/Moscow', None)
        reply = service.handle_message('третье время', 'Europe/Moscow',
                                       question.conversation_id)
        self.assertEqual(reply.status, AssistantStatus.needs_clarification)
        self.assertEqual(repository.items, [])
        self.assertEqual(conversations.get(reply.conversation_id).pending_command.pending_step,
                         PendingStep.time_choice)

    def test_search_uses_title_words_and_rejects_unrelated_vector_hit(self):
        repository = FakeRepository()
        entries = EntryService(repository, SearchTestEmbedder())
        film = entries.create(
            kind=EntryKind.reminder, title='посмотреть фильм',
            date=datetime.date(2026, 10, 18), time='18:00', timezone='Europe/Moscow',
        )
        tickets = entries.create(
            kind=EntryKind.task, title='Купить билеты',
            date=None, time=None, timezone='Europe/Moscow',
        )
        self.assertEqual(entries.search('просмотр фильма'), [film])
        self.assertEqual(entries.find_candidates('просмотр фильма', kind=EntryKind.reminder),
                         ([film], False))
        self.assertEqual(entries.find_candidates('напоминание о просмотре фильма',
                                                 kind=EntryKind.reminder), ([film], False))
        self.assertEqual(entries.find_candidates('посмотреть кино', kind=EntryKind.reminder),
                         ([film], False))
        self.assertEqual(entries.search('кино'), [film])
        self.assertEqual(entries.search('Гранат', kind=EntryKind.reminder), [])
        self.assertEqual(entries.find_candidates('Гранат', kind=EntryKind.reminder),
                         ([], False))
        self.assertEqual(entries.find_candidates('напоминание', kind=EntryKind.reminder),
                         ([], False))
        self.assertEqual(entries.search('посмотреть фильм'), [film])
        self.assertNotIn(tickets, entries.search('фильм'))

    def test_ambiguous_hour_is_clarified_before_creating_reminder(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'reminder', 'title': 'Позвонить Ивану',
                'date': today.isoformat(), 'time': '05:00', 'time_phrase': 'в пять',
            }),
            AssistantCommand.model_validate({'action': 'unknown'}),
        ]), EntryService(repository), FakeConversations())
        question = service.handle_message('Напомни сегодня в пять позвонить Ивану',
                                          'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertIn('05:00 или 17:00', question.assistant_text)
        self.assertEqual(repository.items, [])
        past = service.handle_message('утром', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(past.status, AssistantStatus.needs_clarification)
        self.assertIn('время уже прошло', past.assistant_text)
        created = service.handle_message('да', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].time, '05:00')

    def test_ambiguous_hour_is_clarified_before_moving_meeting(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        repository = FakeRepository()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': today.isoformat(), 'time': '10:00',
            }),
            AssistantCommand.model_validate({'action': 'update_entry', 'time': '05:00',
                                             'time_phrase': 'на пять'}),
            AssistantCommand.model_validate({'action': 'unknown'}),
        ]), EntryService(repository), FakeConversations())
        created = service.handle_message('Создай встречу с Иваном сегодня в 10:00',
                                         'Europe/Moscow', None)
        question = service.handle_message('Перенеси её на завтра на пять', 'Europe/Moscow',
                                          created.conversation_id)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertIn('05:00 или 17:00', question.assistant_text)
        self.assertEqual(repository.items[0].time, '10:00')
        moved = service.handle_message('вечером', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].time, '17:00')
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))

    def test_inflected_film_query_requires_confirmation_before_delete(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        film = entries.create(kind=EntryKind.reminder, title='посмотреть фильм',
                              date=datetime.date(2026, 10, 18), time='18:00',
                              timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'delete_entry', 'target_query': 'просмотр фильма',
        })), entries, FakeConversations())
        question = service.handle_message('Удали запись на просмотр фильма',
                                          'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(question.entries[0].id, film.id)
        self.assertIsNotNone(repository.get(film.id))
        deleted = service.handle_message('именно', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(deleted.status, AssistantStatus.completed)
        self.assertIsNone(repository.get(film.id))

    def test_vague_film_move_finds_candidate_without_model_target_query(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        film = entries.create(kind=EntryKind.reminder, title='посмотреть фильм',
                              date=datetime.date(2026, 10, 18), time='18:00',
                              timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'update_entry',
        })), entries, FakeConversations())
        question = service.handle_message('Перенеси просмотр фильма',
                                          'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(question.entries[0].id, film.id)
        self.assertIn('Вы имели в виду', question.assistant_text)
        selected = service.handle_message('да, именно', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(selected.status, AssistantStatus.needs_clarification)
        self.assertIn('На какую дату или время перенести', selected.assistant_text)
        self.assertEqual(repository.get(film.id).date, datetime.date(2026, 10, 18))

    def test_future_obligation_creates_task_even_if_model_says_complete(self):
        repository = FakeRepository()
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'target_query': 'проверочный отчёт',
        })), EntryService(repository), FakeConversations())
        turn = service.handle_message('Надо бы закончить проверочный отчёт',
                                      'Europe/Moscow', None)
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual(turn.intent, AssistantIntent.create_entry)
        self.assertEqual(turn.entries[0].title, 'Закончить проверочный отчёт')

    def test_explicit_delete_pronoun_overrides_wrong_update_intent(self):
        repository = FakeRepository()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'Проверочный отчёт',
            }),
            AssistantCommand.model_validate({'action': 'update_entry'}),
        ]), EntryService(repository), FakeConversations())
        created = service.handle_message('Создай задачу Проверочный отчёт', 'Europe/Moscow', None)
        asked = service.handle_message('Удали её', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(asked.status, AssistantStatus.needs_clarification)
        self.assertEqual(asked.suggestions, ['Да', 'Нет'])
        deleted = service.handle_message('жа', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(deleted.status, AssistantStatus.completed)
        self.assertEqual(deleted.intent, AssistantIntent.delete_entry)
        self.assertEqual(repository.items, [])

    def test_meeting_defaults_to_hour_and_shows_end_time(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
            'date': today.isoformat(), 'time': '10:00',
        })), EntryService(FakeRepository()), FakeConversations())
        turn = service.handle_message('Создай встречу с Иваном сегодня в 10:00', 'Europe/Moscow', None)
        self.assertEqual(turn.entries[0].duration_minutes, 60)
        self.assertIn('с 10:00 до 11:00', turn.assistant_text)
        self.assertIn('по умолчанию', turn.assistant_text)

    def test_meeting_move_keeps_or_changes_duration(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': today.isoformat(), 'time': '10:00', 'duration_minutes': 30,
            }),
            AssistantCommand.model_validate({'action': 'update_entry', 'time': '12:00'}),
            AssistantCommand.model_validate({'action': 'update_entry', 'time': '13:00'}),
        ]), EntryService(FakeRepository()), FakeConversations())
        created = service.handle_message('Создай встречу с Иваном сегодня в 10:00 на 30 минут',
                                        'Europe/Moscow', None)
        self.assertEqual(created.entries[0].duration_minutes, 30)
        self.assertIn('с 10:00 до 10:30', created.assistant_text)
        moved = service.handle_message('А перенеси её на 12:00', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.entries[0].duration_minutes, 30)
        self.assertIn('с 12:00 до 12:30', moved.assistant_text)
        shortened = service.handle_message('А перенеси её на 13:00 и сократи до 20 минут',
                                           'Europe/Moscow', created.conversation_id)
        self.assertEqual(shortened.entries[0].duration_minutes, 20)
        self.assertIn('с 13:00 до 13:20', shortened.assistant_text)

    def test_move_with_new_duration_waits_for_schedule(self):
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': today.isoformat(), 'time': '10:00',
            }),
            AssistantCommand.model_validate({'action': 'update_entry'}),
            AssistantCommand.model_validate({'action': 'unknown'}),
        ]), EntryService(FakeRepository()), FakeConversations())
        created = service.handle_message('Создай встречу с Иваном сегодня в 10:00', 'Europe/Moscow', None)
        question = service.handle_message('А перенеси её и сократи до 30 минут',
                                          'Europe/Moscow', created.conversation_id)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(question.entries[0].duration_minutes, 60)
        moved = service.handle_message('на завтра', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].duration_minutes, 30)
        self.assertEqual(moved.entries[0].time, '10:00')

    def test_explicit_move_verb_recovers_unknown_model_intent(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': (today + datetime.timedelta(days=3)).isoformat(), 'time': '16:00',
            }),
            AssistantCommand.model_validate({'action': 'unknown'}),
        ]), entries, conversations)
        created = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('А перенеси её на завтра', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.intent, AssistantIntent.update_entry)
        self.assertEqual(moved.entries[0].id, created.entries[0].id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '16:00')

    def test_explicit_move_verb_recovers_wrong_create_intent(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': (today + datetime.timedelta(days=3)).isoformat(), 'time': '16:00',
            }),
            AssistantCommand.model_validate({'action': 'create_entry', 'kind': 'meeting'}),
        ]), entries, conversations)
        created = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('А перенеси её на завтра', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, created.entries[0].id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))

    def test_pronoun_move_ignores_model_old_date_and_query(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': (today + datetime.timedelta(days=3)).isoformat(), 'time': '17:00',
            }),
            AssistantCommand.model_validate({
                'action': 'update_entry', 'kind': 'meeting',
                'target_query': 'перенеси',
                'target_date': (today + datetime.timedelta(days=1)).isoformat(),
                'target_date_phrase': 'на завтра',
            }),
        ]), entries, conversations)
        created = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('А перенеси её на завтра', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, created.entries[0].id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '17:00')

    def test_move_pronoun_waits_for_new_date_and_keeps_selected_meeting(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        original_date = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date() + datetime.timedelta(days=3)
        extractor = FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': original_date.isoformat(), 'time': '16:00',
            }),
            AssistantCommand.model_validate({'action': 'update_entry', 'kind': 'meeting'}),
            AssistantCommand.model_validate({'action': 'create_entry'}),
        ])
        service = AssistantService(extractor, entries, conversations)
        created = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        question = service.handle_message('А перенеси её', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertIn('На какую дату или время перенести', question.assistant_text)
        self.assertEqual(question.entries[0].id, created.entries[0].id)
        pending = conversations.get(created.conversation_id).pending_command
        self.assertEqual(pending.selected_entry_id, created.entries[0].id)
        self.assertTrue(pending.move_intent)
        self.assertEqual(repository.get(created.entries[0].id).date, original_date)

        moved = service.handle_message('на завтра', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, created.entries[0].id)
        self.assertEqual(moved.entries[0].date,
                         datetime.datetime.now(ZoneInfo('Europe/Moscow')).date() + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '16:00')
        self.assertIn('теперь', moved.assistant_text)
        self.assertIsNone(conversations.get(created.conversation_id).pending_command)

    def test_ambiguous_move_selects_meeting_then_asks_for_new_schedule(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                               date=today + datetime.timedelta(days=2), time='11:00',
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                date=today + datetime.timedelta(days=3), time='16:00',
                                timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
            }),
            AssistantCommand.model_validate({'action': 'unknown'}),
        ]), entries, FakeConversations())
        choice = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        self.assertEqual(choice.status, AssistantStatus.needs_clarification)
        self.assertEqual(len(choice.entries), 2)
        schedule = service.handle_message('вторую', 'Europe/Moscow', choice.conversation_id)
        self.assertEqual(schedule.status, AssistantStatus.needs_clarification)
        self.assertIn('На какую дату или время перенести', schedule.assistant_text)
        self.assertEqual(schedule.entries[0].id, second.id)
        self.assertEqual(repository.get(first.id).date, first.date)
        self.assertEqual(repository.get(second.id).date, second.date)
        moved = service.handle_message('на завтра в 18:00', 'Europe/Moscow', choice.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, second.id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '18:00')
        self.assertEqual(repository.get(first.id).date, first.date)

    def test_candidate_selection_can_include_new_date(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                               date=today + datetime.timedelta(days=2), time='11:00',
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                date=today + datetime.timedelta(days=3), time='16:00',
                                timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
        })), entries, FakeConversations())
        question = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('2 на завтра', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, second.id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '16:00')
        self.assertEqual(repository.get(first.id).date, first.date)

    def test_classified_selection_distinguishes_new_and_old_schedule(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                               date=today + datetime.timedelta(days=1), time='11:00',
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                date=today + datetime.timedelta(days=2), time='16:00',
                                timezone='Europe/Moscow')
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
        }), [PendingReply(act=ReplyAct.select, choice=2,
                         schedule_role=ScheduleRole.replacement)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('пожалуйста перенеси вторую на завтра в 18:00',
                                       'Europe/Moscow', question.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, second.id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))
        self.assertEqual(moved.entries[0].time, '18:00')
        self.assertEqual(repository.get(first.id).time, '11:00')

        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
        }), [PendingReply(act=ReplyAct.select, choice=1,
                         schedule_role=ScheduleRole.existing)])
        service = AssistantService(extractor, entries, FakeConversations())
        question = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        choice = service.handle_message('ту, что завтра', 'Europe/Moscow',
                                        question.conversation_id)
        self.assertEqual(choice.status, AssistantStatus.needs_clarification)
        self.assertEqual(choice.entries[0].id, first.id)
        self.assertEqual(repository.get(first.id).date, first.date)

    def test_bare_new_date_cannot_select_existing_meeting(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                               date=today + datetime.timedelta(days=1), time='11:00',
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                date=today + datetime.timedelta(days=2), time='16:00',
                                timezone='Europe/Moscow')
        extractor = ReplyExtractor([
            AssistantCommand.model_validate({
                'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
            }),
            AssistantCommand.model_validate({
                'action': 'list_entries', 'date_from': (today + datetime.timedelta(days=1)).isoformat(),
                'date_to': (today + datetime.timedelta(days=1)).isoformat(),
            }),
        ], [PendingReply(act=ReplyAct.select, choice=1,
                         schedule_role=ScheduleRole.existing),
             PendingReply(act=ReplyAct.select, choice=2)])
        conversations = FakeConversations()
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Перенеси встречу с Иваном', 'Europe/Moscow', None)
        reply = service.handle_message('на завтра', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(reply.status, AssistantStatus.needs_clarification)
        self.assertEqual(len(reply.entries), 2)
        self.assertIn('запомнил', reply.assistant_text)
        self.assertEqual(conversations.get(reply.conversation_id).pending_command.date,
                         today + datetime.timedelta(days=1))
        self.assertEqual(repository.get(first.id).date, first.date)
        self.assertEqual(repository.get(second.id).date, second.date)
        moved = service.handle_message('вторую', 'Europe/Moscow', reply.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].id, second.id)
        self.assertEqual(moved.entries[0].date, today + datetime.timedelta(days=1))

    def test_new_move_date_does_not_select_old_meeting_by_same_date(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.meeting, title='Проверочная встреча',
                               date=today + datetime.timedelta(days=1), time='11:00',
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Проверочная встреча',
                                date=today + datetime.timedelta(days=2), time='16:00',
                                timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'update_entry', 'kind': 'meeting', 'target_query': 'Проверочная встреча',
        })), entries, FakeConversations())
        question = service.handle_message('Перенеси проверочную встречу', 'Europe/Moscow', None)
        still_ambiguous = service.handle_message('на завтра', 'Europe/Moscow',
                                                 question.conversation_id)
        self.assertEqual(still_ambiguous.status, AssistantStatus.needs_clarification)
        self.assertEqual(len(still_ambiguous.entries), 2)
        self.assertEqual(repository.get(first.id).date, first.date)
        self.assertEqual(repository.get(second.id).date, second.date)

    def test_new_request_interrupts_candidate_question(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                       date=today + datetime.timedelta(days=2), time='11:00',
                       timezone='Europe/Moscow')
        entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                       date=today + datetime.timedelta(days=3), time='16:00',
                       timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'delete_entry', 'target_query': 'Встреча с Иваном',
            }),
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'Купить молоко',
            }),
        ]), entries, FakeConversations())
        question = service.handle_message('Удали встречу с Иваном', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        created = service.handle_message('Создай задачу купить молоко',
                                        'Europe/Moscow', question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].kind, EntryKind.task)

    def test_complete_ambiguous_task_after_spoken_selection(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        first = entries.create(kind=EntryKind.task, title='Купить билеты',
                               date=datetime.date(2026, 10, 15), time=None, timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Купить билеты',
                                date=datetime.date(2026, 10, 16), time=None, timezone='Europe/Moscow')
        conversations = FakeConversations()
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
        })), entries, conversations)
        question = service.handle_message('Отметь задачу купить билеты выполненной', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual([item.id for item in question.entries], [first.id, second.id])
        self.assertEqual(repository.get(first.id).status, EntryStatus.pending)
        done = service.handle_message('вторую', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(repository.get(first.id).status, EntryStatus.pending)
        self.assertEqual(repository.get(second.id).status, EntryStatus.completed)
        self.assertIsNone(conversations.get(done.conversation_id).pending_command)

    def test_classified_choice_completes_only_selected_task(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.task, title='Купить билеты',
                               date=today, time=None, timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Купить билеты',
                                date=today + datetime.timedelta(days=1), time=None,
                                timezone='Europe/Moscow')
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
        }), [PendingReply(act=ReplyAct.select, choice=2)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Отметь билеты выполненными', 'Europe/Moscow', None)
        self.assertEqual(conversations.get(question.conversation_id).pending_command.pending_step,
                         PendingStep.candidate)
        done = service.handle_message('пожалуйста, вторую', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(done.status, AssistantStatus.completed)
        self.assertEqual(repository.get(first.id).status, EntryStatus.pending)
        self.assertEqual(repository.get(second.id).status, EntryStatus.completed)
        self.assertEqual(extractor.reply_context[0][1], PendingStep.candidate)

    def test_classified_new_command_interrupts_pending_selection(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.task, title='Купить билеты',
                               date=today, time=None, timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Купить билеты',
                                date=today + datetime.timedelta(days=1), time=None,
                                timezone='Europe/Moscow')
        conversations = FakeConversations()
        extractor = ReplyExtractor([
            AssistantCommand.model_validate({
                'action': 'delete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
            }),
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'Купить молоко',
            }),
        ], [PendingReply(act=ReplyAct.new_command)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Удали билеты', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        created = service.handle_message('Мне нужно купить молоко', 'Europe/Moscow',
                                         question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].title, 'Купить молоко')
        self.assertIsNotNone(repository.get(first.id))
        self.assertIsNotNone(repository.get(second.id))
        self.assertIsNone(conversations.get(created.conversation_id).pending_command)

    def test_unclear_reply_can_still_be_verified_as_independent_task(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        old = entries.create(kind=EntryKind.task, title='Купить билеты',
                             date=today, time=None, timezone='Europe/Moscow')
        entries.create(kind=EntryKind.task, title='Купить билеты',
                       date=today + datetime.timedelta(days=1), time=None,
                       timezone='Europe/Moscow')
        conversations = FakeConversations()
        extractor = ReplyExtractor([
            AssistantCommand.model_validate({
                'action': 'delete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
            }),
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'купить молоко',
            }),
        ], [PendingReply(act=ReplyAct.unclear)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Удали билеты', 'Europe/Moscow', None)
        created = service.handle_message('Мне нужно купить молоко', 'Europe/Moscow',
                                         question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].title, 'Купить молоко')
        self.assertIsNotNone(repository.get(old.id))

    def test_unclear_reply_with_new_reminder_does_not_reuse_pending_title(self):
        repository = FakeRepository()
        conversations = FakeConversations()
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        extractor = ReplyExtractor([
            AssistantCommand(action=AssistantIntent.create_entry,
                             kind=EntryKind.reminder, title='позвонить Ивану'),
            AssistantCommand(action=AssistantIntent.create_entry,
                             kind=EntryKind.reminder, title='купить лекарства',
                             date=today + datetime.timedelta(days=1), time='18:00'),
        ], [PendingReply(act=ReplyAct.unclear)])
        service = AssistantService(extractor, EntryService(repository), conversations)
        question = service.handle_message('Напомни позвонить Ивану', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        created = service.handle_message('Напомни завтра в 18:00 купить лекарства',
                                         'Europe/Moscow', question.conversation_id)
        self.assertEqual(created.status, AssistantStatus.completed)
        self.assertEqual(created.entries[0].title, 'Купить лекарства')
        self.assertEqual(len(repository.items), 1)
        self.assertIsNone(conversations.get(created.conversation_id).pending_command)

    def test_classified_unclear_reply_keeps_pending_selection(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        today = datetime.datetime.now(ZoneInfo('Europe/Moscow')).date()
        first = entries.create(kind=EntryKind.task, title='Купить билеты',
                               date=today, time=None, timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Купить билеты',
                                date=today + datetime.timedelta(days=1), time=None,
                                timezone='Europe/Moscow')
        conversations = FakeConversations()
        extractor = ReplyExtractor(AssistantCommand.model_validate({
            'action': 'delete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
        }), [PendingReply(act=ReplyAct.unclear)])
        service = AssistantService(extractor, entries, conversations)
        question = service.handle_message('Удали билеты', 'Europe/Moscow', None)
        reply = service.handle_message('может быть', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(reply.status, AssistantStatus.needs_clarification)
        self.assertEqual(len(reply.entries), 2)
        self.assertIsNotNone(repository.get(first.id))
        self.assertIsNotNone(repository.get(second.id))
        self.assertEqual(conversations.get(reply.conversation_id).pending_command.pending_step,
                         PendingStep.candidate)

    def test_update_last_entry_by_pronoun(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        conversations = FakeConversations()
        extractor = FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'meeting', 'title': 'Встреча с Иваном',
                'date': '2026-10-15', 'time': '16:00',
            }),
            AssistantCommand.model_validate({
                'action': 'update_entry', 'kind': 'meeting', 'date': '2026-10-16',
            }),
        ])
        service = AssistantService(extractor, entries, conversations)
        created = service.handle_message('Создай встречу с Иваном', 'Europe/Moscow', None)
        moved = service.handle_message('Перенеси её на 16 октября', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(moved.status, AssistantStatus.completed)
        self.assertEqual(moved.entries[0].date, datetime.date(2026, 10, 16))
        self.assertEqual(moved.entries[0].time, '16:00')

    def test_delete_ambiguous_record_only_after_selection(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        first = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                               date=datetime.date(2026, 10, 15), time='11:00', timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                                date=datetime.date(2026, 10, 15), time='16:00', timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'delete_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
        })), entries, FakeConversations())
        question = service.handle_message('Удали встречу с Иваном', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(len(repository.items), 2)
        deleted = service.handle_message('ту, что в 16:00', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(deleted.status, AssistantStatus.completed)
        self.assertIsNotNone(repository.get(first.id))
        self.assertIsNone(repository.get(second.id))

    def test_negated_candidate_does_not_delete_first_record(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        first = entries.create(kind=EntryKind.task, title='Проверочный отчёт',
                               date=datetime.date(2026, 10, 15), time=None,
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Проверочный отчёт',
                                date=datetime.date(2026, 10, 16), time=None,
                                timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'delete_entry', 'kind': 'task', 'target_query': 'Проверочный отчёт',
        })), entries, FakeConversations())
        question = service.handle_message('Удали проверочный отчёт', 'Europe/Moscow', None)
        rejected = service.handle_message('Не первую', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(rejected.status, AssistantStatus.needs_clarification)
        self.assertIsNotNone(repository.get(first.id))
        self.assertIsNotNone(repository.get(second.id))

    def test_named_target_beats_pronoun_reference_to_last_entry(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        other = entries.create(kind=EntryKind.task, title='Проверочный отчёт Б',
                               date=None, time=None, timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor([
            AssistantCommand.model_validate({
                'action': 'create_entry', 'kind': 'task', 'title': 'Проверочный отчёт А',
            }),
            AssistantCommand.model_validate({
                'action': 'delete_entry', 'kind': 'task', 'target_query': 'Проверочный отчёт Б',
            }),
        ]), entries, FakeConversations())
        created = service.handle_message('Создай задачу Проверочный отчёт А',
                                         'Europe/Moscow', None)
        service.handle_message('Удали эту задачу Проверочный отчёт Б',
                               'Europe/Moscow', created.conversation_id)
        deleted = service.handle_message('да', 'Europe/Moscow', created.conversation_id)
        self.assertEqual(deleted.status, AssistantStatus.completed)
        self.assertIsNotNone(repository.get(created.entries[0].id))
        self.assertIsNone(repository.get(other.id))

    def test_semantic_target_requires_confirmation(self):
        repository = FakeRepository()
        entries = EntryService(repository, FakeEmbedder())
        task = entries.create(kind=EntryKind.task, title='Купить билеты',
                              date=None, time=None, timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
        })), entries, FakeConversations())
        question = service.handle_message('Отметь выполненной задачу про поезд', 'Europe/Moscow', None)
        self.assertEqual(question.status, AssistantStatus.needs_clarification)
        self.assertEqual(repository.get(task.id).status, EntryStatus.pending)
        confirmed = service.handle_message('да', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(confirmed.status, AssistantStatus.completed)
        self.assertEqual(repository.get(task.id).status, EntryStatus.completed)

    def test_common_approval_replies_confirm_one_suggested_record(self):
        for reply in ('именно', 'Да, именно!', 'угу', 'точно', 'всё верно',
                      'это она', 'подтверждаю', 'согласен', 'окей', 'хорошо'):
            with self.subTest(reply=reply):
                repository = FakeRepository()
                entries = EntryService(repository, FakeEmbedder())
                task = entries.create(kind=EntryKind.task, title='Купить билеты',
                                      date=None, time=None, timezone='Europe/Moscow')
                service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
                    'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
                })), entries, FakeConversations())
                question = service.handle_message('Отметь задачу про поезд',
                                                  'Europe/Moscow', None)
                self.assertEqual(question.status, AssistantStatus.needs_clarification)
                answer = service.handle_message(reply, 'Europe/Moscow', question.conversation_id)
                self.assertEqual(answer.status, AssistantStatus.completed)
                self.assertEqual(repository.get(task.id).status, EntryStatus.completed)

    def test_negative_or_ambiguous_reply_does_not_confirm_record(self):
        for reply in ('нет!', 'не та', 'неверно', 'да, но не эту'):
            with self.subTest(reply=reply):
                repository = FakeRepository()
                entries = EntryService(repository, FakeEmbedder())
                task = entries.create(kind=EntryKind.task, title='Купить билеты',
                                      date=None, time=None, timezone='Europe/Moscow')
                service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
                    'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
                })), entries, FakeConversations())
                question = service.handle_message('Отметь задачу про поезд',
                                                  'Europe/Moscow', None)
                answer = service.handle_message(reply, 'Europe/Moscow', question.conversation_id)
                self.assertNotEqual(answer.status, AssistantStatus.completed)
                self.assertEqual(repository.get(task.id).status, EntryStatus.pending)

        repository = FakeRepository()
        entries = EntryService(repository)
        first = entries.create(kind=EntryKind.task, title='Купить билеты',
                               date=datetime.date(2026, 10, 15), time=None,
                               timezone='Europe/Moscow')
        second = entries.create(kind=EntryKind.task, title='Купить билеты',
                                date=datetime.date(2026, 10, 16), time=None,
                                timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'kind': 'task', 'target_query': 'Купить билеты',
        })), entries, FakeConversations())
        question = service.handle_message('Отметь задачу купить билеты', 'Europe/Moscow', None)
        answer = service.handle_message('именно', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(answer.status, AssistantStatus.needs_clarification)
        self.assertEqual(repository.get(first.id).status, EntryStatus.pending)
        self.assertEqual(repository.get(second.id).status, EntryStatus.pending)

    def test_model_classifies_unlisted_reply_only_for_one_candidate(self):
        class ClassifyingExtractor(FakeExtractor):
            def __init__(self, command, decision):
                super().__init__(command)
                self.decision = decision
                self.calls = []

            def classify_confirmation(self, text, suggested_title):
                self.calls.append((text, suggested_title))
                return self.decision

        for decision, expected_status in (
            (True, AssistantStatus.completed),
            (False, AssistantStatus.no_action),
            (None, AssistantStatus.needs_clarification),
        ):
            with self.subTest(decision=decision):
                repository = FakeRepository()
                entries = EntryService(repository, FakeEmbedder())
                task = entries.create(kind=EntryKind.task, title='Купить билеты',
                                      date=None, time=None, timezone='Europe/Moscow')
                extractor = ClassifyingExtractor(AssistantCommand.model_validate({
                    'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
                }), decision)
                service = AssistantService(extractor, entries, FakeConversations())
                question = service.handle_message('Отметь задачу про поезд',
                                                  'Europe/Moscow', None)
                answer = service.handle_message('да это же нужная запись',
                                                'Europe/Moscow', question.conversation_id)
                self.assertEqual(answer.status, expected_status)
                self.assertEqual(extractor.calls, [('да это же нужная запись', 'Купить билеты')])
                self.assertEqual(repository.get(task.id).status,
                                 EntryStatus.completed if decision else EntryStatus.pending)

    def test_model_confirmation_does_not_override_new_instruction(self):
        class ClassifyingExtractor(FakeExtractor):
            def __init__(self, command):
                super().__init__(command)
                self.calls = []

            def classify_confirmation(self, text, suggested_title):
                self.calls.append((text, suggested_title))
                return True

        repository = FakeRepository()
        entries = EntryService(repository, FakeEmbedder())
        task = entries.create(kind=EntryKind.task, title='Купить билеты',
                              date=None, time=None, timezone='Europe/Moscow')
        extractor = ClassifyingExtractor(AssistantCommand.model_validate({
            'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
        }))
        service = AssistantService(extractor, entries, FakeConversations())
        question = service.handle_message('Отметь задачу про поезд', 'Europe/Moscow', None)
        answer = service.handle_message('удали другую запись', 'Europe/Moscow',
                                        question.conversation_id)
        self.assertEqual(answer.status, AssistantStatus.needs_clarification)
        self.assertEqual(extractor.calls, [])
        self.assertEqual(repository.get(task.id).status, EntryStatus.pending)

    def test_model_yes_cannot_override_explicit_negation(self):
        class OverconfidentExtractor(FakeExtractor):
            def classify_confirmation(self, text, suggested_title):
                return True

        repository = FakeRepository()
        entries = EntryService(repository, FakeEmbedder())
        task = entries.create(kind=EntryKind.task, title='Купить билеты',
                              date=None, time=None, timezone='Europe/Moscow')
        service = AssistantService(OverconfidentExtractor(
            AssistantCommand.model_validate({
                'action': 'complete_entry', 'kind': 'task', 'target_query': 'поезд',
            }),
        ), entries, FakeConversations())
        question = service.handle_message('Отметь задачу про поезд', 'Europe/Moscow', None)
        answer = service.handle_message('да, но не эту', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(answer.status, AssistantStatus.needs_clarification)
        self.assertEqual(repository.get(task.id).status, EntryStatus.pending)

    def test_cancel_ambiguous_delete_keeps_all_records(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        for time in ('11:00', '16:00'):
            entries.create(kind=EntryKind.meeting, title='Встреча с Иваном',
                           date=datetime.date(2026, 10, 15), time=time, timezone='Europe/Moscow')
        service = AssistantService(FakeExtractor(AssistantCommand.model_validate({
            'action': 'delete_entry', 'kind': 'meeting', 'target_query': 'Встреча с Иваном',
        })), entries, FakeConversations())
        question = service.handle_message('Удали встречу с Иваном', 'Europe/Moscow', None)
        cancelled = service.handle_message('отмена', 'Europe/Moscow', question.conversation_id)
        self.assertEqual(cancelled.status, AssistantStatus.no_action)
        self.assertEqual(len(repository.items), 2)

    def test_weekday_selection_is_not_mistaken_for_ordinal(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        candidates = [entries.create(
            kind=EntryKind.meeting, title='Планёрка',
            date=datetime.date(2026, 10, 15) + datetime.timedelta(days=offset),
            time='11:00', timezone='Europe/Moscow',
        ) for offset in range(5)]
        selected = DialogueInterpreter._grounded_candidate(
            'В пятницу', candidates, datetime.date(2026, 10, 14),
            allow_schedule_match=True,
        )
        self.assertEqual(selected.id, candidates[1].id)

    def test_create_list_and_search_commands(self):
        repository = FakeRepository()
        entries = EntryService(repository, FakeEmbedder())
        date = datetime.date(2026, 10, 10)
        commands = [
            {'action': 'create_entry', 'kind': 'task', 'title': 'Купить билеты', 'date': '2026-10-10'},
            {'action': 'list_entries', 'date_from': '2026-10-10', 'date_to': '2026-10-10'},
            {'action': 'search_entries', 'query': 'билеты'},
        ]
        for payload, expected_intent in zip(commands, (
            AssistantIntent.create_entry, AssistantIntent.list_entries,
            AssistantIntent.search_entries,
        )):
            turn = AssistantService(entries=entries, extractor=FakeExtractor(
                AssistantCommand.model_validate(payload),
            )).handle_message('Запрос', 'Europe/Moscow', None)
            self.assertEqual(turn.status, AssistantStatus.completed)
            self.assertEqual(turn.intent, expected_intent)
            self.assertEqual(len(turn.entries), 1)
            self.assertEqual(turn.entries[0].title, 'Купить билеты')

        self.assertEqual(len(repository.last_embedding), 384)
        self.assertEqual(entries.list_period(date, date, kind=EntryKind.task), repository.items)

    def test_list_command_filters_requested_kind(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        date = datetime.date(2026, 10, 10)
        for kind, title in ((EntryKind.task, 'Отчёт'), (EntryKind.meeting, 'Встреча')):
            entries.create(kind=kind, title=title, date=date, time=None,
                           timezone='Europe/Moscow')

        command = AssistantCommand.model_validate({
            'action': 'list_entries', 'kind': 'meeting',
            'date_from': '2026-10-10', 'date_to': '2026-10-10',
        })
        turn = AssistantService(entries=entries, extractor=FakeExtractor(command)).handle_message(
            'Какие встречи сегодня?', 'Europe/Moscow', None,
        )
        self.assertEqual(turn.status, AssistantStatus.completed)
        self.assertEqual([entry.title for entry in turn.entries], ['Встреча'])

        search = AssistantCommand.model_validate({
            'action': 'search_entries', 'kind': 'task', 'query': 'Отчёт',
        })
        turn = AssistantService(entries=entries, extractor=FakeExtractor(search)).handle_message(
            'Найди задачу про отчёт', 'Europe/Moscow', None,
        )
        self.assertEqual([entry.title for entry in turn.entries], ['Отчёт'])

    def test_repeated_create_command_returns_existing_entry(self):
        repository = FakeRepository()
        entries = EntryService(repository)
        command = AssistantCommand.model_validate({
            'action': 'create_entry', 'kind': 'task', 'title': 'Разобраться с уроками',
        })
        service = AssistantService(entries=entries, extractor=FakeExtractor(command))
        first = service.handle_message('Создай задачу разобраться с уроками', 'Europe/Moscow', None)
        second = service.handle_message('Создай задачу разобраться с уроками', 'Europe/Moscow', None)
        self.assertEqual(len(repository.items), 1)
        self.assertEqual(first.entries[0].id, second.entries[0].id)
        self.assertIn('уже есть', second.assistant_text)

    def test_text_search_survives_embedding_failure(self):
        class BrokenEmbedder:
            name = 'e5'
            dimensions = 384

            def embed_document(self, text):
                raise ValueError('wrong vector')

            def embed_query(self, text):
                raise ValueError('wrong vector')

        repository = FakeRepository()
        entries = EntryService(repository, BrokenEmbedder())
        entries.create(kind=EntryKind.task, title='Купить билеты', date=None,
                       time=None, timezone='Europe/Moscow')
        self.assertIsNone(repository.last_embedding)
        self.assertEqual(len(entries.search('билеты')), 1)

    def test_reindex_keeps_vectors_from_both_models(self):
        repository = FakeRepository()
        e5 = EntryService(repository, FakeEmbedder('e5', 384))
        entry = e5.create(kind=EntryKind.task, title='Купить билеты', date=None,
                          time=None, timezone='Europe/Moscow')
        frida = EntryService(repository, FakeEmbedder('frida', 1536))
        self.assertEqual(frida.reindex_missing(), 1)
        self.assertEqual(frida.reindex_missing(), 0)
        self.assertEqual(len(repository.embeddings[(entry.id, 'e5')]), 384)
        self.assertEqual(len(repository.embeddings[(entry.id, 'frida')]), 1536)
        self.assertEqual(len(frida.search('поезд')), 1)


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
