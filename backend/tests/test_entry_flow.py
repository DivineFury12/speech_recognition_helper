import datetime
import unittest
import uuid

from src.schemas.assistant import AssistantIntent, AssistantStatus, Entry, EntryKind, EntryStatus
from src.schemas.assistant_command import AssistantCommand
from src.services.assistant import AssistantService
from src.services.entries import EntryService


class FakeRepository:
    def __init__(self):
        self.items = []
        self.last_embedding = None
        self.embeddings = {}

    def create(self, *, kind, title, date, time, timezone, embedding, embedding_model):
        self.last_embedding = embedding
        entry = Entry(
            id=uuid.uuid4(), kind=kind, title=title, date=date, time=time,
            timezone=timezone, status=EntryStatus.pending,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        self.items.append(entry)
        if embedding is not None:
            self.embeddings[(entry.id, embedding_model)] = embedding
        return entry

    def list_period(self, date_from, date_to, *, kind=None, status=None):
        return [entry for entry in self.items if entry.date is not None
                and date_from <= entry.date <= date_to
                and (kind is None or entry.kind == kind)
                and (status is None or entry.status == status)]

    def search_text(self, query, *, kind=None, limit=5):
        return [entry for entry in self.items if query.lower() in entry.title.lower()
                and (kind is None or entry.kind == kind)][:limit]

    def search_similar(self, embedding, *, embedding_model, kind=None, limit=5):
        return [entry for entry in self.items if (entry.id, embedding_model) in self.embeddings
                and (kind is None or entry.kind == kind)][:limit]

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


class FakeExtractor:
    def __init__(self, command):
        self.command = command

    def extract(self, text, timezone, today):
        return self.command


class EntryFlowTests(unittest.TestCase):
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
