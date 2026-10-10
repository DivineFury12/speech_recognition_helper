import datetime
import unittest
import uuid

from sqlalchemy import delete

from src.adapters.entry_repository import SqlAlchemyEntryRepository
from src.db import create_session_factory
from src.db_config import get_database_url
from src.models.entry import EntryRow
from src.schemas.assistant import EntryKind


class EntryRepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database_url = get_database_url()
        if database_url is None:
            raise unittest.SkipTest('Database not configured')
        cls.sessions, cls.engine = create_session_factory(database_url)
        cls.repository = SqlAlchemyEntryRepository(cls.sessions)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_two_vector_dimensions_remain_separate(self):
        title = f'Integration {uuid.uuid4()}'
        entry = self.repository.create(
            kind=EntryKind.task, title=title, date=datetime.date(2026, 10, 10),
            time=None, timezone='Europe/Moscow',
            embedding=[0.1] * 384, embedding_model='intfloat/multilingual-e5-small',
        )
        try:
            missing = self.repository.list_unindexed('ai-forever/FRIDA')
            self.assertIn((entry.id, title), missing)
            self.repository.save_embedding(entry.id, 'ai-forever/FRIDA', [0.1] * 1536)
            self.assertNotIn((entry.id, title), self.repository.list_unindexed('ai-forever/FRIDA'))
            for model_name, dimensions in (
                ('intfloat/multilingual-e5-small', 384),
                ('ai-forever/FRIDA', 1536),
            ):
                with self.subTest(model_name=model_name):
                    found = self.repository.search_similar(
                        [0.1] * dimensions, embedding_model=model_name, limit=100,
                    )
                    self.assertIn(entry.id, [item.id for item in found])
        finally:
            with self.sessions.begin() as session:
                session.execute(delete(EntryRow).where(EntryRow.id == entry.id))


if __name__ == '__main__':
    unittest.main()
