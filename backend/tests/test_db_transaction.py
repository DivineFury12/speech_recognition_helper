"""A failed batch rolls back its own writes while the request stays usable."""

import datetime
import unittest
import uuid
from unittest.mock import patch

from sqlalchemy import Column, Integer, MetaData, Table, create_engine, func, select
from sqlalchemy.orm import Session

from src.db_transaction import request_savepoint, use_session
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn
from src.services.assistant import AssistantService


class RequestSavepointTests(unittest.TestCase):
    def test_rollback_discards_batch_and_allows_result_to_be_saved(self):
        engine = create_engine('sqlite+pysqlite:///:memory:')
        table = Table('items', MetaData(), Column('id', Integer, primary_key=True))
        table.create(engine)
        with Session(engine) as session, session.begin():
            with use_session(session):
                with request_savepoint() as savepoint:
                    session.execute(table.insert().values(id=1))
                    savepoint.rollback()
                session.execute(table.insert().values(id=2))
        with Session(engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(table)), 1)
            self.assertEqual(session.scalar(select(table.c.id)), 2)

    def test_assistant_batch_failure_rolls_back_earlier_action(self):
        engine = create_engine('sqlite+pysqlite:///:memory:')
        table = Table('items', MetaData(), Column('id', Integer, primary_key=True))
        table.create(engine)
        service = AssistantService(None)
        conversation_id = uuid.uuid4()

        def handle_part(text, timezone, previous_id, today):
            if text == 'first':
                session.execute(table.insert().values(id=1))
                status = AssistantStatus.completed
            else:
                status = AssistantStatus.no_action
            return AssistantTurn(
                conversation_id=conversation_id, turn_id=uuid.uuid4(),
                user_text=text, assistant_text=text, status=status,
                intent=AssistantIntent.create_entry, entries=[],
            )

        with Session(engine) as session, session.begin():
            with use_session(session), patch.object(service, '_handle_at', handle_part):
                result = service._run_parts(
                    ['first', 'second'], 'Europe/Moscow', conversation_id,
                    datetime.date(2026, 10, 29),
                )
                self.assertEqual(result.status, AssistantStatus.no_action)
                self.assertIn('отменены', result.assistant_text)
        with Session(engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(table)), 0)


if __name__ == '__main__':
    unittest.main()
