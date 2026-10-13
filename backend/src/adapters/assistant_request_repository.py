"""Serialize a request ID and commit its effects together with its response."""

import hashlib
import uuid
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from src.db_transaction import use_session
from src.models.assistant_request import AssistantRequestRow
from src.schemas.assistant import AssistantTurn


class RequestIdConflictError(ValueError):
    pass


class SqlAlchemyAssistantRequestRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def execute_once(
        self, request_id: uuid.UUID, fingerprint: str,
        conversation_id: uuid.UUID | None, execute: Callable[[], AssistantTurn],
    ) -> AssistantTurn:
        # PostgreSQL transaction locks also serialize concurrent retries from
        # different API workers. The receipt, entry and dialogue commit together.
        lock_key = int.from_bytes(
            hashlib.sha256(request_id.bytes).digest()[:8], 'big', signed=True,
        )
        with self._sessions.begin() as session:
            session.execute(select(func.pg_advisory_xact_lock(lock_key)))
            existing = session.get(AssistantRequestRow, request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise RequestIdConflictError('request_id уже использован для другого запроса')
                return AssistantTurn.model_validate(existing.response)
            if conversation_id is not None:
                conversation_lock = int.from_bytes(
                    hashlib.sha256(conversation_id.bytes).digest()[:8], 'big', signed=True,
                )
                session.execute(select(func.pg_advisory_xact_lock(conversation_lock)))
            with use_session(session):
                turn = execute().model_copy(update={'request_id': request_id})
            session.add(AssistantRequestRow(
                id=request_id, fingerprint=fingerprint,
                response=turn.model_dump(mode='json'),
            ))
            return turn
