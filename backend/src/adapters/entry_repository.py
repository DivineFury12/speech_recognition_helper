"""PostgreSQL entry persistence and nearest-neighbor search."""

import datetime
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from src.models.entry import EntryEmbeddingRow, EntryRow
from src.schemas.assistant import Entry, EntryKind, EntryStatus


def _to_entry(row: EntryRow) -> Entry:
    return Entry.model_validate(row, from_attributes=True)


class SqlAlchemyEntryRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create(
        self,
        *,
        kind: EntryKind,
        title: str,
        date: datetime.date | None,
        time: str | None,
        timezone: str,
        embedding: list[float] | None,
        embedding_model: str | None,
    ) -> Entry:
        with self._sessions.begin() as session:
            row = EntryRow(
                id=uuid.uuid4(), kind=kind.value, title=title, date=date,
                time=time, timezone=timezone, status=EntryStatus.pending.value,
            )
            session.add(row)
            session.flush()
            if embedding is not None:
                if embedding_model is None:
                    raise ValueError('embedding_model is required for a vector')
                session.add(EntryEmbeddingRow(
                    entry_id=row.id, model_name=embedding_model, embedding=embedding,
                ))
            session.refresh(row)
            return _to_entry(row)

    def list_period(
        self,
        date_from: datetime.date,
        date_to: datetime.date,
        *,
        kind: EntryKind | None = None,
        status: EntryStatus | None = None,
    ) -> list[Entry]:
        statement = select(EntryRow).where(
            EntryRow.date >= date_from,
            EntryRow.date <= date_to,
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        if status is not None:
            statement = statement.where(EntryRow.status == status.value)
        statement = statement.order_by(EntryRow.date, EntryRow.time, EntryRow.created_at)
        with self._sessions() as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def search_text(self, query: str, *, kind: EntryKind | None = None, limit: int = 5) -> list[Entry]:
        statement = select(EntryRow).where(
            func.strpos(func.lower(EntryRow.title), query.lower()) > 0
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        statement = statement.order_by(EntryRow.created_at.desc()).limit(limit)
        with self._sessions() as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def search_similar(
        self, embedding: list[float], *, embedding_model: str,
        kind: EntryKind | None = None, limit: int = 5
    ) -> list[Entry]:
        distance = EntryEmbeddingRow.embedding.cosine_distance(embedding)
        statement = select(EntryRow).join(EntryEmbeddingRow).where(
            EntryEmbeddingRow.model_name == embedding_model,
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        statement = statement.order_by(distance).limit(limit)
        with self._sessions() as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def list_unindexed(self, embedding_model: str, *, limit: int = 100) -> list[tuple[uuid.UUID, str]]:
        indexed = select(EntryEmbeddingRow.entry_id).where(
            EntryEmbeddingRow.entry_id == EntryRow.id,
            EntryEmbeddingRow.model_name == embedding_model,
        ).exists()
        statement = select(EntryRow.id, EntryRow.title).where(~indexed).order_by(
            EntryRow.created_at, EntryRow.id,
        ).limit(limit)
        with self._sessions() as session:
            return list(session.execute(statement).all())

    def save_embedding(self, entry_id: uuid.UUID, embedding_model: str, embedding: list[float]) -> None:
        with self._sessions.begin() as session:
            row = session.get(EntryEmbeddingRow, (entry_id, embedding_model))
            if row is None:
                session.add(EntryEmbeddingRow(
                    entry_id=entry_id, model_name=embedding_model, embedding=embedding,
                ))
            else:
                row.embedding = embedding
