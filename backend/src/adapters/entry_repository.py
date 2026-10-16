"""PostgreSQL entry persistence and nearest-neighbor search."""

import datetime
import hashlib
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from src.db_transaction import database_session
from src.models.entry import EntryEmbeddingRow, EntryRow
from src.schemas.assistant import Entry, EntryKind, EntryStatus, Recurrence
from src.services.recurrence import occurrences
from src.user_context import current_user


def _to_entry(row: EntryRow) -> Entry:
    return Entry.model_validate(row, from_attributes=True)


class SqlAlchemyEntryRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    @staticmethod
    def _owned(row: EntryRow | None) -> EntryRow | None:
        return row if row is not None and row.user_id == current_user() else None

    def get(self, entry_id: uuid.UUID) -> Entry | None:
        with database_session(self._sessions) as session:
            row = self._owned(session.get(EntryRow, entry_id))
            return _to_entry(row) if row is not None else None

    def all_entries(self) -> list[Entry]:
        statement = select(EntryRow).where(EntryRow.user_id == current_user()).order_by(
            EntryRow.created_at.desc())
        with database_session(self._sessions) as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def list_undated(self) -> list[Entry]:
        statement = select(EntryRow).where(
            EntryRow.user_id == current_user(), EntryRow.date.is_(None),
        ).order_by(EntryRow.status == EntryStatus.completed.value, EntryRow.created_at.desc())
        with database_session(self._sessions) as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def restore(self, entry: Entry) -> Entry:
        """Put a deleted entry back with the same id (undo of a deletion)."""
        with database_session(self._sessions, write=True) as session:
            row = EntryRow(
                id=entry.id, user_id=current_user(), kind=entry.kind.value, title=entry.title,
                date=entry.date, time=entry.time, timezone=entry.timezone,
                duration_minutes=entry.duration_minutes,
                recurrence=entry.recurrence.value if entry.recurrence else None,
                recurrence_until=entry.recurrence_until, remind_minutes=entry.remind_minutes,
                priority=getattr(entry.priority, 'value', entry.priority),
                exdates=[d.isoformat() for d in entry.exdates] if entry.exdates else None,
                status=(entry.status or EntryStatus.pending).value, created_at=entry.created_at,
                updated_at=datetime.datetime.now(datetime.timezone.utc),
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return _to_entry(row)

    def update(
        self, entry_id: uuid.UUID, changes: dict, *, embedding: list[float] | None,
        embedding_model: str | None,
    ) -> Entry | None:
        with database_session(self._sessions, write=True) as session:
            row = self._owned(session.get(EntryRow, entry_id, with_for_update=True))
            if row is None:
                return None
            proposed_title = changes.get('title', row.title)
            proposed_date = changes.get('date', row.date)
            proposed_time = changes.get('time', row.time)
            proposed_timezone = changes.get('timezone', row.timezone)
            proposed_status = changes.get('status', row.status)
            if isinstance(proposed_status, EntryStatus):
                proposed_status = proposed_status.value
            if proposed_status == EntryStatus.pending.value:
                normalized_title = ' '.join(proposed_title.casefold().split())
                signature = '\x1f'.join((
                    row.kind, normalized_title,
                    proposed_date.isoformat() if proposed_date else '',
                    proposed_time or '', proposed_timezone,
                ))
                lock_key = int.from_bytes(
                    hashlib.sha256(signature.encode('utf-8')).digest()[:8],
                    'big', signed=True,
                )
                session.execute(select(func.pg_advisory_xact_lock(lock_key)))
                duplicate = session.scalar(select(EntryRow.id).where(
                    EntryRow.id != entry_id,
                    EntryRow.user_id == row.user_id,
                    EntryRow.kind == row.kind,
                    func.lower(EntryRow.title) == normalized_title,
                    EntryRow.date == proposed_date,
                    EntryRow.time == proposed_time,
                    EntryRow.timezone == proposed_timezone,
                    EntryRow.status == EntryStatus.pending.value,
                ).limit(1))
                if duplicate is not None:
                    raise ValueError('Такая запись уже есть')
            title_changed = 'title' in changes and changes['title'] != row.title
            for name, value in changes.items():
                if name == 'exdates' and value is not None:
                    value = sorted({d.isoformat() if isinstance(d, datetime.date) else d for d in value})
                setattr(row, name, getattr(value, 'value', value))
            if title_changed:
                session.query(EntryEmbeddingRow).filter(
                    EntryEmbeddingRow.entry_id == entry_id,
                ).delete(synchronize_session=False)
                if embedding is not None and embedding_model is not None:
                    session.add(EntryEmbeddingRow(
                        entry_id=entry_id, model_name=embedding_model, embedding=embedding,
                    ))
            row.updated_at = datetime.datetime.now(datetime.timezone.utc)
            session.flush()
            session.refresh(row)
            return _to_entry(row)

    def delete(self, entry_id: uuid.UUID) -> bool:
        with database_session(self._sessions, write=True) as session:
            row = self._owned(session.get(EntryRow, entry_id, with_for_update=True))
            if row is None:
                return False
            session.delete(row)
            return True

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
        duration_minutes: int | None = None,
        recurrence: Recurrence | None = None,
        remind_minutes: int | None = None,
        recurrence_until: datetime.date | None = None,
        priority: str | None = None,
    ) -> Entry:
        with database_session(self._sessions, write=True) as session:
            return self._insert(
                session, kind=kind, title=title, date=date, time=time,
                timezone=timezone, embedding=embedding, embedding_model=embedding_model,
                duration_minutes=duration_minutes, recurrence=recurrence,
                remind_minutes=remind_minutes, recurrence_until=recurrence_until, priority=priority,
            )

    @staticmethod
    def _insert(
        session: Session, *, kind: EntryKind, title: str,
        date: datetime.date | None, time: str | None, timezone: str,
        embedding: list[float] | None, embedding_model: str | None,
        duration_minutes: int | None = None, recurrence: Recurrence | None = None,
        remind_minutes: int | None = None, recurrence_until: datetime.date | None = None,
        priority: str | None = None,
    ) -> Entry:
        row = EntryRow(
            id=uuid.uuid4(), user_id=current_user(), kind=kind.value, title=title, date=date,
            time=time, timezone=timezone, duration_minutes=duration_minutes,
            recurrence=recurrence.value if recurrence else None,
            remind_minutes=remind_minutes, recurrence_until=recurrence_until,
            priority=getattr(priority, 'value', priority),
            status=EntryStatus.pending.value,
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

    def create_once(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str, embedding: list[float] | None,
        embedding_model: str | None, duration_minutes: int | None = None,
        recurrence: Recurrence | None = None, remind_minutes: int | None = None,
        recurrence_until: datetime.date | None = None, priority: str | None = None,
    ) -> tuple[Entry, bool]:
        normalized_title = ' '.join(title.casefold().split())
        user_id = current_user()
        signature = '\x1f'.join((
            user_id, kind.value, normalized_title, date.isoformat() if date else '',
            time or '', timezone,
        ))
        lock_key = int.from_bytes(
            hashlib.sha256(signature.encode('utf-8')).digest()[:8], 'big', signed=True,
        )
        with database_session(self._sessions, write=True) as session:
            session.execute(select(func.pg_advisory_xact_lock(lock_key)))
            existing = session.scalar(
                select(EntryRow).where(
                    EntryRow.user_id == user_id,
                    EntryRow.kind == kind.value,
                    func.lower(EntryRow.title) == normalized_title,
                    EntryRow.date == date,
                    EntryRow.time == time,
                    EntryRow.timezone == timezone,
                    EntryRow.status == EntryStatus.pending.value,
                ).order_by(EntryRow.created_at.desc()).limit(1)
            )
            if existing is not None:
                return _to_entry(existing), False
            return self._insert(
                session, kind=kind, title=title, date=date, time=time,
                timezone=timezone, embedding=embedding, embedding_model=embedding_model,
                duration_minutes=duration_minutes, recurrence=recurrence,
                remind_minutes=remind_minutes, recurrence_until=recurrence_until, priority=priority,
            ), True

    def list_period(
        self,
        date_from: datetime.date,
        date_to: datetime.date,
        *,
        kind: EntryKind | None = None,
        status: EntryStatus | None = None,
    ) -> list[Entry]:
        statement = select(EntryRow).where(
            EntryRow.user_id == current_user(),
            EntryRow.date <= date_to,
            (EntryRow.date >= date_from) | EntryRow.recurrence.is_not(None),
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        if status is not None:
            statement = statement.where(EntryRow.status == status.value)
        with database_session(self._sessions) as session:
            rows = [_to_entry(row) for row in session.scalars(statement)]
        result = []
        for entry in rows:
            last = min(date_to, entry.recurrence_until) if entry.recurrence_until else date_to
            skipped = set(entry.exdates or [])
            for day in occurrences(entry.date, entry.recurrence, date_from, last):
                if day in skipped:
                    continue
                result.append(entry if day == entry.date else entry.model_copy(update={'date': day}))
        result.sort(key=lambda e: (e.date, e.time is None, e.time or '', e.created_at))
        return result

    def search_text(self, query: str, *, kind: EntryKind | None = None, limit: int = 5) -> list[Entry]:
        statement = select(EntryRow).where(
            EntryRow.user_id == current_user(),
            func.strpos(func.lower(EntryRow.title), query.lower()) > 0,
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        statement = statement.order_by(EntryRow.created_at.desc()).limit(limit)
        with database_session(self._sessions) as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def search_similar(
        self, embedding: list[float], *, embedding_model: str,
        kind: EntryKind | None = None, limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[Entry]:
        distance = EntryEmbeddingRow.embedding.cosine_distance(embedding)
        statement = select(EntryRow).join(EntryEmbeddingRow).where(
            EntryRow.user_id == current_user(),
            EntryEmbeddingRow.model_name == embedding_model,
            distance <= 1 - min_similarity,
        )
        if kind is not None:
            statement = statement.where(EntryRow.kind == kind.value)
        statement = statement.order_by(distance).limit(limit)
        with database_session(self._sessions) as session:
            return [_to_entry(row) for row in session.scalars(statement)]

    def list_unindexed(self, embedding_model: str, *, limit: int = 100) -> list[tuple[uuid.UUID, str]]:
        indexed = select(EntryEmbeddingRow.entry_id).where(
            EntryEmbeddingRow.entry_id == EntryRow.id,
            EntryEmbeddingRow.model_name == embedding_model,
        ).exists()
        statement = select(EntryRow.id, EntryRow.title).where(~indexed).order_by(
            EntryRow.created_at, EntryRow.id,
        ).limit(limit)
        with database_session(self._sessions) as session:
            return list(session.execute(statement).all())

    def save_embedding(self, entry_id: uuid.UUID, embedding_model: str, embedding: list[float]) -> None:
        with database_session(self._sessions, write=True) as session:
            row = session.get(EntryEmbeddingRow, (entry_id, embedding_model))
            if row is None:
                session.add(EntryEmbeddingRow(
                    entry_id=entry_id, model_name=embedding_model, embedding=embedding,
                ))
            else:
                row.embedding = embedding
