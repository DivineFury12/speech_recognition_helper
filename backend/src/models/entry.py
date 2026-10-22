"""Persistent task, reminder, and meeting."""

import datetime
import uuid

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import JSON, Date, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EntryRow(Base):
    __tablename__ = 'entries'
    __table_args__ = (
        Index('ix_entries_date', 'date'),
        Index('ix_entries_user_date', 'user_id', 'date'),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default='local', server_default='local')
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    time: Mapped[str | None] = mapped_column(String(5), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recurrence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    recurrence_until: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    recurrence_interval: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exdates: Mapped[list | None] = mapped_column(JSON, nullable=True)
    remind_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[str | None] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default='pending')
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class EntryEmbeddingRow(Base):
    __tablename__ = 'entry_embeddings'
    __table_args__ = (Index('ix_entry_embeddings_model_name', 'model_name'),)

    entry_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey('entries.id', ondelete='CASCADE'), primary_key=True,
    )
    model_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(), nullable=False)
