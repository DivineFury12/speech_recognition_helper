"""Persistent conversation state and ordered messages."""

import datetime
import uuid

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from src.models.entry import Base


class ConversationRow(Base):
    __tablename__ = 'conversations'

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default='local', server_default='local')
    pending_command: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    undo_record: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_entry_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now(),
    )


class ConversationMessageRow(Base):
    __tablename__ = 'conversation_messages'

    sequence: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), unique=True, nullable=False)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


class UserSettingsRow(Base):
    __tablename__ = 'user_settings'

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    feed_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default='0')
