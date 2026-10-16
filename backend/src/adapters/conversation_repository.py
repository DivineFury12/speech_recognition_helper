"""PostgreSQL conversation history and pending-command persistence."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.db_transaction import database_session
from src.models.conversation import ConversationMessageRow, ConversationRow, UserSettingsRow
from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, ConversationMessage
from src.schemas.assistant_command import AssistantCommand
from src.user_context import current_user


def _message(row: ConversationMessageRow) -> ConversationMessage:
    return ConversationMessage.model_validate(row, from_attributes=True)


class SqlAlchemyConversationRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def is_foreign(self, conversation_id: uuid.UUID) -> bool:
        with database_session(self._sessions) as session:
            row = session.get(ConversationRow, conversation_id)
            return row is not None and row.user_id != current_user()

    def get_undo(self, conversation_id: uuid.UUID) -> dict | None:
        with database_session(self._sessions) as session:
            row = session.get(ConversationRow, conversation_id)
            return row.undo_record if row is not None and row.user_id == current_user() else None

    def set_undo(self, conversation_id: uuid.UUID, record: dict | None) -> None:
        with database_session(self._sessions, write=True) as session:
            row = session.get(ConversationRow, conversation_id, with_for_update=True)
            if row is None:
                row = ConversationRow(id=conversation_id, user_id=current_user())
                session.add(row)
            if row.user_id == current_user():
                row.undo_record = record

    def get(self, conversation_id: uuid.UUID) -> ConversationState | None:
        with database_session(self._sessions) as session:
            row = session.get(ConversationRow, conversation_id)
            if row is None or row.user_id != current_user():
                return None
            recent = list(session.scalars(
                select(ConversationMessageRow)
                .where(ConversationMessageRow.conversation_id == conversation_id)
                .order_by(ConversationMessageRow.sequence.desc())
                .limit(12)
            ))
            return ConversationState(
                id=row.id,
                pending_command=AssistantCommand.model_validate(row.pending_command)
                if row.pending_command else None,
                messages=[_message(item) for item in reversed(recent)],
                last_entry_id=row.last_entry_id,
            )

    def save_turn(
        self, turn: AssistantTurn, pending_command: AssistantCommand | None,
    ) -> None:
        with database_session(self._sessions, write=True) as session:
            row = session.get(ConversationRow, turn.conversation_id, with_for_update=True)
            if row is None:
                row = ConversationRow(id=turn.conversation_id, user_id=current_user())
                session.add(row)
                session.flush()
            elif row.user_id != current_user():
                raise PermissionError('Диалог принадлежит другому пользователю')
            row.pending_command = (
                pending_command.model_dump(mode='json') if pending_command else None
            )
            if turn.status == AssistantStatus.completed:
                if turn.intent == AssistantIntent.delete_entry:
                    row.last_entry_id = None
                elif len(turn.entries) == 1:
                    row.last_entry_id = turn.entries[0].id
                elif turn.intent in (AssistantIntent.list_entries, AssistantIntent.search_entries):
                    row.last_entry_id = None
            session.add_all([
                ConversationMessageRow(
                    id=uuid.uuid4(), conversation_id=turn.conversation_id,
                    role='user', text=turn.user_text,
                ),
                ConversationMessageRow(
                    id=uuid.uuid4(), conversation_id=turn.conversation_id,
                    role='assistant', text=turn.assistant_text,
                ),
            ])

    def list_messages(self, conversation_id: uuid.UUID) -> list[ConversationMessage] | None:
        with database_session(self._sessions) as session:
            row = session.get(ConversationRow, conversation_id)
            if row is None or row.user_id != current_user():
                return None
            rows = session.scalars(
                select(ConversationMessageRow)
                .where(ConversationMessageRow.conversation_id == conversation_id)
                .order_by(ConversationMessageRow.sequence)
            )
            return [_message(row) for row in rows]


class SqlAlchemyUserSettingsRepository:
    """Per-user settings: the version of the personal calendar link."""

    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def feed_version(self, user_id: str) -> int:
        with database_session(self._sessions) as session:
            row = session.get(UserSettingsRow, user_id)
            return row.feed_version if row is not None else 0

    def bump_feed_version(self, user_id: str) -> int:
        with database_session(self._sessions, write=True) as session:
            row = session.get(UserSettingsRow, user_id, with_for_update=True)
            if row is None:
                row = UserSettingsRow(user_id=user_id, feed_version=0)
                session.add(row)
            row.feed_version += 1
            return row.feed_version
