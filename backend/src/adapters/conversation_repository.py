"""PostgreSQL conversation history and pending-command persistence."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.db_transaction import database_session
from src.models.conversation import ConversationMessageRow, ConversationRow
from src.ports.conversations import ConversationState
from src.schemas.assistant import AssistantIntent, AssistantStatus, AssistantTurn, ConversationMessage
from src.schemas.assistant_command import AssistantCommand


def _message(row: ConversationMessageRow) -> ConversationMessage:
    return ConversationMessage.model_validate(row, from_attributes=True)


class SqlAlchemyConversationRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def get(self, conversation_id: uuid.UUID) -> ConversationState | None:
        with database_session(self._sessions) as session:
            row = session.get(ConversationRow, conversation_id)
            if row is None:
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
                row = ConversationRow(id=turn.conversation_id)
                session.add(row)
                session.flush()
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
            if session.get(ConversationRow, conversation_id) is None:
                return None
            rows = session.scalars(
                select(ConversationMessageRow)
                .where(ConversationMessageRow.conversation_id == conversation_id)
                .order_by(ConversationMessageRow.sequence)
            )
            return [_message(row) for row in rows]
