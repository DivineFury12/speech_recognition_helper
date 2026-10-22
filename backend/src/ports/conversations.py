"""Storage contract for assistant conversations."""

import dataclasses
import typing
import uuid

from src.schemas.assistant import AssistantTurn, ConversationMessage
from src.schemas.assistant_command import AssistantCommand


@dataclasses.dataclass(frozen=True)
class ConversationState:
    id: uuid.UUID
    pending_command: AssistantCommand | None
    messages: list[ConversationMessage]
    last_entry_id: uuid.UUID | None = None


class ConversationRepository(typing.Protocol):
    def get(self, conversation_id: uuid.UUID) -> ConversationState | None: ...

    def save_turn(
        self, turn: AssistantTurn, pending_command: AssistantCommand | None,
    ) -> None: ...

    def list_messages(self, conversation_id: uuid.UUID) -> list[ConversationMessage] | None: ...

    def is_foreign(self, conversation_id: uuid.UUID) -> bool: ...

    def get_undo(self, conversation_id: uuid.UUID) -> dict | None: ...

    def set_undo(self, conversation_id: uuid.UUID, record: dict | None) -> None: ...

    def created_entries(self, conversation_id: uuid.UUID) -> list[uuid.UUID]: ...

    def add_created(self, conversation_id: uuid.UUID, entry_id: uuid.UUID) -> None: ...
