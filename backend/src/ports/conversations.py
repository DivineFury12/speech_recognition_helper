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


class ConversationRepository(typing.Protocol):
    def get(self, conversation_id: uuid.UUID) -> ConversationState | None: ...

    def save_turn(
        self, turn: AssistantTurn, pending_command: AssistantCommand | None,
    ) -> None: ...

    def list_messages(self, conversation_id: uuid.UUID) -> list[ConversationMessage] | None: ...
