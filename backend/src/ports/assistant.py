import typing
import uuid

import src.schemas.assistant


class AssistantService(typing.Protocol):
    """The future business service; both HTTP routes call this method."""

    def handle_message(
        self,
        text: str,
        timezone: str,
        conversation_id: uuid.UUID | None,
    ) -> src.schemas.assistant.AssistantTurn:
        ...
