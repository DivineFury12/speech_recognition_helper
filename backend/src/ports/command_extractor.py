import datetime
import typing

from src.schemas.assistant_command import AssistantCommand
from src.schemas.assistant import AssistantIntent
from src.schemas.dialogue import PendingReply, PendingStep


class CommandExtractor(typing.Protocol):
    """Turns user text into a validated command using an external model."""

    def extract(
        self,
        text: str,
        timezone: str,
        today: datetime.date,
        previous: AssistantCommand | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AssistantCommand:
        ...

    def classify_pending_reply(
        self, text: str, *, step: PendingStep, action: AssistantIntent,
        question: str | None, candidates: list[dict] | None = None,
        pending: AssistantCommand | None = None,
    ) -> PendingReply:
        ...
