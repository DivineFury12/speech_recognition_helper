import datetime
import typing

from src.schemas.assistant_command import AssistantCommand


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
