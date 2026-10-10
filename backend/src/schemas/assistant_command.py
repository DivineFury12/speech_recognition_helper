"""Structured result expected from the external language model."""

import datetime

import pydantic

from src.schemas.assistant import AssistantIntent, EntryKind


class AssistantCommand(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    action: AssistantIntent
    kind: EntryKind | None = None
    title: str | None = None
    date: datetime.date | None = None
    time: str | None = pydantic.Field(
        default=None,
        pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$',
    )
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
    query: str | None = None

    @pydantic.field_validator('title', 'query')
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @pydantic.model_validator(mode='after')
    def validate_period(self) -> 'AssistantCommand':
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError('date_from must be on or before date_to')
        return self
