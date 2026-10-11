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
    date_phrase: str | None = None
    time: str | None = pydantic.Field(
        default=None,
        pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$',
    )
    time_phrase: str | None = None
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
    query: str | None = None

    @pydantic.field_validator('title', 'query', 'date_phrase', 'time_phrase')
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
        if self.action == AssistantIntent.create_entry:
            if self.date_from is not None or self.date_to is not None or self.query is not None:
                raise ValueError('create_entry contains unrelated fields')
        elif self.action == AssistantIntent.list_entries:
            if any(value is not None for value in (
                self.title, self.date, self.time, self.date_phrase, self.time_phrase, self.query,
            )):
                raise ValueError('list_entries contains unrelated fields')
        elif self.action == AssistantIntent.search_entries:
            if any(value is not None for value in (
                self.title, self.date, self.time, self.date_phrase, self.time_phrase,
                self.date_from, self.date_to,
            )):
                raise ValueError('search_entries contains unrelated fields')
        return self

    @classmethod
    def provider_schema(cls) -> dict:
        """Require explicit nulls so the model cannot silently omit fields."""
        schema = cls.model_json_schema()
        schema['required'] = list(schema['properties'])
        return schema
