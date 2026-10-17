"""Structured result expected from the external language model."""

import datetime
import enum
import uuid

import pydantic

from src.schemas.assistant import AssistantIntent, EntryKind, Recurrence
from src.schemas.dialogue import PendingStep


class UpdateMode(str, enum.Enum):
    move = 'move'
    edit = 'edit'


class AssistantCommand(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    action: AssistantIntent
    action_phrase: str | None = None
    target_reference_phrase: str | None = None
    source_reference_phrase: str | None = None
    update_mode: UpdateMode | None = None
    kind: EntryKind | None = None
    title: str | None = None
    date: datetime.date | None = None
    date_phrase: str | None = None
    time: str | None = pydantic.Field(
        default=None,
        pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$',
    )
    time_phrase: str | None = None
    duration_minutes: int | None = pydantic.Field(default=None, ge=1, le=1440)
    duration_phrase: str | None = None
    date_from: datetime.date | None = None
    date_to: datetime.date | None = None
    query: str | None = None
    target_query: str | None = None
    target_date: datetime.date | None = None
    target_date_phrase: str | None = None
    target_time: str | None = pydantic.Field(default=None, pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$')
    target_time_phrase: str | None = None
    new_title: str | None = None
    # Server-owned selection state. Never include IDs in the provider schema.
    candidate_ids: list[uuid.UUID] | None = None
    selected_entry_id: uuid.UUID | None = None
    move_intent: bool | None = None
    pending_step: PendingStep | None = None
    signals_verified: bool = False
    # Server-owned: recurrence comes from the user's words, confirmations from guards.
    recurrence: Recurrence | None = None
    confirm: str | None = None
    confirm_entry_id: uuid.UUID | None = None
    forced: list[str] | None = None
    shift_minutes: int | None = None
    remind_minutes: int | None = None
    recurrence_until: datetime.date | None = None
    priority: str | None = None

    @pydantic.field_validator('title', 'query', 'date_phrase', 'time_phrase',
                             'duration_phrase', 'target_query', 'target_date_phrase',
                             'target_time_phrase', 'new_title', 'action_phrase',
                             'target_reference_phrase', 'source_reference_phrase')
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
                self.title, self.date, self.time, self.date_phrase, self.time_phrase,
            )):
                raise ValueError('list_entries contains unrelated fields')
        elif self.action == AssistantIntent.search_entries:
            if any(value is not None for value in (
                self.title, self.date, self.time, self.date_phrase, self.time_phrase,
            )):
                raise ValueError('search_entries contains unrelated fields')
        elif self.action in (AssistantIntent.complete_entry, AssistantIntent.update_entry,
                             AssistantIntent.delete_entry):
            if self.date_from is not None or self.date_to is not None or self.query is not None:
                raise ValueError('mutation contains unrelated fields')
        return self

    @classmethod
    def provider_schema(cls) -> dict:
        """Require explicit nulls so the model cannot silently omit fields."""
        schema = cls.model_json_schema()
        for name in ('candidate_ids', 'selected_entry_id', 'move_intent',
                     'pending_step', 'signals_verified', 'recurrence', 'confirm',
                     'confirm_entry_id', 'forced', 'shift_minutes', 'remind_minutes',
                     'recurrence_until', 'priority'):
            schema['properties'].pop(name)
        schema['required'] = list(schema['properties'])
        return schema
