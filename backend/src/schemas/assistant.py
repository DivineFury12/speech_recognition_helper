import datetime
import enum
import uuid

import pydantic


class EntryKind(str, enum.Enum):
    task = 'task'
    reminder = 'reminder'
    meeting = 'meeting'


class EntryStatus(str, enum.Enum):
    pending = 'pending'
    completed = 'completed'
    cancelled = 'cancelled'


class AssistantStatus(str, enum.Enum):
    completed = 'completed'
    needs_clarification = 'needs_clarification'
    no_action = 'no_action'


class AssistantIntent(str, enum.Enum):
    create_entry = 'create_entry'
    list_entries = 'list_entries'
    search_entries = 'search_entries'
    unknown = 'unknown'


class Entry(pydantic.BaseModel):
    id: uuid.UUID
    kind: EntryKind
    title: str
    date: datetime.date | None
    time: str | None
    timezone: str
    duration_minutes: int | None = None
    status: EntryStatus | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime | None = None


class TextMessageRequest(pydantic.BaseModel):
    text: str = pydantic.Field(min_length=1)
    timezone: str = pydantic.Field(min_length=1)
    conversation_id: uuid.UUID | None = None

    @pydantic.field_validator('text', 'timezone')
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Поле не может быть пустым')
        return value


class AssistantTurn(pydantic.BaseModel):
    conversation_id: uuid.UUID
    turn_id: uuid.UUID
    user_text: str
    assistant_text: str
    status: AssistantStatus
    intent: AssistantIntent
    entries: list[Entry]
