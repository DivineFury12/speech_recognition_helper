import datetime
import enum
import typing
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


class Recurrence(str, enum.Enum):
    daily = 'daily'
    weekdays = 'weekdays'
    weekly = 'weekly'
    monthly = 'monthly'


class AssistantStatus(str, enum.Enum):
    completed = 'completed'
    needs_clarification = 'needs_clarification'
    no_action = 'no_action'


class AssistantIntent(str, enum.Enum):
    create_entry = 'create_entry'
    list_entries = 'list_entries'
    search_entries = 'search_entries'
    complete_entry = 'complete_entry'
    update_entry = 'update_entry'
    delete_entry = 'delete_entry'
    unknown = 'unknown'


class Entry(pydantic.BaseModel):
    id: uuid.UUID
    kind: EntryKind
    title: str
    date: datetime.date | None
    time: str | None
    timezone: str
    duration_minutes: int | None = None
    recurrence: Recurrence | None = None
    recurrence_until: datetime.date | None = None
    exdates: list[datetime.date] | None = None
    remind_minutes: int | None = None
    status: EntryStatus | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime | None = None


class EntryList(pydantic.BaseModel):
    items: list[Entry]


class EntryUpdate(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    title: str | None = pydantic.Field(default=None, min_length=1)
    date: datetime.date | None = None
    time: str | None = pydantic.Field(default=None, pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$')
    timezone: str | None = pydantic.Field(default=None, min_length=1)
    duration_minutes: int | None = pydantic.Field(default=None, ge=1, le=1440)
    recurrence: Recurrence | None = None
    recurrence_until: datetime.date | None = None
    exdates: list[datetime.date] | None = None
    remind_minutes: int | None = pydantic.Field(default=None, ge=0, le=10080)
    status: EntryStatus | None = None

    @pydantic.model_validator(mode='after')
    def validate_patch(self) -> 'EntryUpdate':
        if not self.model_fields_set:
            raise ValueError('Укажите хотя бы одно поле для изменения')
        if 'title' in self.model_fields_set and (self.title is None or not self.title.strip()):
            raise ValueError('Название не может быть пустым')
        if 'timezone' in self.model_fields_set and (self.timezone is None or not self.timezone.strip()):
            raise ValueError('Часовой пояс не может быть пустым')
        if 'status' in self.model_fields_set and self.status is None:
            raise ValueError('Статус не может быть пустым')
        return self


class EntryCreate(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='forbid')

    kind: EntryKind
    title: str = pydantic.Field(min_length=1, max_length=500)
    date: datetime.date | None = None
    time: str | None = pydantic.Field(default=None, pattern=r'^([01][0-9]|2[0-3]):[0-5][0-9]$')
    timezone: str = pydantic.Field(min_length=1, max_length=64)
    duration_minutes: int | None = pydantic.Field(default=None, ge=1, le=1440)
    recurrence: Recurrence | None = None
    remind_minutes: int | None = pydantic.Field(default=None, ge=0, le=10080)

    @pydantic.model_validator(mode='after')
    def validate_create(self) -> 'EntryCreate':
        if not self.title.strip():
            raise ValueError('Название не может быть пустым')
        if self.kind in (EntryKind.meeting, EntryKind.reminder) and (self.date is None or self.time is None):
            raise ValueError('Для встречи и напоминания нужны дата и время')
        if self.time is not None and self.date is None:
            raise ValueError('Время требует даты')
        return self


class TextMessageRequest(pydantic.BaseModel):
    text: str = pydantic.Field(min_length=1, max_length=1000)
    timezone: str = pydantic.Field(min_length=1)
    conversation_id: uuid.UUID | None = None
    request_id: uuid.UUID | None = None

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
    request_id: uuid.UUID | None = None
    user_text: str
    assistant_text: str
    status: AssistantStatus
    intent: AssistantIntent
    entries: list[Entry]
    suggestions: list[str] = pydantic.Field(
        default_factory=list, description='Короткие ответы для кнопок под сообщением',
    )


class ConversationMessage(pydantic.BaseModel):
    id: uuid.UUID
    role: typing.Literal['user', 'assistant']
    text: str
    created_at: datetime.datetime


class ConversationHistory(pydantic.BaseModel):
    conversation_id: uuid.UUID
    messages: list[ConversationMessage]
