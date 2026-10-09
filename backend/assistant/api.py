import datetime as dt
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

from .service import Assistant, InvalidTimezone, resolve_timezone

TIME_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
EntryKind = Literal["task", "reminder", "meeting"]
EntryStatus = Literal["pending", "completed", "cancelled"]


class Entry(BaseModel):
    id: str
    kind: EntryKind
    title: str
    date: str | None
    time: str | None
    timezone: str
    duration_minutes: int | None = None
    status: EntryStatus | None = None
    created_at: str
    updated_at: str | None = None


class EntryList(BaseModel):
    items: list[Entry]


class AssistantTurn(BaseModel):
    conversation_id: str
    turn_id: str
    user_text: str
    assistant_text: str
    status: Literal["completed", "needs_clarification", "no_action"]
    intent: Literal["create_entry", "list_entries", "search_entries", "unknown"]
    entries: list[Entry]


class TextMessageRequest(BaseModel):
    text: str = Field(min_length=1)
    timezone: str
    conversation_id: uuid.UUID | None = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text не может быть пустым")
        return v.strip()


def _check_tz(v: str | None) -> str | None:
    if v is not None:
        try:
            resolve_timezone(v)
        except InvalidTimezone as e:
            raise ValueError(str(e)) from e
    return v


class EntryCreate(BaseModel):
    kind: EntryKind
    title: str = Field(min_length=1)
    date: dt.date | None = None
    time: str | None = Field(default=None, pattern=TIME_PATTERN)
    timezone: str
    duration_minutes: int | None = Field(default=None, ge=1)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, v: str) -> str:
        return _check_tz(v)

    @model_validator(mode="after")
    def date_time_required(self):
        if self.kind in ("meeting", "reminder") and (self.date is None or self.time is None):
            raise ValueError("для встречи и напоминания обязательны date и time")
        return self


class EntryUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    date: dt.date | None = None
    time: str | None = Field(default=None, pattern=TIME_PATTERN)
    timezone: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1)
    status: EntryStatus | None = None

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, v: str | None) -> str | None:
        return _check_tz(v)


class ConversationMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    text: str
    created_at: str


class ConversationMessages(BaseModel):
    conversation_id: str
    messages: list[ConversationMessage]


def _assistant(request: Request) -> Assistant:
    return request.app.state.assistant


def _uuid_or_422(value: str | None, name: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError as e:
        raise HTTPException(422, f"{name} должен быть UUID") from e


def _tz_or_422(value: str) -> str:
    try:
        resolve_timezone(value)
    except InvalidTimezone as e:
        raise HTTPException(422, str(e)) from e
    return value


router = APIRouter(prefix="/v1")


@router.post("/assistant/messages", response_model=AssistantTurn, tags=["Assistant"])
def send_text_message(body: TextMessageRequest, request: Request):
    tz = _tz_or_422(body.timezone)
    cid = str(body.conversation_id) if body.conversation_id else None
    return _assistant(request).handle(body.text, tz, cid)


@router.post("/assistant/audio", response_model=AssistantTurn, tags=["Assistant"])
def send_audio_message(
    request: Request,
    file: UploadFile = File(...),
    timezone: str = Form(...),
    conversation_id: str | None = Form(default=None),
):
    tz = _tz_or_422(timezone)
    cid = _uuid_or_422(conversation_id, "conversation_id")
    text = request.app.state.transcribe_upload(file)
    assistant = _assistant(request)
    if not text.strip():
        cid = assistant.storage.ensure_conversation(cid)
        return {
            "conversation_id": cid,
            "turn_id": str(uuid.uuid4()),
            "user_text": "",
            "assistant_text": "Не расслышал, повторите, пожалуйста.",
            "status": "no_action",
            "intent": "unknown",
            "entries": [],
        }
    return assistant.handle(text.strip(), tz, cid)


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessages, tags=["Conversations"])
def list_conversation_messages(conversation_id: uuid.UUID, request: Request):
    storage = _assistant(request).storage
    cid = str(conversation_id)
    if not storage.conversation_exists(cid):
        raise HTTPException(404, "Диалог не найден")
    return {"conversation_id": cid, "messages": storage.list_messages(cid)}


@router.get("/entries", response_model=EntryList, tags=["Entries"])
def list_entries(
    request: Request,
    date_from: Annotated[dt.date, Query()],
    date_to: Annotated[dt.date, Query()],
    kind: EntryKind | None = None,
    status: EntryStatus | None = None,
):
    if date_from > date_to:
        raise HTTPException(422, "date_from позже date_to")
    items = _assistant(request).storage.list_entries(date_from.isoformat(), date_to.isoformat(), kind, status)
    return {"items": items}


@router.post("/entries", response_model=Entry, status_code=201, tags=["Entries"])
def create_entry(body: EntryCreate, request: Request):
    return _assistant(request).storage.create_entry(
        kind=body.kind,
        title=body.title.strip(),
        date=body.date.isoformat() if body.date else None,
        time=body.time,
        timezone=body.timezone,
        duration_minutes=body.duration_minutes if body.kind == "meeting" else None,
    )


@router.get("/entries/search", response_model=EntryList, tags=["Entries"])
def search_entries(request: Request, q: Annotated[str, Query(min_length=1)], kind: EntryKind | None = None):
    from .search import search_entries as rank

    storage = _assistant(request).storage
    return {"items": rank(q, storage.all_entries(kind))}


@router.get("/entries/{entry_id}", response_model=Entry, tags=["Entries"])
def get_entry(entry_id: uuid.UUID, request: Request):
    entry = _assistant(request).storage.get_entry(str(entry_id))
    if entry is None:
        raise HTTPException(404, "Запись не найдена")
    return entry


@router.patch("/entries/{entry_id}", response_model=Entry, tags=["Entries"])
def update_entry(entry_id: uuid.UUID, body: EntryUpdate, request: Request):
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(422, "Нужно передать хотя бы одно поле")
    if "date" in changes and changes["date"] is not None:
        changes["date"] = changes["date"].isoformat()
    entry = _assistant(request).storage.update_entry(str(entry_id), changes)
    if entry is None:
        raise HTTPException(404, "Запись не найдена")
    return entry


@router.delete("/entries/{entry_id}", status_code=204, tags=["Entries"])
def delete_entry(entry_id: uuid.UUID, request: Request):
    if not _assistant(request).storage.delete_entry(str(entry_id)):
        raise HTTPException(404, "Запись не найдена")
    return Response(status_code=204)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)


@router.post("/speech/synthesize", tags=["Speech synthesis"])
def synthesize_speech(body: SynthesizeRequest):
    raise HTTPException(501, "Озвучивание пока не реализовано")
