import datetime as dt
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .calendar_ics import build_calendar
from .service import Assistant, InvalidTimezone, resolve_timezone

TIME_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
EntryKind = Literal["task", "reminder", "meeting"]
EntryStatus = Literal["pending", "completed", "cancelled"]
Recurrence = Literal["daily", "weekdays", "weekly", "monthly"]
Intent = Literal[
    "create_entry", "list_entries", "search_entries",
    "update_entry", "delete_entry", "complete_entry", "unknown",
]


class Entry(BaseModel):
    id: str
    kind: EntryKind
    title: str
    date: str | None
    time: str | None
    timezone: str
    duration_minutes: int | None = None
    recurrence: Recurrence | None = None
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
    intent: Intent
    entries: list[Entry]


class TextMessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
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
    title: str = Field(min_length=1, max_length=500)
    date: dt.date | None = None
    time: str | None = Field(default=None, pattern=TIME_PATTERN)
    timezone: str
    duration_minutes: int | None = Field(default=None, ge=1)
    recurrence: Recurrence | None = None

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, v: str) -> str:
        return _check_tz(v)

    @model_validator(mode="after")
    def date_time_required(self):
        if self.kind in ("meeting", "reminder") and (self.date is None or self.time is None):
            raise ValueError("для встречи и напоминания обязательны date и time")
        if self.recurrence and self.date is None:
            raise ValueError("для повторяющейся записи нужна date")
        return self


class EntryUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    date: dt.date | None = None
    time: str | None = Field(default=None, pattern=TIME_PATTERN)
    timezone: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1)
    recurrence: Recurrence | None = None
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


class SessionInfo(BaseModel):
    user_id: str
    access_code_required: bool
    unlocked: bool
    llm_provider: str
    asr_enabled: bool
    voice_enabled: bool


class UnlockRequest(BaseModel):
    code: str = Field(min_length=1, max_length=200)


class CalendarLink(BaseModel):
    url: str


def _assistant(request: Request) -> Assistant:
    return request.app.state.assistant


def _user(request: Request) -> str:
    return request.state.session.user_id


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


def _rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "assistant_limiter", None)
    if limiter and not limiter.allow(_user(request)):
        raise HTTPException(429, "Слишком много запросов, подождите минуту")


router = APIRouter(prefix="/v1")


@router.get("/session", response_model=SessionInfo, tags=["Session"])
def get_session(request: Request):
    sessions = request.app.state.sessions
    session = request.state.session
    return {
        "user_id": session.user_id,
        "access_code_required": bool(sessions.access_code),
        "unlocked": session.unlocked or not sessions.access_code,
        "llm_provider": request.app.state.llm_name,
        "asr_enabled": request.app.state.asr_ready(),
        "voice_enabled": request.app.state.voice_enabled,
    }


@router.post("/session/unlock", status_code=204, tags=["Session"])
def unlock_session(body: UnlockRequest, request: Request):
    sessions = request.app.state.sessions
    if not sessions.access_code:
        return Response(status_code=204)
    client = request.client.host if request.client else "unknown"
    if not request.app.state.unlock_limiter.allow(client):
        raise HTTPException(429, "Слишком много попыток, попробуйте позже")
    if not sessions.check_code(body.code):
        raise HTTPException(401, "Неверный код доступа")
    request.state.session.unlocked = True
    request.state.cookie_dirty = True
    return Response(status_code=204)


@router.post("/assistant/messages", response_model=AssistantTurn, response_model_exclude_unset=True, tags=["Assistant"])
def send_text_message(body: TextMessageRequest, request: Request):
    _rate_limit(request)
    tz = _tz_or_422(body.timezone)
    cid = str(body.conversation_id) if body.conversation_id else None
    return _assistant(request).handle(body.text, tz, cid, user_id=_user(request))


@router.post("/assistant/audio", response_model=AssistantTurn, response_model_exclude_unset=True, tags=["Assistant"])
def send_audio_message(
    request: Request,
    file: UploadFile = File(...),
    timezone: str = Form(...),
    conversation_id: str | None = Form(default=None),
):
    _rate_limit(request)
    tz = _tz_or_422(timezone)
    cid = _uuid_or_422(conversation_id, "conversation_id")
    text = request.app.state.transcribe_upload(file)
    assistant = _assistant(request)
    if not text.strip():
        cid = assistant.storage.ensure_conversation(cid, _user(request))
        return {
            "conversation_id": cid,
            "turn_id": str(uuid.uuid4()),
            "user_text": "",
            "assistant_text": "Не расслышал, повторите, пожалуйста.",
            "status": "no_action",
            "intent": "unknown",
            "entries": [],
        }
    return assistant.handle(text.strip(), tz, cid, user_id=_user(request))


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessages, tags=["Conversations"])
def list_conversation_messages(conversation_id: uuid.UUID, request: Request):
    storage = _assistant(request).storage
    cid = str(conversation_id)
    if not storage.conversation_exists(cid, _user(request)):
        raise HTTPException(404, "Диалог не найден")
    return {"conversation_id": cid, "messages": storage.list_messages(cid)}


@router.get("/entries", response_model=EntryList, response_model_exclude_unset=True, tags=["Entries"])
def list_entries(
    request: Request,
    date_from: Annotated[dt.date, Query()],
    date_to: Annotated[dt.date, Query()],
    kind: EntryKind | None = None,
    status: EntryStatus | None = None,
):
    if date_from > date_to:
        raise HTTPException(422, "date_from позже date_to")
    if (date_to - date_from).days > 366:
        raise HTTPException(422, "Период не может быть больше года")
    items = _assistant(request).storage.list_entries(
        date_from.isoformat(), date_to.isoformat(), kind, status, user_id=_user(request)
    )
    return {"items": items}


@router.get("/entries/undated", response_model=EntryList, response_model_exclude_unset=True, tags=["Entries"])
def list_undated_entries(request: Request):
    return {"items": _assistant(request).storage.undated_entries(_user(request))}


@router.post("/entries", response_model=Entry, status_code=201, response_model_exclude_unset=True, tags=["Entries"])
def create_entry(body: EntryCreate, request: Request):
    return _assistant(request).storage.create_entry(
        kind=body.kind,
        title=body.title.strip(),
        date=body.date.isoformat() if body.date else None,
        time=body.time,
        timezone=body.timezone,
        duration_minutes=(body.duration_minutes or 60) if body.kind == "meeting" else None,
        recurrence=body.recurrence,
        user_id=_user(request),
    )


@router.get("/entries/search", response_model=EntryList, response_model_exclude_unset=True, tags=["Entries"])
def search_entries(request: Request, q: Annotated[str, Query(min_length=1, max_length=500)],
                   kind: EntryKind | None = None):
    from .search import search_entries as rank

    assistant = _assistant(request)
    user_id = _user(request)
    items = rank(q, assistant.storage.all_entries(kind, user_id), llm=assistant.llm,
                 index=assistant.index, user_id=user_id)
    return {"items": items}


@router.get("/entries/{entry_id}", response_model=Entry, response_model_exclude_unset=True, tags=["Entries"])
def get_entry(entry_id: uuid.UUID, request: Request):
    entry = _assistant(request).storage.get_entry(str(entry_id), _user(request))
    if entry is None:
        raise HTTPException(404, "Запись не найдена")
    return entry


@router.patch("/entries/{entry_id}", response_model=Entry, response_model_exclude_unset=True, tags=["Entries"])
def update_entry(entry_id: uuid.UUID, body: EntryUpdate, request: Request):
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(422, "Нужно передать хотя бы одно поле")
    if changes.get("date") is not None:
        changes["date"] = changes["date"].isoformat()
    storage = _assistant(request).storage
    current = storage.get_entry(str(entry_id), _user(request))
    if current is None:
        raise HTTPException(404, "Запись не найдена")
    merged = {**current, **changes}
    if merged["kind"] in ("meeting", "reminder") and (not merged.get("date") or not merged.get("time")):
        raise HTTPException(422, "для встречи и напоминания обязательны date и time")
    if merged.get("recurrence") and not merged.get("date"):
        raise HTTPException(422, "для повторяющейся записи нужна date")
    return storage.update_entry(str(entry_id), changes, _user(request))


@router.delete("/entries/{entry_id}", status_code=204, tags=["Entries"])
def delete_entry(entry_id: uuid.UUID, request: Request):
    if not _assistant(request).storage.delete_entry(str(entry_id), _user(request)):
        raise HTTPException(404, "Запись не найдена")
    return Response(status_code=204)


@router.get("/calendar", response_model=CalendarLink, tags=["Calendar"])
def calendar_link(request: Request):
    token = request.app.state.sessions.feed_token(_user(request))
    return {"url": str(request.base_url).rstrip("/") + f"/v1/calendar/feed/{token}.ics"}


@router.get("/calendar/feed/{token}.ics", tags=["Calendar"], response_class=PlainTextResponse)
def calendar_feed(token: str, request: Request):
    user_id = request.app.state.sessions.feed_user(token)
    if user_id is None:
        raise HTTPException(404, "Календарь не найден")
    body = build_calendar(_assistant(request).storage.all_entries(None, user_id))
    return Response(content=body, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition": 'inline; filename="assistant.ics"'})


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = Field(default="alina", min_length=1, max_length=40)


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: str


class VoiceList(BaseModel):
    available: bool
    loading: bool
    voices: list[VoiceInfo]


@router.get("/speech/voices", response_model=VoiceList, tags=["Speech synthesis"])
def list_voices(request: Request):
    engine = request.app.state.tts
    if engine is None:
        return {"available": False, "loading": False, "voices": []}
    voices = [{"id": v.id, "name": v.name, "description": v.description} for v in engine.voices()]
    return {"available": engine.ready, "loading": not engine.ready and engine.error is None, "voices": voices}


@router.post("/speech/synthesize", tags=["Speech synthesis"],
             responses={200: {"content": {"audio/wav": {}}, "description": "Ответ, озвученный выбранным голосом"}})
def synthesize_speech(body: SynthesizeRequest, request: Request):
    from .tts import TTSUnavailableError

    engine = request.app.state.tts
    if engine is None:
        raise HTTPException(503, "Озвучка на сервере отключена")
    limiter = getattr(request.app.state, "tts_limiter", None)
    if limiter and not limiter.allow(_user(request)):
        raise HTTPException(429, "Слишком много запросов на озвучку, подождите минуту")
    try:
        audio = engine.synthesize(body.text, body.voice)
    except KeyError:
        raise HTTPException(404, f"Нет голоса «{body.voice}»")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except TTSUnavailableError as e:
        raise HTTPException(503, str(e))
    return Response(content=audio, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})
