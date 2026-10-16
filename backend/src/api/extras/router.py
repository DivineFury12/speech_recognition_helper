"""Session, daily summary, services status, calendar export and speech synthesis."""

import datetime
import typing
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import fastapi
import pydantic

from src.schemas.assistant import Entry, EntryList, EntryStatus
from src.services import phrasing
from src.services.calendar_ics import build_calendar
from src.user_context import current_user


router = fastapi.APIRouter(prefix='/v1')


class SessionInfo(pydantic.BaseModel):
    user_id: str
    access_code_required: bool
    unlocked: bool
    llm_provider: str
    asr_enabled: bool
    voice_enabled: bool


class UnlockRequest(pydantic.BaseModel):
    code: str = pydantic.Field(min_length=1, max_length=200)


class DailySummary(pydantic.BaseModel):
    assistant_text: str
    entries: list[Entry]


class ServiceState(pydantic.BaseModel):
    name: str
    state: typing.Literal['ok', 'loading', 'error', 'off']
    detail: str


class ServicesStatus(pydantic.BaseModel):
    services: list[ServiceState]


class CalendarLink(pydantic.BaseModel):
    url: str


class SynthesizeRequest(pydantic.BaseModel):
    text: str = pydantic.Field(min_length=1, max_length=1000)
    voice: str = pydantic.Field(default='alina', min_length=1, max_length=40)


class VoiceInfo(pydantic.BaseModel):
    id: str
    name: str
    description: str


class VoiceList(pydantic.BaseModel):
    available: bool
    loading: bool
    voices: list[VoiceInfo]


def _entries(request: fastapi.Request):
    service = request.app.state.entries_service
    if service is None:
        raise fastapi.HTTPException(status_code=503, detail='База данных не подключена')
    return service


def _zone(timezone: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise fastapi.HTTPException(status_code=422, detail='Укажите корректный часовой пояс IANA') from exc


@router.get('/session', response_model=SessionInfo, tags=['Session'], operation_id='getSession')
def get_session(request: fastapi.Request) -> SessionInfo:
    sessions = request.app.state.sessions
    session = request.state.session
    state = request.app.state
    return SessionInfo(
        user_id=session.user_id,
        access_code_required=bool(sessions.access_code),
        unlocked=session.unlocked or not sessions.access_code,
        llm_provider='gigachat' if state.llm_health is not None else 'rules',
        asr_enabled=state.voice_enabled and state.transcription_service is not None,
        voice_enabled=state.voice_enabled,
    )


@router.post('/session/unlock', status_code=204, tags=['Session'], operation_id='unlockSession')
def unlock_session(body: UnlockRequest, request: fastapi.Request) -> fastapi.Response:
    sessions = request.app.state.sessions
    if not sessions.access_code:
        return fastapi.Response(status_code=204)
    client = request.client.host if request.client else 'unknown'
    lockout = request.app.state.unlock_lockout
    if lockout.blocked(client) or not request.app.state.unlock_limiter.allow(client):
        raise fastapi.HTTPException(status_code=429, detail='Слишком много неверных попыток, попробуйте через 15 минут')
    if not sessions.check_code(body.code):
        lockout.fail(client)
        raise fastapi.HTTPException(status_code=401, detail='Неверный код доступа')
    lockout.reset(client)
    request.state.session.unlocked = True
    request.state.cookie_dirty = True
    return fastapi.Response(status_code=204)


@router.get('/status', response_model=ServicesStatus, tags=['Session'], operation_id='getServicesStatus')
def services_status(request: fastapi.Request) -> ServicesStatus:
    state = request.app.state
    items = []
    health = getattr(state, 'llm_health', None)
    if health is None:
        items.append(('Понимание команд', 'off', 'ключ GigaChat не задан, работают правила'))
    elif health.healthy:
        items.append(('Понимание команд', 'ok', 'GigaChat отвечает'))
    else:
        items.append(('Понимание команд', 'error', 'GigaChat не отвечает, работают правила — проверьте ключ'))
    items.append(('База данных', 'ok' if state.entries_service is not None else 'error',
                  'PostgreSQL подключён' if state.entries_service is not None else 'PostgreSQL не подключён'))
    if not state.voice_enabled:
        items.append(('Распознавание речи', 'off', 'голос выключен'))
    else:
        ready = state.transcription_service is not None
        items.append(('Распознавание речи', 'ok' if ready else 'error',
                      'GigaAM готов' if ready else 'GigaAM недоступен'))
    tts = state.tts
    if tts is None:
        items.append(('Озвучка', 'off', 'озвучка на сервере выключена'))
    elif tts.ready:
        items.append(('Озвучка', 'ok', 'Silero готов'))
    else:
        items.append(('Озвучка', 'error' if tts.error else 'loading', 'ошибка загрузки' if tts.error else 'загружается'))
    return ServicesStatus(services=[ServiceState(name=n, state=s, detail=d) for n, s, d in items])


@router.get('/assistant/summary', response_model=DailySummary, tags=['Assistant'], operation_id='getDailySummary')
def daily_summary(request: fastapi.Request, timezone: str = fastapi.Query(max_length=64)) -> DailySummary:
    today = datetime.datetime.now(_zone(timezone)).date()
    service = _entries(request)
    entries = [e for e in service.list_period(today, today) if e.status != EntryStatus.completed]
    overdue = [e for e in service.all_entries()
               if e.status == EntryStatus.pending and e.date is not None and e.recurrence is None and e.date < today]
    if entries:
        lines = [phrasing.entry_line(e, today, show_date=False) for e in entries]
        noun = phrasing.plural(len(entries), 'запись', 'записи', 'записей')
        text = f'Сегодня {len(entries)} {noun}:\n' + '\n'.join(lines)
    else:
        text = 'На сегодня ничего не запланировано.'
    if overdue:
        noun = phrasing.plural(len(overdue), 'запись', 'записи', 'записей')
        text += f'\nНе выполнено с прошлых дней: {len(overdue)} {noun}.'
    return DailySummary(assistant_text=text, entries=entries)


@router.get('/entries/undated', response_model=EntryList, tags=['Entries'], operation_id='listUndatedEntries')
def list_undated(request: fastapi.Request) -> EntryList:
    return EntryList(items=_entries(request).list_undated())


def _feed_url(request: fastapi.Request, user_id: str) -> str:
    version = request.app.state.user_settings.feed_version(user_id)
    token = request.app.state.sessions.feed_token(user_id, version)
    return str(request.base_url).rstrip('/') + f'/v1/calendar/feed/{token}.ics'


@router.get('/calendar', response_model=CalendarLink, tags=['Calendar'], operation_id='getCalendarLink')
def calendar_link(request: fastapi.Request) -> CalendarLink:
    return CalendarLink(url=_feed_url(request, current_user()))


@router.post('/calendar/rotate', response_model=CalendarLink, tags=['Calendar'], operation_id='rotateCalendarLink')
def rotate_calendar_link(request: fastapi.Request) -> CalendarLink:
    request.app.state.user_settings.bump_feed_version(current_user())
    return CalendarLink(url=_feed_url(request, current_user()))


@router.get('/calendar/feed/{token}.ics', tags=['Calendar'], operation_id='getCalendarFeed',
            response_class=fastapi.responses.PlainTextResponse)
def calendar_feed(token: str, request: fastapi.Request) -> fastapi.Response:
    found = request.app.state.sessions.feed_user(token)
    if found is None or request.app.state.user_settings.feed_version(found[0]) != found[1]:
        raise fastapi.HTTPException(status_code=404, detail='Календарь не найден')
    from src.user_context import acting_as

    with acting_as(found[0]):
        entries = [e.model_dump(mode='json') for e in _entries(request).all_entries()]
    return fastapi.Response(content=build_calendar(entries), media_type='text/calendar; charset=utf-8',
                            headers={'Content-Disposition': 'inline; filename="assistant.ics"'})


@router.get('/speech/voices', response_model=VoiceList, tags=['Speech synthesis'], operation_id='listVoices')
def list_voices(request: fastapi.Request) -> VoiceList:
    engine = request.app.state.tts
    if engine is None:
        return VoiceList(available=False, loading=False, voices=[])
    voices = [VoiceInfo(id=v.id, name=v.name, description=v.description) for v in engine.voices()]
    return VoiceList(available=engine.ready, loading=not engine.ready and engine.error is None, voices=voices)


@router.post('/speech/synthesize', tags=['Speech synthesis'], operation_id='synthesizeSpeech',
             responses={200: {'content': {'audio/wav': {}}, 'description': 'Ответ, озвученный выбранным голосом'}})
def synthesize_speech(body: SynthesizeRequest, request: fastapi.Request) -> fastapi.Response:
    from src.services.tts import TTSUnavailableError

    engine = request.app.state.tts
    if engine is None:
        raise fastapi.HTTPException(status_code=503, detail='Озвучка на сервере отключена')
    if not request.app.state.tts_limiter.allow(current_user()):
        raise fastapi.HTTPException(status_code=429, detail='Слишком много запросов на озвучку, подождите минуту')
    try:
        audio = engine.synthesize(body.text, body.voice)
    except KeyError:
        raise fastapi.HTTPException(status_code=404, detail=f'Нет голоса «{body.voice}»')
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=422, detail=str(exc))
    except TTSUnavailableError as exc:
        raise fastapi.HTTPException(status_code=503, detail=str(exc))
    return fastapi.Response(content=audio, media_type='audio/wav', headers={'Cache-Control': 'private, max-age=3600'})
