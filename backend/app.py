import logging
import contextlib
import threading

import fastapi
import fastapi.exceptions
import fastapi.responses
import fastapi.staticfiles
from fastapi.middleware.cors import CORSMiddleware

import src.adapters.transcriber
import src.adapters.ffmeg
import src.adapters.gigachat
import src.adapters.rules_model
import src.adapters.local_embeddings
import src.adapters.entry_repository
import src.adapters.conversation_repository
import src.adapters.assistant_request_repository
import src.api.assistant.router
import src.api.conversations.router
import src.api.entries.router
import src.api.transcribe.router
import src.api.extras.router
from src.api.auth import Lockout, RateLimiter, SessionMiddleware, Sessions, load_secret
import src.db
import src.errors
import src.services.assistant
import src.services.command_extractor
import src.services.entries
import src.services.transcriber
import src.settings


logger = logging.getLogger(__name__)


class RedactSecrets(logging.Filter):
    """Keep the personal calendar link token out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            if args[2].startswith('/v1/calendar/feed/'):
                record.args = (*args[:2], '/v1/calendar/feed/…', *args[3:])
        return True


logging.getLogger('uvicorn.access').addFilter(RedactSecrets())


class LLMHealth:
    """Remember whether the last GigaChat call worked, for the status panel."""

    def __init__(self, client):
        self._client = client
        self.healthy = True

    def complete_json(self, *args, **kwargs):
        try:
            result = self._client.complete_json(*args, **kwargs)
        except Exception:
            self.healthy = False
            raise
        self.healthy = True
        return result

    def close(self):
        self._client.close()

    def __getattr__(self, name):
        return getattr(self._client, name)


def create_app(settings: src.settings.Settings | None = None) -> fastapi.FastAPI:
    if settings is None:
        src.settings.load_env_file()
    config = settings or src.settings.Settings.from_env()
    sessions_store = Sessions(load_secret(config.session_secret, str(config.data_dir / 'assistant')),
                              config.access_code)

    @contextlib.asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        if config.voice_enabled:
            transcriber = src.adapters.transcriber.GigaAMTransciber.load(
                model_name=config.model_name,
                requested_device=config.device,
                cache_dir=config.cache_dir,
            )
            converter = src.adapters.ffmeg.FFmpegConverter(
                timeout_seconds=config.ffmpeg_timeout_seconds,
            )
            app.state.transcription_service = src.services.transcriber.TranscriptionService(
                converter=converter,
                transcriber=transcriber,
            )
            app.state.health_info = {'model': config.model_name, 'device': transcriber.device}
        else:
            logger.info('Голос выключен (VOICE_ENABLED=false): работаем только с текстом')
            app.state.transcription_service = None
            app.state.health_info = {'model': 'voice-disabled', 'device': 'none'}

        app.state.tts = None
        if config.tts_enabled:
            from src.services.tts import create_engine

            app.state.tts = create_engine(True, str(config.tts_model_dir), 'cpu')
            if app.state.tts is not None:
                threading.Thread(target=app.state.tts.load, daemon=True).start()

        gigachat_client = None
        engine = None
        if config.gigachat_auth_key:
            gigachat_client = LLMHealth(src.adapters.gigachat.GigaChatClient(
                auth_key=config.gigachat_auth_key,
                scope=config.gigachat_scope,
                model=config.gigachat_model,
                ca_bundle=config.gigachat_ca_bundle,
            ))
            app.state.llm_health = gigachat_client
            gigachat_client = src.adapters.rules_model.FallbackModel(gigachat_client)
        else:
            logger.warning('Ключ GigaChat не задан: команды разбираются правилами')
            gigachat_client = src.adapters.rules_model.RulesModel()

        if config.database_url is not None:
            sessions, engine = src.db.create_session_factory(config.database_url)
            app.state.entries_service = src.services.entries.EntryService(
                src.adapters.entry_repository.SqlAlchemyEntryRepository(sessions),
                src.adapters.local_embeddings.create_embedding_model(
                    name=config.embedding_model_name,
                    cache_dir=str(config.cache_dir.parent / 'embeddings'),
                ),
                topic_client=gigachat_client if config.gigachat_auth_key else None,
            )
            app.state.conversations = (
                src.adapters.conversation_repository.SqlAlchemyConversationRepository(sessions)
            )
            app.state.user_settings = (
                src.adapters.conversation_repository.SqlAlchemyUserSettingsRepository(sessions)
            )

        if gigachat_client is not None:
            app.state.assistant_service = src.services.assistant.AssistantService(
                src.services.command_extractor.ExternalCommandExtractor(
                    gigachat_client, temporal_retry=bool(config.gigachat_auth_key),
                ),
                app.state.entries_service,
                app.state.conversations,
                src.adapters.assistant_request_repository.SqlAlchemyAssistantRequestRepository(
                    sessions,
                ) if config.database_url is not None else None,
            )

        try:
            yield
        finally:
            app.state.transcription_service = None
            app.state.assistant_service = None
            app.state.entries_service = None
            app.state.conversations = None
            if gigachat_client is not None:
                gigachat_client.close()
            if engine is not None:
                engine.dispose()

    app = fastapi.FastAPI(
        title='GigaAM-v3 ASR',
        lifespan=lifespan,
    )
    app.state.assistant_service = None
    app.state.entries_service = None
    app.state.conversations = None
    app.state.user_settings = None
    app.state.transcription_service = None
    app.state.tts = None
    app.state.llm_health = None
    app.state.voice_enabled = config.voice_enabled
    app.state.sessions = sessions_store
    app.state.assistant_limiter = RateLimiter(config.rate_limit_per_minute, 60)
    app.state.unlock_limiter = RateLimiter(10, 600)
    app.state.unlock_lockout = Lockout(5, 900)
    app.state.tts_limiter = RateLimiter(60, 60)
    app.add_middleware(SessionMiddleware, sessions=sessions_store, cookie_secure=config.cookie_secure)
    if config.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(config.cors_origins), allow_credentials=True,
                           allow_methods=['*'], allow_headers=['*'])
    app.include_router(src.api.transcribe.router.router)
    app.include_router(src.api.assistant.router.router)
    app.include_router(src.api.conversations.router.router)
    app.include_router(src.api.extras.router.router)
    app.include_router(src.api.entries.router.router)

    @app.exception_handler(src.errors.AudioConversionError)
    async def handle_conversion_error(
        request: fastapi.Request,
        exc: src.errors.AudioConversionError,
    ) -> fastapi.responses.JSONResponse:
        logger.warning('Audio conversion failed: %s', exc)
        return fastapi.responses.JSONResponse(
            status_code=400,
            content={'detail': str(exc)},
        )

    @app.exception_handler(src.errors.InvalidAudioError)
    async def handle_invalid_audio(
        request: fastapi.Request,
        exc: src.errors.InvalidAudioError,
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(
            status_code=400,
            content={'detail': str(exc)},
        )

    @app.exception_handler(src.errors.TranscriptionError)
    async def handle_transcription_error(
        request: fastapi.Request,
        exc: src.errors.TranscriptionError,
    ) -> fastapi.responses.JSONResponse:
        logger.error(
            'Transcription failed',
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return fastapi.responses.JSONResponse(
            status_code=500,
            content={'detail': 'Ошибка при распознавании аудио'},
        )

    @app.exception_handler(fastapi.exceptions.RequestValidationError)
    async def handle_validation_error(
        request: fastapi.Request,
        exc: fastapi.exceptions.RequestValidationError,
    ) -> fastapi.responses.JSONResponse:
        errors = exc.errors()
        if any(err.get('type') == 'string_too_long' and 'text' in err.get('loc', ()) for err in errors):
            detail = 'Сообщение слишком длинное: не больше 1000 символов. Сократите его и отправьте ещё раз.'
        else:
            parts = []
            for err in errors:
                field = '.'.join(str(x) for x in err.get('loc', ()) if x not in ('body', 'query', 'path', 'form'))
                message = str(err.get('msg', '')).removeprefix('Value error, ')
                parts.append(f'{field}: {message}' if field else message)
            detail = '; '.join(parts) or 'Некорректный запрос'
        return fastapi.responses.JSONResponse(status_code=422, content={'detail': detail})

    @app.exception_handler(src.services.assistant.InvalidTimezoneError)
    async def handle_invalid_timezone(
        request: fastapi.Request,
        exc: src.services.assistant.InvalidTimezoneError,
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(status_code=422, content={'detail': str(exc)})

    @app.exception_handler(src.adapters.assistant_request_repository.RequestIdConflictError)
    async def handle_request_id_conflict(
        request: fastapi.Request,
        exc: src.adapters.assistant_request_repository.RequestIdConflictError,
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(status_code=409, content={'detail': str(exc)})

    @app.exception_handler(src.adapters.gigachat.GigaChatAPIError)
    @app.exception_handler(src.services.command_extractor.InvalidModelResponseError)
    async def handle_model_error(
        request: fastapi.Request,
        exc: Exception,
    ) -> fastapi.responses.JSONResponse:
        logger.error('External model failed', exc_info=(type(exc), exc, exc.__traceback__))
        return fastapi.responses.JSONResponse(
            status_code=502,
            content={'detail': 'Не удалось обработать команду внешней моделью'},
        )

    if config.frontend_dist is not None and (config.frontend_dist / 'index.html').is_file():
        app.mount('/', fastapi.staticfiles.StaticFiles(directory=config.frontend_dist, html=True), name='frontend')
    return app


app = create_app()
