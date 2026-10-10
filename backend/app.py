import logging
import contextlib

import fastapi
import fastapi.responses

import src.adapters.transcriber
import src.adapters.ffmeg
import src.adapters.gigachat
import src.adapters.local_embeddings
import src.adapters.entry_repository
import src.api.assistant.router
import src.api.entries.router
import src.api.transcribe.router
import src.db
import src.errors
import src.services.assistant
import src.services.command_extractor
import src.services.entries
import src.services.transcriber
import src.settings


logger = logging.getLogger(__name__)


def create_app(settings: src.settings.Settings | None = None) -> fastapi.FastAPI:
    config = settings or src.settings.Settings.from_env()

    @contextlib.asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        config.cache_dir.mkdir(parents=True, exist_ok=True)

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
        app.state.health_info = {
            'model': config.model_name,
            'device': transcriber.device,
        }

        gigachat_client = None
        engine = None
        if config.gigachat_auth_key:
            gigachat_client = src.adapters.gigachat.GigaChatClient(
                auth_key=config.gigachat_auth_key,
                scope=config.gigachat_scope,
                model=config.gigachat_model,
                ca_bundle=config.gigachat_ca_bundle,
            )

        if config.database_url is not None:
            sessions, engine = src.db.create_session_factory(config.database_url)
            app.state.entries_service = src.services.entries.EntryService(
                src.adapters.entry_repository.SqlAlchemyEntryRepository(sessions),
                src.adapters.local_embeddings.create_embedding_model(
                    name=config.embedding_model_name,
                    cache_dir=str(config.cache_dir.parent / 'embeddings'),
                ),
            )

        if gigachat_client is not None:
            app.state.assistant_service = src.services.assistant.AssistantService(
                src.services.command_extractor.ExternalCommandExtractor(gigachat_client),
                app.state.entries_service,
            )

        try:
            yield
        finally:
            app.state.transcription_service = None
            app.state.assistant_service = None
            app.state.entries_service = None
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
    app.include_router(src.api.transcribe.router.router)
    app.include_router(src.api.assistant.router.router)
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

    @app.exception_handler(src.services.assistant.InvalidTimezoneError)
    async def handle_invalid_timezone(
        request: fastapi.Request,
        exc: src.services.assistant.InvalidTimezoneError,
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(status_code=422, content={'detail': str(exc)})

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

    return app


app = create_app()
