"""Private CPU model service: ASR, embeddings and optional speech synthesis."""

import contextlib
import os
import pathlib
import threading
from typing import Literal

import fastapi
import fastapi.responses
import pydantic

from src.adapters.ffmeg import FFmpegConverter
from src.adapters.local_embeddings import create_embedding_model
from src.adapters.transcriber import GigaAMTransciber
from src.errors import AudioConversionError, InvalidAudioError, TranscriptionError
from src.ports.embeddings import EmbeddingUnavailableError
from src.schemas.transcribe import TranscribeLongformResponse, TranscribeResponse
from src.services.transcriber import TranscriptionService
from src.services.tts import TTSUnavailableError, create_engine


class EmbedRequest(pydantic.BaseModel):
    text: str = pydantic.Field(min_length=1, max_length=2000)
    purpose: Literal['document', 'query']


class SynthesizeRequest(pydantic.BaseModel):
    text: str = pydantic.Field(min_length=1, max_length=1000)
    voice: str = pydantic.Field(default='ksenia', min_length=1, max_length=40)


def create_inference_app() -> fastapi.FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        cache = pathlib.Path(os.getenv('MODEL_CACHE_DIR', '/cache'))
        cache.mkdir(parents=True, exist_ok=True)
        voice_enabled = os.getenv('VOICE_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
        if voice_enabled:
            model_name = os.getenv('GIGAAM_MODEL', 'v3_e2e_rnnt')
            model = GigaAMTransciber.load(
                model_name=model_name,
                requested_device=os.getenv('GIGAAM_DEVICE', 'cpu'),
                cache_dir=cache / 'gigaam',
            )
            app.state.transcription = TranscriptionService(
                FFmpegConverter(timeout_seconds=int(os.getenv('FFMPEG_TIMEOUT_SECONDS', '120'))),
                model,
            )
            app.state.health = {'status': 'ok', 'model': model_name, 'device': model.device}
        else:
            app.state.health = {'status': 'ok', 'model': 'voice-disabled', 'device': 'none'}
        app.state.embedding = create_embedding_model(
            os.getenv('EMBEDDING_MODEL', 'e5'), str(cache / 'embeddings'),
        )
        app.state.tts = None
        if voice_enabled and os.getenv('TTS_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on'):
            app.state.tts = create_engine(True, str(cache / 'tts'), 'cpu')
            if app.state.tts is not None:
                threading.Thread(target=app.state.tts.load, daemon=True).start()
        try:
            yield
        finally:
            app.state.transcription = None
            app.state.embedding = None
            app.state.tts = None

    app = fastapi.FastAPI(title='Assistant inference', lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    app.state.transcription = None
    app.state.embedding = None
    app.state.tts = None
    app.state.health = {'status': 'loading'}

    @app.get('/health')
    def health(request: fastapi.Request):
        return request.app.state.health

    @app.post('/transcribe', response_model=TranscribeResponse)
    def transcribe(request: fastapi.Request, file: fastapi.UploadFile = fastapi.File(...)):
        if request.app.state.transcription is None:
            raise fastapi.HTTPException(status_code=503, detail='Распознавание речи отключено')
        return TranscribeResponse(
            text=request.app.state.transcription.transribe(file.file, file.filename),
        )

    @app.post('/transcribe_longform', response_model=TranscribeLongformResponse)
    def transcribe_longform(request: fastapi.Request,
                            file: fastapi.UploadFile = fastapi.File(...)):
        if request.app.state.transcription is None:
            raise fastapi.HTTPException(status_code=503, detail='Распознавание речи отключено')
        return TranscribeLongformResponse(segments=
            request.app.state.transcription.transcribe_longform(file.file, file.filename))

    @app.post('/embed')
    def embed(request: fastapi.Request, body: EmbedRequest):
        model = request.app.state.embedding
        vector = (model.embed_document(body.text) if body.purpose == 'document'
                  else model.embed_query(body.text))
        return {'model': model.name, 'vector': vector}

    @app.get('/speech/voices')
    def voices(request: fastapi.Request):
        engine = request.app.state.tts
        if engine is None:
            return {'available': False, 'loading': False, 'voices': []}
        return {'available': engine.ready, 'loading': not engine.ready and engine.error is None,
                'voices': [{'id': voice.id, 'name': voice.name,
                            'description': voice.description} for voice in engine.voices()]}

    @app.post('/speech/synthesize')
    def synthesize(request: fastapi.Request, body: SynthesizeRequest):
        engine = request.app.state.tts
        if engine is None:
            raise fastapi.HTTPException(status_code=503, detail='Озвучка отключена')
        try:
            audio = engine.synthesize(body.text, body.voice)
        except KeyError as exc:
            raise fastapi.HTTPException(status_code=404, detail='Голос не найден') from exc
        except ValueError as exc:
            raise fastapi.HTTPException(status_code=422, detail=str(exc)) from exc
        except TTSUnavailableError as exc:
            raise fastapi.HTTPException(status_code=503, detail=str(exc)) from exc
        return fastapi.Response(audio, media_type='audio/wav')

    @app.exception_handler(AudioConversionError)
    @app.exception_handler(InvalidAudioError)
    async def invalid_audio(_request: fastapi.Request, exc: Exception):
        return fastapi.responses.JSONResponse(status_code=400, content={'detail': str(exc)})

    @app.exception_handler(TranscriptionError)
    @app.exception_handler(EmbeddingUnavailableError)
    async def inference_error(_request: fastapi.Request, _exc: Exception):
        return fastapi.responses.JSONResponse(status_code=503,
                                              content={'detail': 'Модель временно недоступна'})

    return app


app = create_inference_app()
