import logging
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from assistant.api import router as assistant_router
from assistant.auth import Lockout, RateLimiter, SessionMiddleware, Sessions, load_secret
from assistant.config import get_settings
from assistant.embeddings import EmbeddingIndex, create_embedding_model
from assistant.tts import create_engine as create_tts_engine
from assistant.llm import LLMError, create_provider
from assistant.llm.base import TrackedProvider
from assistant.service import Assistant
from assistant.storage import Storage

MODEL_NAME = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE = os.getenv("GIGAAM_DEVICE", "auto").lower()
GIGAAM_CACHE = os.getenv(
    "GIGAAM_CACHE",
    "/cache/gigaam" if os.path.isdir("/cache") else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gigaam"),
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BODY_BYTES = 6 * 1024 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "media-src 'self' blob:; connect-src 'self'; font-src 'self' data:; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'; object-src 'none'"
    ),
}
ml = {}
infer_lock = threading.Lock()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


class RedactSecrets(logging.Filter):
    """Убирает из журнала запросов одноразовый код Google и токен ссылки на календарь."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path = args[2]
            if path.startswith("/v1/google/callback"):
                path = "/v1/google/callback?…"
            elif path.startswith("/v1/calendar/feed/"):
                path = "/v1/calendar/feed/…"
            record.args = (*args[:2], path, *args[3:])
        return True


logging.getLogger("uvicorn.access").addFilter(RedactSecrets())
settings = get_settings()


def find_ffmpeg() -> str | None:
    candidates = [os.getenv("FFMPEG_BINARY"), shutil.which("ffmpeg")]
    local = os.getenv("LOCALAPPDATA")
    if local:
        candidates.append(os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"))
    for path in candidates:
        if path and os.path.isfile(path):
            folder = os.path.dirname(os.path.abspath(path))
            if folder not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            return path
    return None


FFMPEG = find_ffmpeg()


def load_asr():
    import gigaam
    import torch

    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("GIGAAM_DEVICE должен быть auto, cpu или cuda")
    device = "cuda" if DEVICE == "auto" and torch.cuda.is_available() else DEVICE
    if device == "auto":
        device = "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU запрошен, но CUDA недоступна внутри контейнера")

        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    ml["model"] = gigaam.load_model(
        MODEL_NAME, device=device, download_root=GIGAAM_CACHE
    )
    ml["device"] = str(next(ml["model"].parameters()).device)


def _warm_up(model) -> None:
    try:
        model.load()
        log.info("Модель эмбеддингов %s загружена", model.name)
    except Exception as e:
        log.warning("Модель эмбеддингов не загрузилась, поиск будет без неё: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.voice_enabled:
        log.info("Голос выключен (VOICE_ENABLED=false): работаем только с текстом")
    if settings.asr_enabled:
        try:
            load_asr()
        except ImportError as e:
            log.error("GigaAM не установлен, распознавание речи отключено: %s", e)
        if "model" in ml:
            log.info("ffmpeg: %s", FFMPEG or "НЕ НАЙДЕН — голосовые запросы будут отвечать 503")
    storage = Storage(settings.db_path)
    try:
        llm = create_provider(settings)
        llm = TrackedProvider(llm) if llm is not None else None
    except LLMError as e:
        log.error("LLM недоступна, работаем на правилах: %s", e)
        llm = None
    log.info("LLM provider: %s", llm.name if llm else "rules")
    if sessions.access_code:
        log.info("Включён код доступа (ACCESS_CODE)")
    index = None
    try:
        model = create_embedding_model(settings.embedding_model, settings.embedding_cache)
    except ValueError as e:
        log.error("%s — поиск по смыслу через эмбеддинги отключён", e)
        model = None
    if model is not None:
        index = EmbeddingIndex(storage, model)
        threading.Thread(target=_warm_up, args=(model,), daemon=True).start()
    log.info("Embeddings: %s", model.name if model else "нет (поиск по словам и через LLM)")
    tts = create_tts_engine(settings.tts_enabled, settings.tts_model_dir, settings.tts_device)
    app.state.tts = tts
    if tts is not None:
        threading.Thread(target=tts.load, daemon=True).start()
        log.info("Озвучка: загружаю Silero в фоне (голоса: %s)", ", ".join(v.name for v in tts.voices()))
    app.state.google = None
    if settings.google_enabled and settings.google_client_id and settings.google_client_secret:
        from assistant.google_calendar import GoogleCalendar

        app.state.google = GoogleCalendar(storage, sessions.secret, settings.google_client_id,
                                          settings.google_client_secret)
        log.info("Google Календарь: настроен")
    app.state.assistant = Assistant(storage, llm, index)
    app.state.llm_name = llm.name if llm else "rules"
    app.state.transcribe_upload = transcribe_upload
    yield
    if llm is not None and hasattr(llm, "close"):
        llm.close()
    storage.close()
    ml.clear()


sessions = Sessions(load_secret(settings.session_secret, settings.db_path), settings.access_code)

app = FastAPI(title="Голосовой ассистент для задач и встреч", lifespan=lifespan)
app.state.sessions = sessions
app.state.assistant_limiter = RateLimiter(settings.rate_limit_per_minute, 60)
app.state.unlock_limiter = RateLimiter(10, 600)
app.state.unlock_lockout = Lockout(5, 900)
app.state.google = None
app.state.google_origins = settings.google_origins
app.state.tts_limiter = RateLimiter(60, 60)
app.state.tts = None
app.state.voice_enabled = settings.voice_enabled
app.state.asr_ready = lambda: "model" in ml and FFMPEG is not None
app.state.llm_name = "rules"
app.add_middleware(SessionMiddleware, sessions=sessions, cookie_secure=settings.cookie_secure)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(assistant_router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Слишком большой запрос"})
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/v1/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path", "form"))
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "Некорректный запрос"})


def to_wav16k(src: str, dst: str):
    if FFMPEG is None:
        raise HTTPException(503, "ffmpeg не найден: установите его или укажите путь в FFMPEG_BINARY")
    r = subprocess.run(
        [FFMPEG, "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
        capture_output=True,
    )
    if r.returncode != 0:
        raise HTTPException(400, "Не удалось декодировать аудио")


def prepare(upload_bytes: bytes, filename: str, tmp: str) -> str:
    suffix = os.path.splitext(filename or "")[1] or ".bin"
    src = os.path.join(tmp, f"in{suffix}")
    dst = os.path.join(tmp, "out.wav")
    with open(src, "wb") as f:
        f.write(upload_bytes)
    to_wav16k(src, dst)
    return dst


def require_asr():
    if "model" not in ml:
        raise HTTPException(503, "Распознавание речи отключено (ASR_ENABLED=false)")


def transcribe_upload(file: UploadFile) -> str:
    require_asr()
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Пустой аудиофайл")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Аудиофайл слишком большой")
    with tempfile.TemporaryDirectory() as tmp:
        wav = prepare(data, file.filename, tmp)
        try:
            with infer_lock:
                result = ml["model"].transcribe(wav)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return result.text


@app.get("/health")
def health():
    if "model" not in ml:
        return {"status": "ok", "model": "asr-disabled", "device": "none"}
    return {"status": "ok", "model": MODEL_NAME, "device": ml["device"]}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    return {"text": transcribe_upload(file)}


@app.post("/transcribe_longform")
def transcribe_longform(file: UploadFile = File(...)):
    require_asr()
    with tempfile.TemporaryDirectory() as tmp:
        data = file.file.read(MAX_UPLOAD_BYTES * 4 + 1)
        if len(data) > MAX_UPLOAD_BYTES * 4:
            raise HTTPException(413, "Аудиофайл слишком большой")
        wav = prepare(data, file.filename, tmp)
        try:
            with infer_lock:
                result = ml["model"].transcribe_longform(wav, fr_batch_size=1)
        except Exception as e:
            raise HTTPException(
                500, f"longform недоступен (проверьте HF_TOKEN и gigaam[longform]): {e}"
            )
    return {
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for s in result.segments
        ]
    }


if os.path.isfile(os.path.join(settings.frontend_dist, "index.html")):
    app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
