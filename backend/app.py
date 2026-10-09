import logging
import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from assistant.api import router as assistant_router
from assistant.config import get_settings
from assistant.llm import LLMError, create_provider
from assistant.service import Assistant
from assistant.storage import Storage

MODEL_NAME = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE = os.getenv("GIGAAM_DEVICE", "auto").lower()
ml = {}
infer_lock = threading.Lock()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
settings = get_settings()


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
        MODEL_NAME, device=device, download_root=os.getenv("GIGAAM_CACHE", "/cache/gigaam")
    )
    ml["device"] = str(next(ml["model"].parameters()).device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.asr_enabled:
        try:
            load_asr()
        except ImportError as e:
            log.error("GigaAM не установлен, распознавание речи отключено: %s", e)
    storage = Storage(settings.db_path)
    try:
        llm = create_provider(settings)
    except LLMError as e:
        log.error("LLM недоступна, работаем на правилах: %s", e)
        llm = None
    log.info("LLM provider: %s", llm.name if llm else "rules")
    app.state.assistant = Assistant(storage, llm)
    app.state.transcribe_upload = transcribe_upload
    yield
    if llm is not None and hasattr(llm, "close"):
        llm.close()
    storage.close()
    ml.clear()


app = FastAPI(title="Голосовой ассистент для задач и встреч", lifespan=lifespan)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(assistant_router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path", "form"))
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "Некорректный запрос"})


def to_wav16k(src: str, dst: str):
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
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
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Пустой аудиофайл")
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
        wav = prepare(file.file.read(), file.filename, tmp)
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
