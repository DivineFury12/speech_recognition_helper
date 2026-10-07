import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager

import gigaam
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_NAME = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE = os.getenv("GIGAAM_DEVICE", "auto").lower()
ml = {}
infer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        MODEL_NAME, device=device, download_root="/cache/gigaam"
    )
    ml["device"] = str(next(ml["model"].parameters()).device)
    yield
    ml.clear()


app = FastAPI(title="GigaAM-v3 ASR", lifespan=lifespan)


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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": ml["device"]}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        wav = prepare(file.file.read(), file.filename, tmp)
        try:
            with infer_lock:
                result = ml["model"].transcribe(wav)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"text": result.text}


@app.post("/transcribe_longform")
def transcribe_longform(file: UploadFile = File(...)):
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