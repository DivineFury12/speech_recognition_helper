"""Озвучка ответов моделью Silero TTS (русская, работает на процессоре)."""

import hashlib
import io
import logging
import re
import shutil
import subprocess
import threading
import urllib.request
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .speech_text import for_speech

log = logging.getLogger(__name__)

MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
MODEL_FILE = "v4_ru.pt"
SAMPLE_RATE = 48000
CACHE_ITEMS = 64
MAX_CHARS = 600
CHUNK_CHARS = 300
PAUSE_SECONDS = 0.15
STUDIO_FILTER = "highpass=f=60,loudnorm=I=-16:TP=-1.5:LRA=8"


def _cut_long(sentence: str, limit: int) -> list[str]:
    pieces = []
    while len(sentence) > limit:
        cut = sentence.rfind(", ", 0, limit)
        if cut < limit // 3:
            cut = sentence.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        pieces.append(sentence[:cut].strip(" ,"))
        sentence = sentence[cut:].strip(" ,")
    if sentence:
        pieces.append(sentence)
    return pieces


def split_for_tts(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    sentences = []
    for sentence in re.split(r"(?<=[.!?…;:])\s+", text.strip()):
        sentences += _cut_long(sentence.strip(), limit)
    chunks: list[str] = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= limit:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
    return [c for c in chunks if c]


def studio(audio: bytes) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return audio
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
             "-af", STUDIO_FILTER, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1"],
            input=audio, capture_output=True, timeout=30,
        )
        return r.stdout if r.returncode == 0 and len(r.stdout) > 44 else audio
    except (OSError, subprocess.SubprocessError):
        return audio


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    description: str
    speaker: str


VOICES = [
    Voice("alina", "Алина", "женский голос", "xenia"),
    Voice("ksenia", "Ксения", "женский голос", "kseniya"),
    Voice("aidar", "Айдар", "мужской голос", "aidar"),
    Voice("evgeny", "Евгений", "мужской голос", "eugene"),
]


class TTSUnavailableError(RuntimeError):
    pass


def _wav_bytes(samples) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def download_model(model_dir: str) -> Path:
    path = Path(model_dir) / MODEL_FILE
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class SileroEngine:
    def __init__(self, model_dir: str, device: str = "cpu", threads: int = 4) -> None:
        self.model_dir = model_dir
        self.device = device
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: str | None = None
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def error(self) -> str | None:
        return self._error

    def voices(self) -> list[Voice]:
        return VOICES

    def load(self) -> None:
        try:
            import torch

            path = download_model(self.model_dir)
            device = self.device
            if device == "auto":
                device = "cpu"
            if device == "cpu":
                torch.set_num_threads(self.threads)
            model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
            model.to(torch.device(device))
            model.apply_tts(text="Готово.", speaker=VOICES[0].speaker, sample_rate=SAMPLE_RATE)
            self._model = model
            self._ready.set()
            log.info("Озвучка Silero готова на %s", device)
        except Exception as e:
            self._error = str(e)
            log.error("Озвучка недоступна: %s", e)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        if not self.ready:
            raise TTSUnavailableError(self._error or "Модель озвучки ещё загружается")
        voice = next((v for v in VOICES if v.id == voice_id), None)
        if voice is None:
            raise KeyError(voice_id)
        spoken = for_speech(text)[:MAX_CHARS]
        if not re.search(r"[а-яёa-z0-9]", spoken, re.I):
            raise ValueError("Пустой текст")
        key = hashlib.sha1(f"{voice_id}\0{spoken}".encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            import numpy as np

            pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype="float32")
            parts = []
            for chunk in split_for_tts(spoken):
                out = self._model.apply_tts(text=chunk, speaker=voice.speaker, sample_rate=SAMPLE_RATE,
                                            put_accent=True, put_yo=True)
                parts += [out.numpy().astype("float32"), pause]
            audio = studio(_wav_bytes(np.concatenate(parts[:-1])))
            self._cache[key] = audio
            if len(self._cache) > CACHE_ITEMS:
                self._cache.popitem(last=False)
            return audio


def create_engine(enabled: bool, model_dir: str, device: str = "cpu") -> SileroEngine | None:
    if not enabled:
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("Озвучка отключена: не установлен torch")
        return None
    return SileroEngine(model_dir, device)
