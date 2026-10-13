"""Озвучка ответов голосами XTTS-v2: клоны по образцам из backend/voices и встроенный голос."""

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import threading
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .speech_text import for_speech

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
CACHE_ITEMS = 64
MAX_CHARS = 600
CHUNK_CHARS = 170
PAUSE_SECONDS = 0.18
STUDIO_FILTER = (
    "highpass=f=70,lowpass=f=11500,afftdn=nr=6:nf=-50:tn=1,deesser=i=0.3,"
    "acompressor=threshold=-21dB:ratio=2.5:attack=8:release=80:makeup=1.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=8"
)


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
    file: str | None = None
    builtin: str | None = None


VOICES = [
    Voice("nikita", "Никита", "мужской голос", file="nikita.wav"),
    Voice("sergey", "Сергей", "мужской голос", file="sergey.wav"),
    Voice("alina", "Алина", "женский голос", builtin="Asya Anara"),
]


class TTSUnavailableError(RuntimeError):
    pass


def _load_audio(path, sampling_rate):
    import soundfile
    import torch
    import torchaudio

    data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy())
    if audio.size(0) != 1:
        audio = audio.mean(dim=0, keepdim=True)
    if rate != sampling_rate:
        audio = torchaudio.functional.resample(audio, rate, sampling_rate)
    return audio.clamp_(-1, 1)


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


GENERATION = {"temperature": 0.55, "top_p": 0.8, "repetition_penalty": 5.0}


class XttsEngine:
    def __init__(self, model_dir: str, voices_dir: str, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self.voices_dir = Path(voices_dir)
        self.device = device
        self._model = None
        self._latents: dict[str, tuple] = {}
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
        if self.ready:
            return [v for v in VOICES if v.id in self._latents]
        return [v for v in VOICES if v.builtin or (self.voices_dir / v.file).is_file()]

    def load(self) -> None:
        try:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            import torch
            import TTS.tts.models.xtts as xtts_module
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            xtts_module.load_audio = _load_audio
            if not (self.model_dir / "model.pth").is_file():
                raise TTSUnavailableError(f"Нет весов XTTS в {self.model_dir}")
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            config = XttsConfig()
            config.load_json(str(self.model_dir / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=str(self.model_dir), use_deepspeed=False)
            model.to(device).eval()
            speakers = model.speaker_manager.speakers if model.speaker_manager else {}
            for v in VOICES:
                if v.builtin and v.builtin in speakers:
                    s = speakers[v.builtin]
                    self._latents[v.id] = (s["gpt_cond_latent"], s["speaker_embedding"])
                elif v.file and (self.voices_dir / v.file).is_file():
                    latent, emb = model.get_conditioning_latents(audio_path=[str(self.voices_dir / v.file)])
                    self._latents[v.id] = (latent, emb)
                else:
                    log.warning("Голос %s пропущен: нет образца", v.id)
            self._model = model
            self._ready.set()
            log.info("Озвучка XTTS готова на %s, голоса: %s", device, ", ".join(self._latents))
        except Exception as e:
            self._error = str(e)
            log.error("Озвучка недоступна: %s", e)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        if not self.ready:
            raise TTSUnavailableError(self._error or "Модель озвучки ещё загружается")
        if voice_id not in self._latents:
            raise KeyError(voice_id)
        spoken = for_speech(text)[:MAX_CHARS]
        if not spoken:
            raise ValueError("Пустой текст")
        key = hashlib.sha1(f"{voice_id}\0{spoken}".encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            import numpy as np

            latent, emb = self._latents[voice_id]
            pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype="float32")
            parts = []
            for chunk in split_for_tts(spoken):
                out = self._model.inference(
                    chunk, "ru", latent, emb, enable_text_splitting=False, **GENERATION,
                )
                parts += [np.asarray(out["wav"], dtype="float32"), pause]
            audio = studio(_wav_bytes(np.concatenate(parts[:-1])))
            self._cache[key] = audio
            if len(self._cache) > CACHE_ITEMS:
                self._cache.popitem(last=False)
            return audio


def create_engine(enabled: bool, model_dir: str, voices_dir: str, device: str = "auto") -> XttsEngine | None:
    if not enabled:
        return None
    try:
        import TTS  # noqa: F401
    except ImportError:
        log.warning("Озвучка отключена: не установлен пакет coqui-tts")
        return None
    return XttsEngine(model_dir, voices_dir, device)
