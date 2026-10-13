"""Скачивает веса моделей заранее, чтобы сервер стартовал без интернета.

Запуск: python -m assistant.models_download [--tts] [--embeddings] [--asr]
"""

import argparse
import os
import sys
from pathlib import Path

from .config import get_settings

def download_tts(target: str) -> None:
    from .tts import download_model

    print(f"Silero TTS: {download_model(target)} — готово")


def download_embeddings(name: str, cache: str) -> None:
    from .embeddings import create_embedding_model

    model = create_embedding_model(name, cache)
    if model is None:
        print("Эмбеддинги отключены — пропускаю")
        return
    model.load()
    print(f"Эмбеддинги: {model.name} — готово")


def download_asr() -> None:
    try:
        import gigaam
    except ImportError:
        print("GigaAM не установлен — пропускаю модель распознавания")
        return
    name = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
    cache = os.getenv("GIGAAM_CACHE") or str(Path(__file__).resolve().parents[1] / ".cache" / "gigaam")
    gigaam.load_model(name, device="cpu", download_root=cache)
    print(f"GigaAM: {name} — готово")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts", action="store_true", help="только модель озвучки")
    parser.add_argument("--embeddings", action="store_true", help="только модель эмбеддингов")
    parser.add_argument("--asr", action="store_true", help="только модель распознавания речи")
    args = parser.parse_args()
    both = not args.tts and not args.embeddings and not args.asr
    settings = get_settings()
    if args.tts or both:
        download_tts(settings.tts_model_dir)
    if args.embeddings or both:
        download_embeddings(settings.embedding_model, settings.embedding_cache)
    if args.asr or both:
        download_asr()
    return 0


if __name__ == "__main__":
    sys.exit(main())
