"""Скачивает веса моделей заранее, чтобы сервер стартовал без интернета.

Запуск: python -m assistant.models_download [--tts] [--embeddings]
"""

import argparse
import sys

from .config import get_settings

XTTS_REPO = "coqui/XTTS-v2"
XTTS_FILES = ["config.json", "vocab.json", "speakers_xtts.pth", "dvae.pth", "mel_stats.pth", "model.pth"]


def download_xtts(target: str, attempts: int = 6) -> None:
    from huggingface_hub import hf_hub_download

    for name in XTTS_FILES:
        for attempt in range(1, attempts + 1):
            try:
                hf_hub_download(XTTS_REPO, name, local_dir=target)
                print(f"XTTS: {name} — готово")
                break
            except Exception as e:
                if attempt == attempts:
                    raise
                print(f"XTTS: {name} — повтор {attempt} ({type(e).__name__})")


def download_embeddings(name: str, cache: str) -> None:
    from .embeddings import create_embedding_model

    model = create_embedding_model(name, cache)
    if model is None:
        print("Эмбеддинги отключены — пропускаю")
        return
    model.load()
    print(f"Эмбеддинги: {model.name} — готово")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts", action="store_true", help="только модель озвучки")
    parser.add_argument("--embeddings", action="store_true", help="только модель эмбеддингов")
    args = parser.parse_args()
    both = not args.tts and not args.embeddings
    settings = get_settings()
    if args.tts or both:
        download_xtts(settings.tts_model_dir)
    if args.embeddings or both:
        download_embeddings(settings.embedding_model, settings.embedding_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
