"""Локальные эмбеддинги для поиска по смыслу.

Модели и пулинг перенесены из backend/src/adapters/local_embeddings.py
ветки main (автор — Сергей Елисеев); здесь они подключены к SQLite-хранилищу.
"""

import logging
import math
import threading
from typing import Any, Protocol

log = logging.getLogger(__name__)

E5_NAME = "intfloat/multilingual-e5-small"
FRIDA_NAME = "ai-forever/FRIDA"
ALIASES = {"e5": E5_NAME, E5_NAME: E5_NAME, "frida": FRIDA_NAME, FRIDA_NAME: FRIDA_NAME}
THRESHOLDS = {E5_NAME: 0.85, FRIDA_NAME: 0.19}
MARGINS = {E5_NAME: 0.015, FRIDA_NAME: 0.04}
MAX_HITS = 3


class EmbeddingUnavailableError(Exception):
    pass


class EmbeddingModel(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class _TransformerEmbeddingModel:
    name: str
    dimensions: int
    document_prefix: str
    query_prefix: str

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        raise NotImplementedError

    def _pool(self, hidden, mask):
        raise NotImplementedError

    def load(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.name, cache_dir=self._cache_dir)
            self._model = self._load_model().to("cpu").eval()

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        import torch
        import torch.nn.functional as functional

        with self._lock:
            try:
                self._ensure_loaded()
                tokens = self._tokenizer(
                    [f"{prefix}: {t}" for t in texts], max_length=512, truncation=True,
                    padding=True, return_tensors="pt",
                )
                with torch.inference_mode():
                    hidden = self._model(**tokens).last_hidden_state
                    pooled = self._pool(hidden, tokens["attention_mask"])
                    vectors = functional.normalize(pooled, p=2, dim=1).tolist()
            except Exception as exc:
                raise EmbeddingUnavailableError(f"Не удалось получить вектор модели {self.name}: {exc}") from exc
        for v in vectors:
            if len(v) != self.dimensions or not all(math.isfinite(x) for x in v):
                raise EmbeddingUnavailableError(f"{self.name} вернула некорректный вектор")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.document_prefix) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prefix)[0]


class E5EmbeddingModel(_TransformerEmbeddingModel):
    name = E5_NAME
    dimensions = 384
    document_prefix = "query"
    query_prefix = "query"

    def _load_model(self):
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.name, cache_dir=self._cache_dir)

    def _pool(self, hidden, mask):
        expanded = mask.unsqueeze(-1)
        return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1)


class FridaEmbeddingModel(_TransformerEmbeddingModel):
    name = FRIDA_NAME
    dimensions = 1536
    document_prefix = "search_document"
    query_prefix = "search_query"

    def _load_model(self):
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(self.name, cache_dir=self._cache_dir)

    def _pool(self, hidden, mask):
        return hidden[:, 0]


def create_embedding_model(name: str, cache_dir: str) -> EmbeddingModel | None:
    key = (name or "").strip().lower()
    if key in ("", "none", "off", "false", "0"):
        return None
    full = ALIASES.get(key) or ALIASES.get(name.strip())
    if full is None:
        raise ValueError(f"Неизвестная модель эмбеддингов: {name} (ожидается e5, frida или none)")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        log.warning("Эмбеддинги отключены: не установлены torch/transformers (%s)", e)
        return None
    return E5EmbeddingModel(cache_dir) if full == E5_NAME else FridaEmbeddingModel(cache_dir)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingIndex:
    def __init__(self, storage, model: EmbeddingModel, threshold: float | None = None) -> None:
        self.storage = storage
        self.model = model
        self.threshold = threshold if threshold is not None else THRESHOLDS.get(model.name, 0.8)

    def index(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        vectors = self.model.embed_documents([e["title"] for e in entries])
        for entry, vector in zip(entries, vectors):
            self.storage.save_embedding(entry["id"], self.model.name, entry["title"], vector)

    def ensure(self, entries: list[dict[str, Any]], user_id: str) -> dict[str, list[float]]:
        known = self.storage.get_embeddings(user_id, self.model.name)
        missing = [e for e in entries if e["id"] not in known or known[e["id"]][0] != e["title"]]
        for start in range(0, len(missing), 64):
            self.index(missing[start: start + 64])
        if missing:
            known = self.storage.get_embeddings(user_id, self.model.name)
        return {k: v for k, (_, v) in known.items()}

    def rank(self, query: str, entries: list[dict[str, Any]], user_id: str) -> list[tuple[float, dict[str, Any]]]:
        if not entries or not query.strip():
            return []
        try:
            vectors = self.ensure(entries, user_id)
            q = self.model.embed_query(query)
        except EmbeddingUnavailableError as e:
            log.warning("Поиск по эмбеддингам недоступен: %s", e)
            return []
        scored = [(_cosine(q, vectors[e["id"]]), e) for e in entries if e["id"] in vectors]
        scored.sort(key=lambda s: s[0], reverse=True)
        if not scored or scored[0][0] < self.threshold:
            return []
        best = scored[0][0]
        margin = MARGINS.get(self.model.name, 0.02)
        return [s for s in scored if s[0] >= best - margin][:MAX_HITS]
