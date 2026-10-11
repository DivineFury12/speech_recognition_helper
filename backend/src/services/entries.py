"""Entry operations, including optional semantic indexing and retrieval."""

import datetime
import logging
import math

from src.ports.embeddings import EmbeddingModel, EmbeddingUnavailableError
from src.ports.entries import EntryRepository
from src.schemas.assistant import Entry, EntryKind, EntryStatus


logger = logging.getLogger(__name__)


class EntryService:
    def __init__(self, repository: EntryRepository, embedder: EmbeddingModel | None = None):
        self._repository = repository
        self._embedder = embedder

    def _embedding(self, text: str, *, query: bool = False) -> list[float] | None:
        if self._embedder is None:
            return None
        try:
            vector = (
                self._embedder.embed_query(text) if query else self._embedder.embed_document(text)
            )
            if len(vector) != self._embedder.dimensions or not all(math.isfinite(v) for v in vector):
                raise ValueError(f'Invalid vector from {self._embedder.name}')
            return vector
        except (EmbeddingUnavailableError, ValueError):
            logger.warning('Embedding unavailable; using text search fallback', exc_info=True)
            return None

    def create(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str,
    ) -> Entry:
        embedding = self._embedding(title)
        return self._repository.create(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            embedding=embedding,
            embedding_model=self._embedder.name if embedding is not None else None,
        )

    def create_once(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str,
    ) -> tuple[Entry, bool]:
        """Return the matching pending entry when the same command is repeated."""
        title = ' '.join(title.split())
        embedding = self._embedding(title)
        return self._repository.create_once(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            embedding=embedding,
            embedding_model=self._embedder.name if embedding is not None else None,
        )

    def list_period(
        self, date_from: datetime.date, date_to: datetime.date, *,
        kind: EntryKind | None = None, status: EntryStatus | None = None,
    ) -> list[Entry]:
        return self._repository.list_period(date_from, date_to, kind=kind, status=status)

    def search(self, query: str, *, kind: EntryKind | None = None, limit: int = 5) -> list[Entry]:
        query = query.strip()
        if not query:
            return []
        matches = self._repository.search_text(query, kind=kind, limit=limit)
        if len(matches) >= limit:
            return matches
        vector = self._embedding(query, query=True)
        if vector is None:
            return matches
        seen = {entry.id for entry in matches}
        for entry in self._repository.search_similar(
            vector, embedding_model=self._embedder.name, kind=kind, limit=limit,
        ):
            if entry.id not in seen:
                matches.append(entry)
                seen.add(entry.id)
                if len(matches) >= limit:
                    break
        return matches

    def reindex_missing(self, *, limit: int | None = None) -> int:
        """Compute the selected model's vectors for entries not yet indexed by it."""
        if self._embedder is None:
            raise EmbeddingUnavailableError('Модель эмбеддингов не подключена')
        completed = 0
        while limit is None or completed < limit:
            batch_size = min(100, limit - completed) if limit is not None else 100
            missing = self._repository.list_unindexed(self._embedder.name, limit=batch_size)
            if not missing:
                break
            for entry_id, title in missing:
                vector = self._embedding(title)
                if vector is None:
                    raise EmbeddingUnavailableError('Индексация остановлена: модель не вернула вектор')
                self._repository.save_embedding(entry_id, self._embedder.name, vector)
                completed += 1
        return completed
