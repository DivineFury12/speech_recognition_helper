"""Entry operations, including optional semantic indexing and retrieval."""

import datetime
import logging
import math
import re
import uuid
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.ports.embeddings import EmbeddingModel, EmbeddingUnavailableError
from src.ports.entries import EntryRepository
from src.schemas.assistant import Entry, EntryKind, EntryStatus, EntryUpdate, Recurrence
from src.services.recurrence import occurrences
from src.services.topics import classify_entry_topics


logger = logging.getLogger(__name__)
DEFAULT_MEETING_DURATION_MINUTES = 60
# Scores were checked against short Russian title/query pairs for each model.
# Prefer missing a weak semantic match to suggesting an unrelated record.
SEMANTIC_MIN_SIMILARITY = {
    'intfloat/multilingual-e5-small': 0.84,
    'ai-forever/FRIDA': 0.30,
}
GENERIC_SEARCH_WORDS = {
    'задача', 'задачу', 'задачи', 'встреча', 'встречу', 'встречи',
    'напоминание', 'напоминания', 'запись', 'записи', 'записку',
    'удали', 'удалить', 'перенеси', 'перенести', 'покажи', 'найди',
    'про', 'для', 'мне', 'мою', 'мой', 'это', 'эту', 'этот',
}


class EntryService:
    def __init__(self, repository: EntryRepository, embedder: EmbeddingModel | None = None,
                 topic_client=None):
        self._repository = repository
        self._embedder = embedder
        self._topic_client = topic_client

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

    @staticmethod
    def _duration_for_create(kind: EntryKind, duration_minutes: int | None) -> int | None:
        if kind != EntryKind.meeting:
            if duration_minutes is not None:
                raise ValueError('Длительность можно задать только для встречи')
            return None
        if duration_minutes is None:
            return DEFAULT_MEETING_DURATION_MINUTES
        if not 1 <= duration_minutes <= 1440:
            raise ValueError('Длительность встречи должна быть от 1 до 1440 минут')
        return duration_minutes

    def get(self, entry_id: uuid.UUID) -> Entry | None:
        return self._repository.get(entry_id)

    def all_entries(self) -> list[Entry]:
        return self._repository.all_entries()

    def list_undated(self) -> list[Entry]:
        return self._repository.list_undated()

    def restore(self, entry: Entry) -> Entry:
        return self._repository.restore(entry)

    def update(self, entry_id: uuid.UUID, patch: EntryUpdate) -> Entry | None:
        existing = self.get(entry_id)
        if existing is None:
            return None
        changes = patch.model_dump(exclude_unset=True)
        if 'timezone' in changes:
            try:
                ZoneInfo(changes['timezone'])
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError('Укажите корректный часовой пояс IANA') from exc
        date = changes.get('date', existing.date)
        time = changes.get('time', existing.time)
        if time is not None and date is None:
            raise ValueError('Время требует даты')
        if changes.get('recurrence', existing.recurrence) is not None and date is None:
            raise ValueError('Для повторяющейся записи нужна дата')
        if existing.kind in (EntryKind.meeting, EntryKind.reminder) and (date is None or time is None):
            raise ValueError('Для встречи или напоминания нужны дата и время')
        if 'duration_minutes' in changes and existing.kind != EntryKind.meeting:
            raise ValueError('Длительность можно задать только для встречи')
        if existing.kind == EntryKind.meeting:
            if 'duration_minutes' in changes and changes['duration_minutes'] is None:
                raise ValueError('Длительность встречи не может быть пустой')
            if existing.duration_minutes is None and 'duration_minutes' not in changes:
                changes['duration_minutes'] = DEFAULT_MEETING_DURATION_MINUTES
        if 'title' in changes:
            changes['title'] = changes['title'].strip()
            changes['topics'] = classify_entry_topics(changes['title'], self._topic_client)
        embedding = self._embedding(changes['title']) if 'title' in changes else None
        return self._repository.update(
            entry_id, changes, embedding=embedding,
            embedding_model=self._embedder.name if embedding is not None else None,
        )

    def delete(self, entry_id: uuid.UUID) -> bool:
        return self._repository.delete(entry_id)

    def find_candidates(
        self, query: str | None, *, kind: EntryKind | None = None,
        date: datetime.date | None = None, time: str | None = None, limit: int = 10,
    ) -> tuple[list[Entry], bool]:
        """Exact text matches are safe to select; semantic matches need clarification."""
        if query:
            matches, exact = self._text_candidates(query, kind=kind, limit=limit)
            if not matches:
                matches = self._semantic_candidates(query, kind=kind, limit=limit)
        elif date is not None:
            matches = self._repository.list_period(date, date, kind=kind)
            exact = True
        else:
            return [], False
        matches = [entry for entry in matches if
                   (date is None or entry.date == date or (
                       entry.recurrence and entry.date is not None
                       and occurrences(entry.date, entry.recurrence, date, date, entry.recurrence_interval or 1)
                       and not (entry.recurrence_until and date > entry.recurrence_until))) and
                   (time is None or entry.time == time)]
        if query and not exact and matches:
            # «Перенеси распитие напитков»: the same words as the title are not a guess.
            wanted = self._search_stems(re.sub(r'^(?:задач\w*|встреч\w*|напоминани\w*|запис\w*)\s+', '',
                                               query.strip(), flags=re.IGNORECASE))
            same = [entry for entry in matches if wanted and self._search_stems(entry.title) == wanted]
            if len(same) == 1:
                return same, True
        return matches[:limit], exact

    @staticmethod
    def _search_stems(text: str) -> set[str]:
        words = re.findall(r'[а-яёa-z]{4,}', text.casefold())
        return {
            word.replace('ё', 'е')[:4] for word in words
            if word not in GENERIC_SEARCH_WORDS
        }

    def _text_candidates(
        self, query: str, *, kind: EntryKind | None, limit: int,
        candidate_ids: set[uuid.UUID] | None = None,
    ) -> tuple[list[Entry], bool]:
        normalized = ' '.join(query.split())
        variants = [normalized]
        shorter = re.sub(
            r'^(?:задач\w*|встреч\w*|напоминани\w*|запис\w*)\s+',
            '', normalized, flags=re.IGNORECASE,
        )
        shorter = re.sub(r'^(?:про|о)\s+', '', shorter, flags=re.IGNORECASE)
        if shorter and shorter != normalized:
            variants.append(shorter)
        stems = self._search_stems(shorter)
        if not stems:
            return [], False
        for variant in variants:
            exact = self._repository.search_text(
                variant, kind=kind, limit=limit, candidate_ids=candidate_ids,
            )
            if exact:
                return exact, True

        candidates: dict[uuid.UUID, Entry] = {}
        for stem in stems:
            for entry in self._repository.search_text(
                stem, kind=kind, limit=max(25, limit * 5), candidate_ids=candidate_ids,
            ):
                title_stems = self._search_stems(entry.title)
                if stem in title_stems:
                    candidates[entry.id] = entry
        ranked = sorted(
            candidates.values(),
            key=lambda entry: (-len(stems & self._search_stems(entry.title)),
                               -entry.created_at.timestamp()),
        )
        return ranked[:limit], False

    def _semantic_candidates(
        self, query: str, *, kind: EntryKind | None, limit: int,
        candidate_ids: set[uuid.UUID] | None = None,
    ) -> list[Entry]:
        if not self._search_stems(query):
            return []
        vector = self._embedding(query, query=True)
        if vector is None:
            return []
        minimum = SEMANTIC_MIN_SIMILARITY.get(self._embedder.name, 0.86)
        return self._repository.search_similar(
            vector, embedding_model=self._embedder.name, kind=kind,
            limit=limit, min_similarity=minimum,
            candidate_ids=candidate_ids,
        )

    def create(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str, duration_minutes: int | None = None,
        recurrence: Recurrence | None = None, **extra,
    ) -> Entry:
        duration_minutes = self._duration_for_create(kind, duration_minutes)
        if recurrence is not None and date is None:
            raise ValueError('Для повторяющейся записи нужна дата')
        embedding = self._embedding(title)
        kwargs = {'recurrence': recurrence} if recurrence is not None else {}
        kwargs.update({k: v for k, v in extra.items() if v is not None})
        kwargs['topics'] = classify_entry_topics(title, self._topic_client)
        return self._repository.create(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            embedding=embedding, duration_minutes=duration_minutes,
            embedding_model=self._embedder.name if embedding is not None else None,
            **kwargs,
        )

    def create_once(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str, duration_minutes: int | None = None,
        recurrence: Recurrence | None = None, **extra,
    ) -> tuple[Entry, bool]:
        """Return the matching pending entry when the same command is repeated."""
        title = ' '.join(title.split())
        duration_minutes = self._duration_for_create(kind, duration_minutes)
        embedding = self._embedding(title)
        kwargs = {'recurrence': recurrence} if recurrence is not None else {}
        kwargs.update({k: v for k, v in extra.items() if v is not None})
        kwargs['topics'] = classify_entry_topics(title, self._topic_client)
        return self._repository.create_once(
            kind=kind, title=title, date=date, time=time, timezone=timezone,
            embedding=embedding, duration_minutes=duration_minutes,
            embedding_model=self._embedder.name if embedding is not None else None,
            **kwargs,
        )

    def list_period(
        self, date_from: datetime.date, date_to: datetime.date, *,
        kind: EntryKind | None = None, status: EntryStatus | None = None,
    ) -> list[Entry]:
        return self._repository.list_period(date_from, date_to, kind=kind, status=status)

    def search(
        self, query: str, *, kind: EntryKind | None = None, limit: int = 5,
        date_from: datetime.date | None = None, date_to: datetime.date | None = None,
    ) -> list[Entry]:
        query = query.strip()
        if not query or limit <= 0:
            return []
        if date_from is not None and date_to is not None:
            eligible = self.list_period(date_from, date_to, kind=kind)
            candidate_ids = {entry.id for entry in eligible}
            if not candidate_ids:
                return []
            matches, _ = self._text_candidates(
                query, kind=kind, limit=len(candidate_ids), candidate_ids=candidate_ids,
            )
            if not matches:
                matches = self._semantic_candidates(
                    query, kind=kind, limit=len(candidate_ids), candidate_ids=candidate_ids,
                )
            matched_ids = {entry.id for entry in matches}
            return [entry for entry in eligible if entry.id in matched_ids]
        matches, _ = self._text_candidates(query, kind=kind, limit=limit)
        return matches if matches else self._semantic_candidates(query, kind=kind, limit=limit)

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
