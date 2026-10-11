"""Storage contract used by the assistant and direct entry routes."""

import datetime
import typing
import uuid

from src.schemas.assistant import Entry, EntryKind, EntryStatus


class EntryRepository(typing.Protocol):
    def create(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str, embedding: list[float] | None,
        embedding_model: str | None,
    ) -> Entry: ...

    def create_once(
        self, *, kind: EntryKind, title: str, date: datetime.date | None,
        time: str | None, timezone: str, embedding: list[float] | None,
        embedding_model: str | None,
    ) -> tuple[Entry, bool]: ...

    def list_period(
        self, date_from: datetime.date, date_to: datetime.date, *,
        kind: EntryKind | None = None, status: EntryStatus | None = None,
    ) -> list[Entry]: ...

    def search_text(
        self, query: str, *, kind: EntryKind | None = None, limit: int = 5,
    ) -> list[Entry]: ...

    def search_similar(
        self, embedding: list[float], *, embedding_model: str,
        kind: EntryKind | None = None, limit: int = 5,
    ) -> list[Entry]: ...

    def list_unindexed(self, embedding_model: str, *, limit: int = 100) -> list[tuple[uuid.UUID, str]]: ...

    def save_embedding(self, entry_id: uuid.UUID, embedding_model: str, embedding: list[float]) -> None: ...
