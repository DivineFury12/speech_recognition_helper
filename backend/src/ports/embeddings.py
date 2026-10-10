"""Contract for interchangeable text embedding models."""

from typing import Protocol


class EmbeddingUnavailableError(Exception):
    """The selected local model could not produce a vector."""


class EmbeddingModel(Protocol):
    name: str
    dimensions: int

    def embed_document(self, text: str) -> list[float]: ...

    def embed_query(self, text: str) -> list[float]: ...
