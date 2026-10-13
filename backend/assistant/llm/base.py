from typing import Protocol


import time


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class TrackedProvider:
    """Обёртка, которая запоминает, когда модель последний раз отвечала и падала."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.last_ok: float | None = None
        self.last_error: float | None = None

    def complete(self, system: str, user: str) -> str:
        try:
            result = self.inner.complete(system, user)
        except LLMError:
            self.last_error = time.time()
            raise
        self.last_ok = time.time()
        return result

    @property
    def healthy(self) -> bool:
        return self.last_error is None or (self.last_ok or 0) > self.last_error

    def close(self) -> None:
        if hasattr(self.inner, "close"):
            self.inner.close()
