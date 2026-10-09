from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...
