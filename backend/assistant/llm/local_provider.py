import threading

from .base import LLMError


class LocalProvider:
    name = "local"

    def __init__(self) -> None:
        try:
            import ml
        except ImportError as e:
            raise LLMError(f"Локальная модель недоступна: {e}") from e
        self._model = ml.get_best_text_gen_model()
        self.model = self._model.MODEL_ID
        self._lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        with self._lock:
            return self._model.get_response(user, system_prompt=system)
