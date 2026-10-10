"""Local CPU implementations of the text embedding contract."""

import threading

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from src.ports.embeddings import EmbeddingUnavailableError


E5_NAME = 'intfloat/multilingual-e5-small'
FRIDA_NAME = 'ai-forever/FRIDA'


class _TransformerEmbeddingModel:
    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        raise NotImplementedError

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode(self, text: str, prefix: str) -> list[float]:
        # The same instance may be called concurrently by FastAPI worker threads.
        with self._lock:
            try:
                if self._model is None:
                    self._load()
                tokens = self._tokenizer(
                    f'{prefix}: {text}', max_length=512, truncation=True,
                    padding=True, return_tensors='pt',
                )
                with torch.inference_mode():
                    hidden = self._model(**tokens).last_hidden_state
                    pooled = self._pool(hidden, tokens['attention_mask'])
                    vector = functional.normalize(pooled, p=2, dim=1)[0].tolist()
                if len(vector) != self.dimensions:
                    raise ValueError(f'{self.name} returned {len(vector)} dimensions')
                return vector
            except Exception as exc:
                raise EmbeddingUnavailableError(f'Не удалось получить вектор модели {self.name}') from exc


class E5EmbeddingModel(_TransformerEmbeddingModel):
    name = E5_NAME
    dimensions = 384

    def _load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, cache_dir=self._cache_dir)
        self._model = AutoModel.from_pretrained(self.name, cache_dir=self._cache_dir).to('cpu').eval()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1)
        return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1)

    def embed_document(self, text: str) -> list[float]:
        # Both sides use query: for the symmetric, short-title task. This also
        # preserves compatibility with E5 vectors stored before this interface.
        return self._encode(text, 'query')

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text, 'query')


class FridaEmbeddingModel(_TransformerEmbeddingModel):
    name = FRIDA_NAME
    dimensions = 1536

    def _load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, cache_dir=self._cache_dir)
        self._model = T5EncoderModel.from_pretrained(self.name, cache_dir=self._cache_dir).to('cpu').eval()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0]

    def embed_document(self, text: str) -> list[float]:
        return self._encode(text, 'search_document')

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text, 'search_query')


def create_embedding_model(name: str, cache_dir: str) -> E5EmbeddingModel | FridaEmbeddingModel:
    if name in ('e5', E5_NAME):
        return E5EmbeddingModel(cache_dir)
    if name in ('frida', FRIDA_NAME):
        return FridaEmbeddingModel(cache_dir)
    raise ValueError(f'Неизвестная модель эмбеддингов: {name}')
