import abc
import typing
import pathlib
import enum
import functools

import transformers
import torch
import torch.nn.functional as f


__all__ = [
    'get_best_text_gen_model',
    'MAX_NEW_TOKENS',
    'get_best_embedding_model',
    'EMBEDDING_LENGTH',
]

MAX_NEW_TOKENS: typing.Final[int] = 1_000
EMBEDDING_LENGTH: typing.Final[int] = 1_536

_CACHE_DIR: typing.Final[pathlib.Path] = (
    pathlib.Path(__file__).parent
    / '.cache'
)


@functools.lru_cache
def get_best_text_gen_model() -> '_BaseLLM':
    return _Qwen3_5_08B()


@functools.lru_cache
def get_best_embedding_model() -> '_EmbeddingModel':
    return _FridaEmbeddingModel()


class _HuggingFaceModel(abc.ABC):
    MODEL_ID: str
    PARAMETERS_BILLIONS: float | int


class _BaseLLM(_HuggingFaceModel):
    @abc.abstractmethod
    def get_response(
        self,
        prompt: str,
        context_blocks: list[str] | None = None,
    ) -> str: ...


class _BaseQwen3_5LLM(_BaseLLM):
    SYSTEM_PROMPT: str = (
        'Ты полезный ассистент. Отвечай точно и кратко.'
        'Не используй Markdown, только обычный текст.'
        ' Каждый вопрос начинай с новой строки и'
        ' нумеруй его: 1., 2., 3. и т. д.'
        ' Генерируй только столько вопросов, сколько указал'
        ' пользователь. Если пользователь не указал количество'
        ' вопросов, генерируй 3 вопроса. Кроме самих вопросов'
        ' ничего не пиши. Формулируй вопросы простым'
        ' языком и не делай их сложными.'
        ' Старайся давать разные вопросы.'
    )

    def __init__(self) -> None:
        super().__init__()
        self.device = 'cpu'
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.MODEL_ID,
            cache_dir=_CACHE_DIR,
        )
        self._model = transformers.AutoModelForMultimodalLM.from_pretrained(
            self.MODEL_ID,
            cache_dir=_CACHE_DIR,
        )

    @property
    @abc.abstractmethod
    def _apply_chat_template(self) -> typing.Callable[[dict], list]: ...

    def get_response(
        self,
        prompt: str,
        context_blocks: list[str] | None = None,
    ) -> str:
        messages = [
            {
                'role': 'system',
                'content': [
                    {'type': 'text', 'text': self.SYSTEM_PROMPT},
                ],
            },
        ]
        if context_blocks is not None:
            context = '\n\n'.join(context_blocks)
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': (
                                'Используй следующий контекст для ответа:\n\n'
                                f'{context}'
                            ),
                        },
                    ],
                },
            )

        messages.append(
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                ],
            },
        )

        inputs = self._apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        response = self._processor.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
        )
        response = (
            response.replace('<|im_end|>', '')
            .replace('<|endoftext|>', '')
        ).strip()
        return response


class _Qwen3_5_08B(_BaseQwen3_5LLM):
    MODEL_ID: str = 'Qwen/Qwen3.5-0.8B'
    PARAMETERS_BILLIONS: float = 0.8

    @property
    def _apply_chat_template(self) -> typing.Callable[[dict], list]:
        return self._processor.apply_chat_template


class _Qwen3_5_2B(_BaseQwen3_5LLM):
    MODEL_ID: str = 'Qwen/Qwen3.5-2B-Base'
    PARAMETERS_BILLIONS: int = 2

    @property
    def _apply_chat_template(self) -> typing.Callable[[dict], list]:
        return self._processor.tokenizer.apply_chat_template


class _Prefix(enum.Enum):
    SEARCH_QUERY = 'search_query'
    SEARCH_DOCUMENT = 'search_document'
    PARAPHRASE = 'paraphrase'
    CATEGORIZE = 'categorize'
    CATEGORIZE_SENTIMENT = 'categorize_sentiment'
    CATEGORIZE_TOPIC = 'categorize_topic'
    CATEGORIZE_ENTAILMENT = 'categorize_entailment'


class _EmbeddingModel(_HuggingFaceModel):
    @functools.lru_cache
    @abc.abstractmethod
    def get_vector(self, text: str) -> list[float]: ...

    def multi_call(self, texts: list[str]) -> list[list[float]]:
        return [self.get_vector(now_text) for now_text in texts]


class _FridaEmbeddingModel(_EmbeddingModel):
    MODEL_ID: str = 'ai-forever/FRIDA'

    def __init__(self: typing.Self) -> None:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            cache_dir=_CACHE_DIR,
        )
        self._model = transformers.T5EncoderModel.from_pretrained(
            self.MODEL_ID,
            cache_dir=_CACHE_DIR,
        )

    def multi_call(
        self,
        texts: list[str],
        prefix: _Prefix = _Prefix.SEARCH_QUERY,
    ) -> list[list[float]]:
        texts = [f'{prefix.value}: {now_text}' for now_text in texts]
        tokenized_inputs = self._tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            outputs = self._model(**tokenized_inputs)

        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = f.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    @functools.lru_cache
    def get_vector(self, text: str, prefix: _Prefix = _Prefix.SEARCH_QUERY) -> list[float]:
        return self.multi_call([text], prefix=prefix)[0]

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> list[str]:
        """Token-aware split, leaving headroom under the 512-token limit for the prefix."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        step = chunk_size - overlap
        chunks = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + chunk_size]
            chunks.append(self._tokenizer.decode(window, skip_special_tokens=True))
            if start + chunk_size >= len(token_ids):
                break
        return chunks


if __name__ == "__main__":
    import time

    prompt = 'Сделай 3 вопросов по теме геометрии для школьников 7-8 класса'

    BEST_MODEL = get_best_text_gen_model()
    old_time = time.time()
    response = BEST_MODEL.get_response(prompt)
    work_time = time.time() - old_time
    report = (
        f'Prompt: {prompt}'
        f'\nModel id: {BEST_MODEL.MODEL_ID}'
        f'\nResponse: {response}'
        f'\nWork time: {work_time:.2f} s.'
    )
    print(report)

    EMBEDDING_MODEL = get_best_embedding_model()
    old_time = time.time()
    response = EMBEDDING_MODEL.get_vector(prompt)
    work_time = time.time() - old_time
    report = (
        f'Prompt: {prompt}'
        f'\nModel id: {EMBEDDING_MODEL.MODEL_ID}'
        f'\nLen response: {len(response)}'
        f'\nWork time: {work_time:.2f} s.'
    )
    print(report)
