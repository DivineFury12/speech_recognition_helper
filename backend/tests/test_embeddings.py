import math
from datetime import datetime
from zoneinfo import ZoneInfo

from assistant.embeddings import EmbeddingIndex, create_embedding_model
from assistant.search import search_entries
from assistant.service import Assistant
from assistant.storage import Storage

TZ = "Europe/Moscow"
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=ZoneInfo(TZ))
TOPICS = [("договор", "соглашени", "контракт"), ("презентац", "слайд"), ("мама", "маме", "родител")]


def _vec(text: str) -> list[float]:
    t = text.lower()
    v = [1.0 if any(w in t for w in words) else 0.0 for words in TOPICS] + [0.3]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class FakeModel:
    name = "fake"
    dimensions = 4

    def __init__(self):
        self.documents = 0

    def embed_documents(self, texts):
        self.documents += len(texts)
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


def make(threshold=0.8):
    storage = Storage(":memory:")
    model = FakeModel()
    return storage, model, EmbeddingIndex(storage, model, threshold=threshold)


def test_semantic_search_finds_synonyms_and_skips_unrelated():
    storage, model, index = make()
    for title in ("Подготовить договор с подрядчиком", "Сделать презентацию", "Позвонить маме"):
        storage.create_entry(kind="task", title=title, date=None, time=None, timezone=TZ)
    pool = storage.all_entries()
    found = search_entries("соглашение с исполнителем", pool, index=index, user_id="local")
    assert [e["title"] for e in found] == ["Подготовить договор с подрядчиком"]
    assert search_entries("рецепт борща", pool, index=index, user_id="local") == []


def test_vectors_are_cached_and_reindexed_on_rename():
    storage, model, index = make()
    e = storage.create_entry(kind="task", title="Сделать презентацию", date=None, time=None, timezone=TZ)
    index.rank("слайды", storage.all_entries(), "local")
    index.rank("слайды", storage.all_entries(), "local")
    assert model.documents == 1
    storage.update_entry(e["id"], {"title": "Позвонить маме"})
    found = index.rank("родители", storage.all_entries(), "local")
    assert [x["title"] for _, x in found] == ["Позвонить маме"] and model.documents == 2


def test_assistant_uses_index_for_search_and_targets():
    storage, model, index = make()
    a = Assistant(storage, None, index)
    a.handle("Добавь задачу подготовить договор с подрядчиком", TZ, now=NOW)
    a.handle("Добавь задачу сделать презентацию", TZ, now=NOW)
    found = a.handle("Найди соглашение", TZ, now=NOW)
    assert [e["title"] for e in found["entries"]] == ["Подготовить договор с подрядчиком"]
    done = a.handle("Отметь контракт выполненным", TZ, now=NOW)
    assert done["intent"] == "complete_entry"
    assert done["entries"][0]["title"] == "Подготовить договор с подрядчиком"


def test_duplicate_command_does_not_create_second_entry():
    a = Assistant(Storage(":memory:"), None)
    first = a.handle("Добавь встречу с командой завтра в 15:00", TZ, now=NOW)
    again = a.handle("Добавь встречу с командой завтра в 15:00", TZ, now=NOW)
    assert again["status"] == "completed" and "уже есть" in again["assistant_text"]
    assert again["entries"][0]["id"] == first["entries"][0]["id"]
    assert len(a.storage.all_entries()) == 1


def test_create_embedding_model_none_and_unknown():
    assert create_embedding_model("none", ".") is None
    try:
        create_embedding_model("bert", ".")
        assert False
    except ValueError:
        pass


def test_embeddings_deleted_with_entry():
    storage, model, index = make()
    e = storage.create_entry(kind="task", title="Сделать презентацию", date=None, time=None, timezone=TZ)
    index.index([e])
    storage.delete_entry(e["id"])
    assert storage.get_embeddings("local", "fake") == {}
