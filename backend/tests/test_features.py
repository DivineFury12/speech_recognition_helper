import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.service import Assistant
from assistant.storage import Storage

TZ = "Europe/Moscow"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=ZoneInfo(TZ))


@pytest.fixture()
def a():
    storage = Storage(":memory:")
    yield Assistant(storage, llm=None)
    storage.close()


class Chat:
    def __init__(self, assistant: Assistant, user_id: str = "local"):
        self.a = assistant
        self.cid = None
        self.user_id = user_id

    def __call__(self, text: str) -> dict:
        turn = self.a.handle(text, TZ, self.cid, now=NOW, user_id=self.user_id)
        self.cid = turn["conversation_id"]
        return turn


def test_until_end_of_week_has_date(a):
    entry = Chat(a)("Поставь задачу купить билеты до конца недели")["entries"][0]
    assert (entry["title"], entry["date"]) == ("Купить билеты", "2026-10-11")


def test_list_filters_by_kind(a):
    say = Chat(a)
    say("Напомни завтра в 10 утра позвонить Ивану")
    say("Добавь встречу с командой завтра в 15:00")
    say("Добавь задачу отправить отчёт завтра")
    tasks = say("Какие задачи у меня на завтра?")
    assert [e["kind"] for e in tasks["entries"]] == ["task"]
    assert "1 задача" in tasks["assistant_text"]
    meetings = say("Есть ли у меня встречи завтра?")
    assert [e["title"] for e in meetings["entries"]] == ["Встреча с командой"]
    none = say("Покажи мои встречи на сегодня")
    assert none["entries"] == [] and "встреч нет" in none["assistant_text"]


def test_when_was_meeting_with_ivan(a):
    say = Chat(a)
    say("Поставь встречу с Иваном в пятницу в 15:00")
    found = say("Когда у меня была встреча с Иваном?")
    assert found["intent"] == "search_entries"
    assert [e["title"] for e in found["entries"]] == ["Встреча с Иваном"]
    assert "16 октября" in found["assistant_text"]


def test_context_move_it_to_tomorrow(a):
    say = Chat(a)
    say("Поставь встречу с дизайнером сегодня в 15:00")
    moved = say("А перенеси её на завтра")
    assert moved["intent"] == "update_entry" and moved["status"] == "completed"
    entry = moved["entries"][0]
    assert (entry["date"], entry["time"]) == ("2026-10-11", "15:00")


def test_move_by_target_and_rename(a):
    say = Chat(a)
    say("Поставь встречу с дизайнером в понедельник в 11")
    say("Добавь задачу купить хлеб")
    moved = Chat(a)("Перенеси встречу с дизайнером на 17:00")
    assert moved["entries"][0]["time"] == "17:00"
    assert moved["entries"][0]["date"] == "2026-10-12"
    renamed = Chat(a)("Переименуй задачу купить хлеб в купить батон")
    assert renamed["entries"][0]["title"] == "Купить батон"


def test_delete_needs_confirmation(a):
    say = Chat(a)
    say("Напомни завтра в 9 выпить таблетку")
    ask = say("Удали напоминание про таблетку")
    assert ask["status"] == "needs_clarification" and "Удалить" in ask["assistant_text"]
    assert say("нет")["status"] == "no_action"
    assert len(a.storage.all_entries()) == 1
    say("Удали напоминание про таблетку")
    done = say("да")
    assert done["intent"] == "delete_entry" and done["status"] == "completed"
    assert a.storage.all_entries() == []


def test_complete_by_target_and_by_context(a):
    say = Chat(a)
    say("Добавь задачу сделать презентацию")
    done = say("Отметь задачу про презентацию выполненной")
    assert done["intent"] == "complete_entry"
    assert done["entries"][0]["status"] == "completed"
    say("Добавь задачу полить цветы")
    ctx_done = say("готово")
    assert ctx_done["entries"][0]["title"] == "Полить цветы"
    assert ctx_done["entries"][0]["status"] == "completed"


def test_choose_between_several(a):
    say = Chat(a)
    say("Встреча с Иваном завтра в 10")
    say("Встреча с Петром завтра в 12")
    ask = Chat(a)
    question = ask("Удали встречу")
    assert question["status"] == "needs_clarification"
    assert "1." in question["assistant_text"] and "2." in question["assistant_text"]
    picked = ask("вторую")
    assert picked["intent"] == "delete_entry" and picked["status"] == "completed"
    assert len(a.storage.all_entries()) == 1


def test_meeting_conflict_confirm_or_new_time(a):
    say = Chat(a)
    say("Встреча с командой завтра в 15:00")
    clash = say("Поставь встречу с Иваном завтра в 15:30")
    assert clash["status"] == "needs_clarification" and "уже есть" in clash["assistant_text"]
    moved = say("тогда в 17:00")
    assert moved["status"] == "completed" and moved["entries"][0]["time"] == "17:00"
    clash2 = say("Поставь встречу с Петром завтра в 15:00")
    assert clash2["status"] == "needs_clarification"
    forced = say("да, всё равно")
    assert forced["status"] == "completed" and forced["entries"][0]["time"] == "15:00"


def test_recurring_meeting_expands_in_lists(a):
    say = Chat(a)
    created = say("Планёрка каждый понедельник в 10")
    entry = created["entries"][0]
    assert (entry["kind"], entry["recurrence"], entry["date"], entry["time"]) == ("meeting", "weekly", "2026-10-12", "10:00")
    week = a.storage.list_entries("2026-10-10", "2026-10-25")
    assert [e["date"] for e in week] == ["2026-10-12", "2026-10-19"]
    daily = say("Напоминай каждый день в 9 утра пить воду")
    assert daily["entries"][0]["recurrence"] == "daily"


class SemanticLLM:
    name = "fake"

    def complete(self, system, user):
        if "Записи:" in user:
            lines = [l for l in user.splitlines() if l[:1].isdigit()]
            n = next(i for i, l in enumerate(lines, 1) if "договор" in l)
            return json.dumps({"matches": [n]})
        raise RuntimeError("only search")


def test_semantic_search_via_llm():
    storage = Storage(":memory:")
    storage.create_entry(kind="task", title="Подготовить договор с подрядчиком", date=None, time=None, timezone=TZ)
    storage.create_entry(kind="task", title="Купить хлеб", date=None, time=None, timezone=TZ)
    from assistant.search import search_entries
    found = search_entries("соглашение с исполнителем", storage.all_entries(), llm=SemanticLLM())
    assert [e["title"] for e in found] == ["Подготовить договор с подрядчиком"]


def test_users_are_isolated(a):
    Chat(a, "u1")("Добавь задачу секрет первого")
    other = Chat(a, "u2")("Найди задачу секрет")
    assert other["entries"] == []
    assert len(a.storage.all_entries(user_id="u1")) == 1


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE entries (id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, date TEXT, time TEXT,
            timezone TEXT NOT NULL, duration_minutes INTEGER, status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL, updated_at TEXT);
        CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
        INSERT INTO entries VALUES ('e1', 'task', 'Старая задача', '2026-10-10', NULL, 'Europe/Moscow', NULL,
            'pending', '2026-10-01T00:00:00Z', NULL);
    """)
    db.commit()
    db.close()
    storage = Storage(str(path))
    [entry] = storage.list_entries("2026-10-10", "2026-10-10")
    assert entry["title"] == "Старая задача" and entry["recurrence"] is None
    storage.close()


def test_conflict_reply_with_yes_word_and_new_time(a):
    say = Chat(a)
    say("Встреча с командой завтра в 15:00")
    say("Поставь встречу с Иваном завтра в 15:30")
    moved = say("давай в 17")
    assert moved["entries"][0]["time"] == "17:00"


def test_delete_question_uses_accusative(a):
    say = Chat(a)
    say("Встреча с командой завтра в 15:00")
    assert say("Удали встречу с командой")["assistant_text"].startswith("Удалить встречу «")


def test_rename_capitalized_and_single_mention(a):
    say = Chat(a)
    say("Добавь задачу купить хлеб")
    turn = say("Переименуй её в купить батон")
    assert turn["entries"][0]["title"] == "Купить батон"
    assert turn["assistant_text"].count("Купить батон") == 1


def test_recurring_line_shows_occurrence_date(a):
    say = Chat(a)
    say("Планёрка каждый понедельник в 10")
    text = say("Что у меня на следующей неделе?")["assistant_text"]
    assert "пн, 12 октября в 10:00 — Планёрка (встреча, каждую неделю)" in text


def test_short_time_answers_in_clarification(a):
    for answer, expected in [("на 16", "16:00"), ("16 00", "16:00"), ("в 4 дня", "16:00"), ("16.30", "16:30")]:
        say = Chat(a)
        say("Поставь встречу с Килкой завтра")
        turn = say(answer)
        assert turn["status"] == "completed", answer
        assert turn["entries"][0]["time"] == expected, answer
        a.storage.delete_entry(turn["entries"][0]["id"])


def test_negative_answer_cancels_draft(a):
    say = Chat(a)
    say("Поставь встречу с Петром завтра")
    turn = say("не ставим встречу")
    assert turn["status"] == "no_action" and "отменил" in turn["assistant_text"]
    assert a.storage.all_entries() == []


def test_llm_time_without_phrase_accepted_when_hour_is_said():
    reply = '{"intent":"create_entry","kind":"meeting","title":"Встреча с Килкой","date":"2026-10-11","time":"16:00"}'
    storage = Storage(":memory:")
    Assistant(storage, None).handle("Поставь встречу с Килкой завтра", TZ, now=NOW)
    cid = storage._fetchone("SELECT id FROM conversations")["id"]

    class LLM:
        name = "fake"

        def complete(self, system, user):
            return reply

    turn = Assistant(storage, LLM()).handle("шестнадцать", TZ, cid, now=NOW)
    assert turn["entries"][0]["time"] == "16:00"
    guessed = Assistant(storage, LLM()).handle("Поставь встречу с Олегом завтра", TZ, now=NOW)
    assert guessed["status"] == "needs_clarification"
