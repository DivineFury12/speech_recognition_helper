import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.llm import LLMError
from assistant.nlu import parse_rules
from assistant.service import Assistant
from assistant.storage import Storage

TZ = "Europe/Moscow"
NOW = datetime(2026, 10, 9, 12, 0, tzinfo=ZoneInfo(TZ))


@pytest.fixture()
def assistant():
    storage = Storage(":memory:")
    yield Assistant(storage, llm=None)
    storage.close()


def say(a: Assistant, text: str, cid: str | None = None):
    return a.handle(text, TZ, cid, now=NOW)


def test_reminder_created_in_one_go(assistant):
    turn = say(assistant, "Напомни мне завтра в 10 утра позвонить Ивану")
    assert turn["status"] == "completed"
    assert turn["intent"] == "create_entry"
    [entry] = turn["entries"]
    assert entry["kind"] == "reminder"
    assert entry["title"] == "Позвонить Ивану"
    assert (entry["date"], entry["time"]) == ("2026-10-10", "10:00")
    assert entry["timezone"] == TZ


def test_meeting_needs_time_then_completes(assistant):
    first = say(assistant, "Поставь встречу с Иваном завтра")
    assert first["status"] == "needs_clarification"
    assert first["entries"] == []
    assert "время" in first["assistant_text"]

    second = say(assistant, "В 15:00", first["conversation_id"])
    assert second["status"] == "completed"
    assert second["conversation_id"] == first["conversation_id"]
    [entry] = second["entries"]
    assert entry["kind"] == "meeting"
    assert entry["title"] == "Встреча с Иваном"
    assert (entry["date"], entry["time"]) == ("2026-10-10", "15:00")
    assert entry["duration_minutes"] == 60


def test_clarification_can_be_cancelled(assistant):
    first = say(assistant, "Поставь встречу с Иваном")
    assert first["status"] == "needs_clarification"
    second = say(assistant, "отмена", first["conversation_id"])
    assert second["status"] == "no_action"
    third = say(assistant, "В 15:00", first["conversation_id"])
    assert third["entries"] == [] or third["entries"][0]["kind"] == "task"


def test_new_command_during_clarification_drops_draft(assistant):
    first = say(assistant, "Поставь встречу с Иваном завтра")
    second = say(assistant, "Напомни послезавтра в 9 купить молоко", first["conversation_id"])
    assert second["status"] == "completed"
    assert second["entries"][0]["title"] == "Купить молоко"


def test_reminder_without_title_asks_what(assistant):
    first = say(assistant, "Напомни завтра в 10")
    assert first["status"] == "needs_clarification"
    second = say(assistant, "позвонить маме", first["conversation_id"])
    assert second["status"] == "completed"
    assert second["entries"][0]["title"] == "Позвонить маме"


def test_task_without_date(assistant):
    turn = say(assistant, "Добавь задачу подготовить презентацию")
    assert turn["status"] == "completed"
    entry = turn["entries"][0]
    assert entry["kind"] == "task"
    assert entry["title"] == "Подготовить презентацию"
    assert entry["date"] is None


def test_time_only_meeting_goes_today_or_tomorrow(assistant):
    later = say(assistant, "Созвон с командой в 18:00")
    assert later["entries"][0]["date"] == "2026-10-09"
    earlier = say(assistant, "Созвон с командой в 9:00")
    assert earlier["entries"][0]["date"] == "2026-10-10"


def test_list_and_search(assistant):
    say(assistant, "Напомни сегодня в 18:00 оплатить интернет")
    say(assistant, "Поставь встречу с Петром завтра в 11")
    say(assistant, "Добавь задачу подготовить презентацию для клиента")

    today = say(assistant, "Что у меня сегодня?")
    assert today["intent"] == "list_entries"
    assert [e["title"] for e in today["entries"]] == ["Оплатить интернет"]

    week = say(assistant, "Что у меня на неделе?")
    assert len(week["entries"]) == 2

    found = say(assistant, "Найди задачу про презентацию")
    assert found["intent"] == "search_entries"
    assert [e["title"] for e in found["entries"]] == ["Подготовить презентацию для клиента"]

    nothing = say(assistant, "Найди отчёт по налогам")
    assert nothing["entries"] == []


def test_unknown(assistant):
    turn = say(assistant, "Привет, как дела?")
    assert turn["status"] == "no_action"
    assert turn["intent"] == "unknown"


def test_messages_are_stored(assistant):
    first = say(assistant, "Поставь встречу с Иваном завтра")
    say(assistant, "В 15:00", first["conversation_id"])
    msgs = assistant.storage.list_messages(first["conversation_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_llm_json_is_used():
    reply = "```json\n" + json.dumps({
        "intent": "create_entry", "kind": "meeting", "title": "Встреча с Анной",
        "date_phrase": "в понедельник", "time_phrase": "в полдень",
        "date": "2026-10-12", "time": "12:00", "duration_minutes": 30,
        "period_phrase": None, "query": None,
    }, ensure_ascii=False) + "\n```"
    llm = FakeLLM(reply)
    a = Assistant(Storage(":memory:"), llm)
    turn = a.handle("созвонимся с Аней в понедельник в обед на полчаса", TZ, now=NOW)
    entry = turn["entries"][0]
    assert entry["title"] == "Встреча с Анной"
    assert (entry["date"], entry["time"], entry["duration_minutes"]) == ("2026-10-12", "12:00", 30)
    assert "2026-10-09" in llm.calls[0]


def test_llm_iso_fallback_when_phrase_unparsed():
    reply = json.dumps({"intent": "create_entry", "kind": "reminder", "title": "Сдать отчёт",
                        "date_phrase": "в конце месяца", "time_phrase": "к обеду",
                        "date": "2026-10-30", "time": "13:00"})
    a = Assistant(Storage(":memory:"), FakeLLM(reply))
    entry = a.handle("напомни сдать отчёт в конце месяца к обеду", TZ, now=NOW)["entries"][0]
    assert (entry["date"], entry["time"]) == ("2026-10-30", "13:00")


@pytest.mark.parametrize("reply", [LLMError("сеть"), "не JSON вовсе", '{"intent": "dance"}'])
def test_llm_failure_falls_back_to_rules(reply):
    a = Assistant(Storage(":memory:"), FakeLLM(reply))
    turn = a.handle("Напомни завтра в 10 утра позвонить Ивану", TZ, now=NOW)
    assert turn["status"] == "completed"
    assert turn["entries"][0]["title"] == "Позвонить Ивану"


def test_rules_intents():
    today = NOW.date()
    assert parse_rules("Что у меня завтра?", today).intent == "list_entries"
    assert parse_rules("Найди встречу про бюджет", today).intent == "search_entries"
    assert parse_rules("Найди встречу про бюджет", today).query == "бюджет"
    assert parse_rules("Напомни, что у меня завтра в 10 врач", today).intent == "create_entry"
