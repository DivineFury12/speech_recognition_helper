from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from assistant.service import Assistant
from assistant.storage import Storage

TZ = "Europe/Moscow"
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=ZoneInfo(TZ))


class Chat:
    def __init__(self, assistant):
        self.a = assistant
        self.cid = None

    def __call__(self, text):
        turn = self.a.handle(text, TZ, self.cid, now=NOW)
        self.cid = turn["conversation_id"]
        return turn


@pytest.fixture()
def say():
    storage = Storage(":memory:")
    yield Chat(Assistant(storage, None))
    storage.close()


def test_done_report_completes_existing_task(say):
    say("Добавь задачу купить молоко")
    turn = say("Я купил молоко")
    assert turn["intent"] == "complete_entry"
    assert turn["entries"][0]["title"] == "Купить молоко" and turn["entries"][0]["status"] == "completed"
    assert len(say.a.storage.all_entries()) == 1


def test_unrelated_past_tense_is_not_completion(say):
    say("Добавь задачу купить молоко")
    turn = say("Я позвонил врачу")
    assert turn["intent"] != "complete_entry"


@pytest.mark.parametrize("command, expected", [
    ("Сдвинь её на час позже", ("2026-10-15", "12:00")),
    ("Перенеси её на полчаса раньше", ("2026-10-15", "10:30")),
    ("Перенеси её на 3 дня", ("2026-10-18", "11:00")),
    ("Перенеси её на 17", ("2026-10-15", "17:00")),
])
def test_relative_and_loose_moves(say, command, expected):
    say("Добавь встречу с дизайнером в четверг в 11")
    entry = say(command)["entries"][0]
    assert (entry["date"], entry["time"]) == expected


def test_change_duration_by_voice(say):
    say("Добавь встречу с дизайнером в четверг в 11")
    assert say("Сделай её на полчаса")["entries"][0]["duration_minutes"] == 30


def test_move_without_details_asks_and_applies_answer(say):
    say("Добавь встречу с дизайнером в четверг в 11")
    ask = say("Перенеси встречу")
    assert ask["status"] == "needs_clarification" and "Что изменить" in ask["assistant_text"]
    done = say("на пятницу в 15:00")
    assert done["intent"] == "update_entry"
    assert (done["entries"][0]["date"], done["entries"][0]["time"]) == ("2026-10-16", "15:00")


def test_action_without_context_asks_which_entry(say):
    say("Добавь встречу с Иваном завтра в 10")
    fresh = Chat(say.a)
    ask = fresh("Отметь выполненной")
    assert ask["status"] == "needs_clarification" and ask["assistant_text"] == "Какую запись отметить выполненной?"
    done = fresh("встречу с Иваном")
    assert done["intent"] == "complete_entry" and done["entries"][0]["title"] == "Встреча с Иваном"


def test_target_question_then_unknown_name(say):
    fresh = Chat(say.a)
    fresh("Удали её")
    again = fresh("встречу с Марсом")
    assert again["status"] == "needs_clarification" and "Не нашёл" in again["assistant_text"]
    assert fresh("отмена")["status"] == "no_action"


def test_generic_noun_with_several_entries_offers_choice(say):
    say("Встреча с Иваном завтра в 10")
    say("Встреча с Петром завтра в 12")
    ask = say("Удали встречу")
    assert ask["status"] == "needs_clarification" and "1." in ask["assistant_text"]
    assert say("вторую")["assistant_text"] == "Удалил встречу «Встреча с Петром»."


def test_choose_then_ask_what_to_change(say):
    say("Встреча с Иваном завтра в 10")
    say("Встреча с Петром завтра в 12")
    say("Перенеси встречу")
    ask = say("первую")
    assert "Что изменить у встречи «Встреча с Иваном»" in ask["assistant_text"]
    assert say("на час позже")["entries"][0]["time"] == "11:00"


def test_complete_recurring_moves_to_next_occurrence(say):
    say("Планёрка каждый понедельник в 10")
    turn = say("Отметь планёрку выполненной")
    entry = turn["entries"][0]
    assert entry["status"] == "pending" and entry["recurrence"] == "weekly"
    assert entry["date"] == "2026-10-26"
    assert "Следующая" in turn["assistant_text"]


def test_move_recurring_series_time(say):
    say("Планёрка каждый понедельник в 10")
    entry = say("Перенеси планёрку на 11")["entries"][0]
    assert (entry["time"], entry["recurrence"]) == ("11:00", "weekly")


def test_conflict_with_recurring_meeting(say):
    say("Планёрка каждый понедельник в 10")
    clash = say("Поставь встречу с клиентом в понедельник в 10:30")
    assert clash["status"] == "needs_clarification" and "Планёрка" in clash["assistant_text"]
    assert say("давай в 12")["entries"][0]["time"] == "12:00"


def test_conflict_when_moving_meeting(say):
    say("Встреча с Иваном завтра в 10")
    say("Встреча с Петром завтра в 12")
    clash = say("Перенеси встречу с Петром на 10:30")
    assert clash["status"] == "needs_clarification" and "Иваном" in clash["assistant_text"]
    assert say("да")["entries"][0]["time"] == "10:30"


def test_delete_series_by_name(say):
    say("По будням в 9:30 стендап")
    say("Планёрка каждый понедельник в 10")
    ask = say("Отмени стендап")
    assert "Стендап" in ask["assistant_text"]
    say("да")
    assert [e["title"] for e in say.a.storage.all_entries()] == ["Планёрка"]
