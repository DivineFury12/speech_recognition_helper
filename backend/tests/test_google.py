import json

import httpx
import pytest

from assistant.google_calendar import GoogleCalendar, GoogleError, event_body, event_id
from assistant.storage import Storage

USER = "11111111-1111-1111-1111-111111111111"


def test_event_body_timed_recurring_and_all_day():
    timed = event_body({"title": "Планёрка", "date": "2026-10-19", "time": "10:00", "timezone": "Europe/Moscow",
                        "kind": "meeting", "duration_minutes": 45, "recurrence": "weekly", "status": "pending"})
    assert timed["start"] == {"dateTime": "2026-10-19T10:00:00", "timeZone": "Europe/Moscow"}
    assert timed["end"]["dateTime"] == "2026-10-19T10:45:00"
    assert timed["recurrence"] == ["RRULE:FREQ=WEEKLY"]
    day = event_body({"title": "Отчёт", "date": "2026-10-19", "time": None, "timezone": "Europe/Moscow",
                      "kind": "task", "status": "completed"})
    assert day["start"] == {"date": "2026-10-19"} and day["end"] == {"date": "2026-10-20"}
    assert day["summary"] == "✓ Отчёт"


@pytest.fixture()
def google():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "at", "refresh_token": "rt", "expires_in": 3600})
        if request.method == "POST" and request.url.path.endswith("/calendars"):
            return httpx.Response(200, json={"id": "cal1"})
        if request.method == "PUT":
            return httpx.Response(404)
        return httpx.Response(200, json={})

    storage = Storage(":memory:")
    g = GoogleCalendar(storage, b"s" * 32, "id", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    g._pool.submit = lambda fn, *a: fn(*a)
    yield g, storage, calls
    storage.close()


def test_connect_encrypts_token_and_syncs_changes(google):
    g, storage, calls = google
    state = g.make_state(USER, "http://127.0.0.1:8000/v1/google/callback")
    assert g.read_state(state, USER).endswith("/callback")
    with pytest.raises(GoogleError):
        g.read_state(state, "22222222-2222-2222-2222-222222222222")
    g.connect(USER, "code", "http://127.0.0.1:8000/v1/google/callback")
    saved = storage.user_settings(USER)
    assert saved["google_calendar_id"] == "cal1" and "rt" not in saved["google_token"]
    entry = storage.create_entry(kind="meeting", title="Созвон", date="2026-10-20", time="11:00",
                                 timezone="Europe/Moscow", duration_minutes=60, user_id=USER)
    posts = [c for c in calls if c[0] == "POST" and c[1].endswith("/events")]
    assert json.loads(posts[-1][2])["id"] == event_id(entry["id"])
    storage.delete_entry(entry["id"], USER)
    assert calls[-1][0] == "DELETE" and calls[-1][1].endswith(event_id(entry["id"]))
    g.disconnect(USER)
    assert not g.connected(USER)
