import uuid

TZ = "Europe/Moscow"


def test_health_without_asr(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_text_message_contract(client):
    r = client.post("/v1/assistant/messages", json={"text": "Добавь задачу купить хлеб", "timezone": TZ})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"conversation_id", "turn_id", "user_text", "assistant_text", "status", "intent", "entries"}
    uuid.UUID(body["conversation_id"])
    assert body["entries"][0]["title"] == "Купить хлеб"


def test_clarification_over_http(client):
    first = client.post("/v1/assistant/messages", json={"text": "Поставь встречу с Иваном завтра", "timezone": TZ}).json()
    assert first["status"] == "needs_clarification"
    second = client.post(
        "/v1/assistant/messages",
        json={"text": "В 15:00", "timezone": TZ, "conversation_id": first["conversation_id"]},
    ).json()
    assert second["status"] == "completed"
    history = client.get(f"/v1/conversations/{first['conversation_id']}/messages").json()
    assert len(history["messages"]) == 4


def test_validation_errors_are_strings(client):
    r = client.post("/v1/assistant/messages", json={"text": "", "timezone": TZ})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)
    r = client.post("/v1/assistant/messages", json={"text": "привет", "timezone": "Mars/Base"})
    assert r.status_code == 422
    r = client.get("/v1/entries", params={"date_from": "2026-10-09"})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)


def test_entries_crud(client):
    r = client.post("/v1/entries", json={"kind": "meeting", "title": "Планёрка", "date": "2026-10-09",
                                         "time": "11:00", "timezone": TZ, "duration_minutes": 30})
    assert r.status_code == 201
    entry = r.json()

    r = client.post("/v1/entries", json={"kind": "reminder", "title": "Без времени", "date": "2026-10-09", "timezone": TZ})
    assert r.status_code == 422

    items = client.get("/v1/entries", params={"date_from": "2026-10-09", "date_to": "2026-10-15"}).json()["items"]
    assert [i["id"] for i in items] == [entry["id"]]

    assert client.get("/v1/entries", params={"date_from": "2026-10-09", "date_to": "2026-10-15", "kind": "task"}).json()["items"] == []

    r = client.patch(f"/v1/entries/{entry['id']}", json={"status": "completed"})
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert "updated_at" in r.json()

    assert client.patch(f"/v1/entries/{entry['id']}", json={}).status_code == 422

    found = client.get("/v1/entries/search", params={"q": "планерка"}).json()["items"]
    assert [i["id"] for i in found] == [entry["id"]]

    assert client.delete(f"/v1/entries/{entry['id']}").status_code == 204
    assert client.get(f"/v1/entries/{entry['id']}").status_code == 404
    assert client.delete(f"/v1/entries/{entry['id']}").status_code == 404


def test_audio_without_asr_is_503(client):
    r = client.post("/v1/assistant/audio", files={"file": ("a.webm", b"123", "audio/webm")}, data={"timezone": TZ})
    assert r.status_code == 503
    assert isinstance(r.json()["detail"], str)


def test_audio_uses_transcript(client, monkeypatch):
    client.app.state.transcribe_upload = lambda f: "Добавь задачу полить цветы"
    r = client.post("/v1/assistant/audio", files={"file": ("a.webm", b"123", "audio/webm")}, data={"timezone": TZ})
    assert r.status_code == 200
    assert r.json()["user_text"] == "Добавь задачу полить цветы"
    assert r.json()["entries"][0]["title"] == "Полить цветы"


def test_audio_empty_transcript(client):
    client.app.state.transcribe_upload = lambda f: "  "
    r = client.post("/v1/assistant/audio", files={"file": ("a.webm", b"123", "audio/webm")}, data={"timezone": TZ})
    assert r.json()["status"] == "no_action"
    assert r.json()["user_text"] == ""


def test_audio_bad_conversation_id(client):
    client.app.state.transcribe_upload = lambda f: "привет"
    r = client.post("/v1/assistant/audio", files={"file": ("a.webm", b"1", "audio/webm")},
                    data={"timezone": TZ, "conversation_id": "nope"})
    assert r.status_code == 422


def test_tts_not_implemented(client):
    assert client.post("/v1/speech/synthesize", json={"text": "привет"}).status_code == 501


def test_unknown_conversation_404(client):
    assert client.get(f"/v1/conversations/{uuid.uuid4()}/messages").status_code == 404


def test_entry_has_no_null_updated_at(client):
    body = client.post("/v1/assistant/messages", json={"text": "Добавь задачу купить хлеб", "timezone": TZ}).json()
    assert "updated_at" not in body["entries"][0]
    assert "duration_minutes" in body["entries"][0]


def _new_client():
    from fastapi.testclient import TestClient

    import app as app_module

    return TestClient(app_module.app, headers={"X-Requested-With": "assistant"})


def test_two_browsers_have_separate_data(client):
    client.post("/v1/assistant/messages", json={"text": "Добавь задачу моя тайна", "timezone": TZ})
    other = _new_client()
    assert other.get("/v1/entries/undated").json()["items"] == []
    mine = client.get("/v1/entries/undated").json()["items"]
    assert [e["title"] for e in mine] == ["Моя тайна"]
    assert other.get(f"/v1/entries/{mine[0]['id']}").status_code == 404


def test_session_cookie_is_httponly(client):
    r = client.get("/v1/session")
    assert r.json()["unlocked"] is True
    cookie = r.headers.get("set-cookie", "")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie.replace("Lax", "lax")


def test_csrf_header_required_with_cookie(client):
    client.get("/v1/session")
    r = client.post("/v1/assistant/messages", json={"text": "привет", "timezone": TZ},
                    headers={"X-Requested-With": ""})
    assert r.status_code == 403


def test_access_code_gate(client):
    sessions = client.app.state.sessions
    sessions.access_code = "секрет-123"
    try:
        assert client.get("/v1/entries/undated").status_code == 401
        assert client.get("/v1/session").json()["unlocked"] is False
        assert client.post("/v1/session/unlock", json={"code": "нет"}).status_code == 401
        assert client.post("/v1/session/unlock", json={"code": "секрет-123"}).status_code == 204
        assert client.get("/v1/entries/undated").status_code == 200
        assert client.get("/health").status_code == 200
    finally:
        sessions.access_code = ""


def test_calendar_feed(client):
    client.post("/v1/assistant/messages", json={"text": "Планёрка каждый понедельник в 10", "timezone": TZ})
    url = client.get("/v1/calendar").json()["url"]
    path = url.split("://", 1)[1].split("/", 1)[1]
    r = _new_client().get("/" + path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar")
    assert "BEGIN:VEVENT" in r.text and "RRULE:FREQ=WEEKLY" in r.text and "Планёрка" in r.text
    assert client.get("/v1/calendar/feed/" + "x.y" + ".ics").status_code == 404


def test_patch_recurrence_and_validation(client):
    e = client.post("/v1/entries", json={"kind": "task", "title": "Полить цветы", "date": "2026-10-10",
                                         "timezone": TZ}).json()
    r = client.patch(f"/v1/entries/{e['id']}", json={"recurrence": "weekly"})
    assert r.status_code == 200 and r.json()["recurrence"] == "weekly"
    m = client.post("/v1/entries", json={"kind": "meeting", "title": "Созвон", "date": "2026-10-10",
                                         "time": "10:00", "timezone": TZ}).json()
    assert m["duration_minutes"] == 60
    assert client.patch(f"/v1/entries/{m['id']}", json={"time": None}).status_code == 422
