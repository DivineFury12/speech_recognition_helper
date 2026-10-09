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
