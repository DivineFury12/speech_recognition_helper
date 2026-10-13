import pytest

from assistant.speech_text import for_speech
from assistant.tts import VOICES, TTSUnavailableError


class FakeEngine:
    def __init__(self, ready=True):
        self.ready = ready
        self.error = None
        self.calls = []

    def voices(self):
        return VOICES

    def synthesize(self, text, voice_id):
        if not self.ready:
            raise TTSUnavailableError("Модель озвучки ещё загружается")
        if voice_id not in {v.id for v in VOICES}:
            raise KeyError(voice_id)
        self.calls.append((text, voice_id))
        return b"RIFF....WAVEfake"


def test_voices_and_synthesis(client):
    client.app.state.tts = FakeEngine()
    data = client.get("/v1/speech/voices").json()
    assert data["available"] is True
    assert [v["name"] for v in data["voices"]] == ["Никита", "Сергей", "Алина"]
    r = client.post("/v1/speech/synthesize", json={"text": "Готово", "voice": "sergey"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.post("/v1/speech/synthesize", json={"text": "Готово", "voice": "alisa"}).status_code == 404
    client.app.state.tts = None


def test_synthesis_while_loading(client):
    client.app.state.tts = FakeEngine(ready=False)
    assert client.get("/v1/speech/voices").json()["loading"] is True
    assert client.post("/v1/speech/synthesize", json={"text": "Готово"}).status_code == 503
    client.app.state.tts = None


@pytest.mark.parametrize("text, expected", [
    ("завтра в 15:00", "завтра в пятнадцать ноль-ноль"),
    ("в 9:30", "в девять тридцать"),
    ("в 11:05", "в одиннадцать ноль пять"),
    ("в 15:00–16:00", "с пятнадцати ноль-ноль до шестнадцати ноль-ноль"),
    ("12 октября", "двенадцатого октября"),
    ("на 21 октября", "на двадцать первое октября"),
    ("С пн, 12 октября", "С понедельника, двенадцатого октября"),
    ("на вт, 14 октября", "на вторник, четырнадцатого октября"),
    ("2 записи", "две записи"),
    ("1 задача", "одна задача"),
    ("5 записей", "пять записей"),
    ("на 30 мин", "на тридцать минут"),
    ("«Встреча с Иваном» (встреча) ✓", "Встреча с Иваном, встреча"),
])
def test_for_speech(text, expected):
    assert for_speech(text) == expected
