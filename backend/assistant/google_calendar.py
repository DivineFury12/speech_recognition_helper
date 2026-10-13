"""Синхронизация записей в Google Календарь пользователя.

Вход через Google OAuth, доступ только к календарям, которые создало само приложение
(scope calendar.app.created). Refresh-токен хранится в базе зашифрованным.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

import httpx

from .storage import Storage

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
API = "https://www.googleapis.com/calendar/v3"
CALENDAR_NAME = "Ассистент"
STATE_TTL = 600
RRULES = {
    "daily": "RRULE:FREQ=DAILY",
    "weekdays": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    "weekly": "RRULE:FREQ=WEEKLY",
    "monthly": "RRULE:FREQ=MONTHLY",
}


class GoogleError(RuntimeError):
    pass


def event_id(entry_id: str) -> str:
    return entry_id.replace("-", "")


def event_body(entry: dict) -> dict:
    title = entry["title"]
    if entry.get("status") == "completed":
        title = f"✓ {title}"
    body: dict = {"summary": title, "description": "Создано голосовым ассистентом", "status": "confirmed"}
    day = date.fromisoformat(entry["date"])
    if entry.get("time"):
        start = datetime.combine(day, datetime.strptime(entry["time"], "%H:%M").time())
        end = start + timedelta(minutes=entry.get("duration_minutes") or 30)
        body["start"] = {"dateTime": start.isoformat(timespec="seconds"), "timeZone": entry["timezone"]}
        body["end"] = {"dateTime": end.isoformat(timespec="seconds"), "timeZone": entry["timezone"]}
    else:
        body["start"] = {"date": day.isoformat()}
        body["end"] = {"date": (day + timedelta(days=1)).isoformat()}
    if entry.get("recurrence") in RRULES:
        body["recurrence"] = [RRULES[entry["recurrence"]]]
    minutes = {"reminder": 0, "meeting": 10}.get(entry["kind"])
    body["reminders"] = ({"useDefault": False, "overrides": [{"method": "popup", "minutes": minutes}]}
                         if minutes is not None and entry.get("time") else {"useDefault": False})
    return body


class GoogleCalendar:
    def __init__(self, storage: Storage, secret: bytes, client_id: str, client_secret: str,
                 scope: str = "https://www.googleapis.com/auth/calendar.app.created",
                 client: httpx.Client | None = None) -> None:
        from cryptography.fernet import Fernet

        self.storage = storage
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self._secret = secret
        self._fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"google-token:" + secret).digest()))
        self._http = client or httpx.Client(timeout=15)
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="google-sync")
        storage.listeners.append(self.on_change)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def connected(self, user_id: str) -> bool:
        return bool(self.storage.user_settings(user_id).get("google_token"))

    def make_state(self, user_id: str, redirect_uri: str) -> str:
        payload = json.dumps({"u": user_id, "r": redirect_uri, "t": int(time.time()), "n": secrets.token_hex(8)})
        body = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        sig = hmac.new(self._secret, b"google-state:" + body.encode(), hashlib.sha256).hexdigest()
        return f"{body}.{sig}"

    def read_state(self, state: str, user_id: str) -> str:
        body, _, sig = state.partition(".")
        expected = hmac.new(self._secret, b"google-state:" + body.encode(), hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(sig, expected):
            raise GoogleError("Неверный state")
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if data["u"] != user_id or time.time() - data["t"] > STATE_TTL:
            raise GoogleError("Ссылка входа устарела или открыта в другом браузере")
        return data["r"]

    def auth_url(self, user_id: str, redirect_uri: str) -> str:
        return AUTH_URL + "?" + urlencode({
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "false",
            "state": self.make_state(user_id, redirect_uri),
        })

    def _token_request(self, data: dict) -> dict:
        r = self._http.post(TOKEN_URL, data={"client_id": self.client_id, "client_secret": self.client_secret, **data})
        if r.status_code != 200:
            raise GoogleError(f"Google отказал в токене ({r.status_code})")
        return r.json()

    def connect(self, user_id: str, code: str, redirect_uri: str) -> None:
        tokens = self._token_request({"code": code, "grant_type": "authorization_code", "redirect_uri": redirect_uri})
        refresh = tokens.get("refresh_token")
        if not refresh:
            raise GoogleError("Google не выдал refresh-токен, отключите доступ приложения в аккаунте и подключите заново")
        with self._lock:
            self._tokens[user_id] = (tokens["access_token"], time.time() + int(tokens.get("expires_in", 3600)) - 60)
        self.storage.update_user_settings(user_id, google_token=self._fernet.encrypt(refresh.encode()).decode())
        calendar_id = self.storage.user_settings(user_id).get("google_calendar_id")
        if not calendar_id or self._api(user_id, "GET", f"/calendars/{calendar_id}").status_code != 200:
            r = self._api(user_id, "POST", "/calendars", json={"summary": CALENDAR_NAME, "timeZone": "Europe/Moscow"})
            if r.status_code != 200:
                raise GoogleError(f"Не удалось создать календарь ({r.status_code})")
            self.storage.update_user_settings(user_id, google_calendar_id=r.json()["id"])
        self._pool.submit(self.sync_all, user_id)

    def disconnect(self, user_id: str) -> None:
        settings = self.storage.user_settings(user_id)
        if settings.get("google_token"):
            try:
                refresh = self._fernet.decrypt(settings["google_token"].encode()).decode()
                self._http.post(REVOKE_URL, data={"token": refresh})
            except Exception as e:
                log.warning("Не удалось отозвать токен Google: %s", type(e).__name__)
        with self._lock:
            self._tokens.pop(user_id, None)
        self.storage.update_user_settings(user_id, google_token=None)

    def _access_token(self, user_id: str) -> str:
        with self._lock:
            cached = self._tokens.get(user_id)
        if cached and cached[1] > time.time():
            return cached[0]
        stored = self.storage.user_settings(user_id).get("google_token")
        if not stored:
            raise GoogleError("Google Календарь не подключён")
        refresh = self._fernet.decrypt(stored.encode()).decode()
        tokens = self._token_request({"refresh_token": refresh, "grant_type": "refresh_token"})
        with self._lock:
            self._tokens[user_id] = (tokens["access_token"], time.time() + int(tokens.get("expires_in", 3600)) - 60)
        return tokens["access_token"]

    def _api(self, user_id: str, method: str, path: str, **kwargs) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._access_token(user_id)}"}
        return self._http.request(method, API + path, headers=headers, **kwargs)

    def on_change(self, op: str, entry: dict, user_id: str) -> None:
        if self.configured and self.connected(user_id):
            self._pool.submit(self._safe_push, op, dict(entry), user_id)

    def _safe_push(self, op: str, entry: dict, user_id: str) -> None:
        try:
            self.push(op, entry, user_id)
        except Exception as e:
            log.warning("Синхронизация с Google не удалась: %s", e)

    def push(self, op: str, entry: dict, user_id: str) -> None:
        calendar_id = self.storage.user_settings(user_id).get("google_calendar_id")
        if not calendar_id:
            return
        path = f"/calendars/{calendar_id}/events"
        eid = event_id(entry["id"])
        if op == "delete" or not entry.get("date") or entry.get("status") == "cancelled":
            r = self._api(user_id, "DELETE", f"{path}/{eid}")
            if r.status_code not in (200, 204, 404, 410):
                raise GoogleError(f"удаление события: {r.status_code}")
            return
        body = event_body(entry)
        r = self._api(user_id, "PUT", f"{path}/{eid}", json=body)
        if r.status_code == 404:
            r = self._api(user_id, "POST", path, json={**body, "id": eid})
        if r.status_code not in (200, 201):
            raise GoogleError(f"сохранение события: {r.status_code}")

    def sync_all(self, user_id: str) -> int:
        count = 0
        for entry in self.storage.all_entries(None, user_id):
            if entry.get("date"):
                try:
                    self.push("upsert", entry, user_id)
                    count += 1
                except Exception as e:
                    log.warning("Синхронизация с Google не удалась: %s", e)
        return count
