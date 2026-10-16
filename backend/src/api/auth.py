"""Cookie users, access code with lockout, CSRF header, rate limits and security headers."""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.user_context import acting_as

log = logging.getLogger(__name__)

COOKIE_NAME = "sa_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365
CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "assistant"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
OPEN_PATHS = {"/health", "/v1/session", "/v1/session/unlock"}
PROTECTED_PREFIXES = ("/v1/", "/transcribe")
CALENDAR_PREFIX = "/v1/calendar/feed/"
MAX_BODY_BYTES = 6 * 1024 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "media-src 'self' blob:; connect-src 'self'; font-src 'self' data:; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'; object-src 'none'"
    ),
}


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def load_secret(explicit: str, db_path: str) -> bytes:
    if explicit:
        return explicit.encode()
    if db_path == ":memory:":
        return secrets.token_bytes(32)
    path = Path(db_path).parent / "session_secret"
    try:
        if path.is_file():
            return bytes.fromhex(path.read_text().strip())
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_bytes(32)
        path.write_text(value.hex())
        return value
    except (OSError, ValueError) as e:
        log.warning("Не удалось сохранить секрет сессий, куки сбросятся после перезапуска: %s", e)
        return secrets.token_bytes(32)


@dataclass
class Session:
    user_id: str
    unlocked: bool


class Sessions:
    def __init__(self, secret: bytes, access_code: str = "") -> None:
        self._secret = secret
        self.access_code = access_code

    @property
    def secret(self) -> bytes:
        return self._secret

    def _sign(self, payload: bytes) -> str:
        return _b64(hmac.new(self._secret, payload, hashlib.sha256).digest())

    def dump(self, session: Session) -> str:
        payload = json.dumps({"u": session.user_id, "k": session.unlocked}, separators=(",", ":")).encode()
        return f"{_b64(payload)}.{self._sign(payload)}"

    def load(self, value: str | None) -> Session | None:
        if not value or "." not in value:
            return None
        body, signature = value.rsplit(".", 1)
        try:
            payload = _unb64(body)
        except ValueError:
            return None
        if not hmac.compare_digest(signature, self._sign(payload)):
            return None
        try:
            data = json.loads(payload)
            return Session(user_id=str(uuid.UUID(data["u"])), unlocked=bool(data.get("k")))
        except (ValueError, KeyError, TypeError):
            return None

    def check_code(self, code: str) -> bool:
        if not self.access_code:
            return False
        given = hashlib.sha256(code.strip().encode()).digest()
        return hmac.compare_digest(given, hashlib.sha256(self.access_code.encode()).digest())

    def _feed_sig(self, user_id: str, version: int) -> str:
        payload = f"feed:{user_id}" if version == 0 else f"feed:{user_id}:{version}"
        return self._sign(payload.encode())[:32]

    def feed_token(self, user_id: str, version: int = 0) -> str:
        if version == 0:
            return f"{user_id}.{self._feed_sig(user_id, 0)}"
        return f"{user_id}.{version}.{self._feed_sig(user_id, version)}"

    def feed_user(self, token: str) -> tuple[str, int] | None:
        parts = token.split(".")
        if len(parts) == 2:
            user_id, version, signature = parts[0], 0, parts[1]
        elif len(parts) == 3 and parts[1].isdigit():
            user_id, version, signature = parts[0], int(parts[1]), parts[2]
        else:
            return None
        ok = hmac.compare_digest(signature, self._feed_sig(user_id, version))
        return (user_id, version) if ok else None


class Lockout:
    """Блокирует перебор кода доступа: после N неудачных попыток — пауза."""

    def __init__(self, attempts: int = 5, seconds: float = 900) -> None:
        self.attempts = attempts
        self.seconds = seconds
        self._fails: dict[str, deque] = {}
        self._lock = threading.Lock()

    def blocked(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            fails = self._fails.get(key)
            while fails and now - fails[0] > self.seconds:
                fails.popleft()
            return bool(fails) and len(fails) >= self.attempts

    def fail(self, key: str) -> None:
        with self._lock:
            self._fails.setdefault(key, deque()).append(time.monotonic())
            if len(self._fails) > 10_000:
                self._fails.clear()

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


class RateLimiter:
    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] <= self.window}
            return True


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


class SessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, sessions: Sessions, cookie_secure: bool = False) -> None:
        super().__init__(app)
        self.sessions = sessions
        self.cookie_secure = cookie_secure

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        cookie = request.cookies.get(COOKIE_NAME)
        session = self.sessions.load(cookie)
        had_cookie = session is not None
        if session is None:
            session = Session(user_id=str(uuid.uuid4()), unlocked=False)
        request.state.session = session
        request.state.cookie_dirty = not had_cookie

        length = request.headers.get('content-length')
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return _error(413, 'Слишком большой запрос')
        protected = path.startswith(PROTECTED_PREFIXES) and not path.startswith(CALENDAR_PREFIX)
        if protected and had_cookie and request.method in UNSAFE_METHODS:
            if request.headers.get(CSRF_HEADER, "").lower() != CSRF_VALUE:
                return _error(403, "Запрос отклонён: нет заголовка X-Requested-With")
        if (protected and path not in OPEN_PATHS and self.sessions.access_code
                and not request.state.session.unlocked):
            response = _error(401, "Нужен код доступа")
        else:
            with acting_as(session.user_id):
                response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if path.startswith('/v1/'):
            response.headers.setdefault('Cache-Control', 'no-store')

        if request.state.cookie_dirty:
            response.set_cookie(
                COOKIE_NAME,
                self.sessions.dump(request.state.session),
                max_age=COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=self.cookie_secure or request.url.scheme == "https",
                path="/",
            )
        return response
