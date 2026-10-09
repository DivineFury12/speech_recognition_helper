import json
import sqlite3
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .dates import occurrences

DEFAULT_USER = "local"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'local',
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    date TEXT,
    time TEXT,
    timezone TEXT NOT NULL,
    duration_minutes INTEGER,
    recurrence TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'local',
    context TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS drafts (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

MIGRATIONS = {
    "entries": {"user_id": "TEXT NOT NULL DEFAULT 'local'", "recurrence": "TEXT"},
    "conversations": {"user_id": "TEXT NOT NULL DEFAULT 'local'", "context": "TEXT"},
}

INDEXES = """
CREATE INDEX IF NOT EXISTS entries_user_date ON entries(user_id, date);
CREATE INDEX IF NOT EXISTS conversations_user ON conversations(user_id);
"""

ENTRY_FIELDS = (
    "id", "kind", "title", "date", "time", "timezone", "duration_minutes",
    "recurrence", "status", "created_at", "updated_at",
)
UPDATABLE_FIELDS = {"title", "date", "time", "timezone", "duration_minutes", "recurrence", "status"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sort_key(entry: dict[str, Any]) -> tuple:
    return (entry["date"] or "9999-99-99", entry["time"] is None, entry["time"] or "", entry["created_at"])


class Storage:
    def __init__(self, path: str, draft_ttl: timedelta = timedelta(minutes=30)) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._draft_ttl = draft_ttl
        with self._lock, self._db:
            self._db.executescript(SCHEMA)
            for table, columns in MIGRATIONS.items():
                existing = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
                for column, ddl in columns.items():
                    if column not in existing:
                        self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._db.executescript(INDEXES)

    def close(self) -> None:
        self._db.close()

    def _fetchone(self, sql: str, args: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, args).fetchone()

    def _fetchall(self, sql: str, args: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _entry(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = {k: row[k] for k in ENTRY_FIELDS}
        if data["updated_at"] is None:
            del data["updated_at"]
        return data

    def create_entry(self, *, kind: str, title: str, date: str | None, time: str | None,
                     timezone: str, duration_minutes: int | None = None, recurrence: str | None = None,
                     user_id: str = DEFAULT_USER) -> dict[str, Any]:
        entry_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO entries (id, user_id, kind, title, date, time, timezone, duration_minutes, recurrence,"
                " status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (entry_id, user_id, kind, title, date, time, timezone, duration_minutes, recurrence, utc_now()),
            )
        return self.get_entry(entry_id, user_id)

    def get_entry(self, entry_id: str, user_id: str = DEFAULT_USER) -> dict[str, Any] | None:
        row = self._fetchone("SELECT * FROM entries WHERE id = ? AND user_id = ?", (entry_id, user_id))
        return self._entry(row)

    def list_entries(self, date_from: str, date_to: str, kind: str | None = None,
                     status: str | None = None, user_id: str = DEFAULT_USER) -> list[dict[str, Any]]:
        sql = ("SELECT * FROM entries WHERE user_id = ? AND date IS NOT NULL AND date <= ?"
               " AND (date >= ? OR recurrence IS NOT NULL)")
        args: list[Any] = [user_id, date_to, date_from]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if status:
            sql += " AND status = ?"
            args.append(status)
        frm, to = date.fromisoformat(date_from), date.fromisoformat(date_to)
        result: list[dict[str, Any]] = []
        for row in self._fetchall(sql, args):
            entry = self._entry(row)
            for day in occurrences(date.fromisoformat(entry["date"]), entry["recurrence"], frm, to):
                result.append({**entry, "date": day.isoformat()})
        result.sort(key=_sort_key)
        return result

    def undated_entries(self, user_id: str = DEFAULT_USER) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT * FROM entries WHERE user_id = ? AND date IS NULL"
            " ORDER BY status = 'completed', created_at DESC",
            (user_id,),
        )
        return [self._entry(r) for r in rows]

    def all_entries(self, kind: str | None = None, user_id: str = DEFAULT_USER) -> list[dict[str, Any]]:
        sql = "SELECT * FROM entries WHERE user_id = ?"
        args: list[Any] = [user_id]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY created_at DESC"
        return [self._entry(r) for r in self._fetchall(sql, args)]

    def update_entry(self, entry_id: str, changes: dict[str, Any],
                     user_id: str = DEFAULT_USER) -> dict[str, Any] | None:
        changes = {k: v for k, v in changes.items() if k in UPDATABLE_FIELDS}
        if not changes:
            return self.get_entry(entry_id, user_id)
        sets = ", ".join(f"{k} = ?" for k in changes)
        with self._lock, self._db:
            cur = self._db.execute(
                f"UPDATE entries SET {sets}, updated_at = ? WHERE id = ? AND user_id = ?",
                (*changes.values(), utc_now(), entry_id, user_id),
            )
        return self.get_entry(entry_id, user_id) if cur.rowcount else None

    def delete_entry(self, entry_id: str, user_id: str = DEFAULT_USER) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM entries WHERE id = ? AND user_id = ?", (entry_id, user_id))
        return cur.rowcount > 0

    def ensure_conversation(self, conversation_id: str | None, user_id: str = DEFAULT_USER) -> str:
        with self._lock, self._db:
            if conversation_id:
                row = self._db.execute("SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if row is not None and row["user_id"] == user_id:
                    return conversation_id
                if row is not None:
                    conversation_id = None
            conversation_id = conversation_id or str(uuid.uuid4())
            self._db.execute(
                "INSERT INTO conversations (id, user_id, created_at) VALUES (?, ?, ?)",
                (conversation_id, user_id, utc_now()),
            )
        return conversation_id

    def conversation_exists(self, conversation_id: str, user_id: str = DEFAULT_USER) -> bool:
        row = self._fetchone("SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        return row is not None

    def get_context(self, conversation_id: str) -> dict[str, Any]:
        row = self._fetchone("SELECT context FROM conversations WHERE id = ?", (conversation_id,))
        return json.loads(row["context"]) if row and row["context"] else {}

    def set_context(self, conversation_id: str, context: dict[str, Any]) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE conversations SET context = ? WHERE id = ?",
                (json.dumps(context, ensure_ascii=False), conversation_id),
            )

    def add_message(self, conversation_id: str, role: str, text: str) -> str:
        message_id = str(uuid.uuid4())
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (id, conversation_id, role, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, text, utc_now()),
            )
        return message_id

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall(
            "SELECT id, role, text, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
            (conversation_id,),
        )
        return [dict(r) for r in rows]

    def get_draft(self, conversation_id: str) -> dict[str, Any] | None:
        row = self._fetchone("SELECT data, updated_at FROM drafts WHERE conversation_id = ?", (conversation_id,))
        if row is None:
            return None
        updated = datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - updated > self._draft_ttl:
            self.delete_draft(conversation_id)
            return None
        return json.loads(row["data"])

    def save_draft(self, conversation_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO drafts (conversation_id, data, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(conversation_id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (conversation_id, json.dumps(data, ensure_ascii=False), utc_now()),
            )

    def delete_draft(self, conversation_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM drafts WHERE conversation_id = ?", (conversation_id,))
