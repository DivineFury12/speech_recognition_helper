"""Share one SQLAlchemy transaction across an assistant request."""

import contextlib
import contextvars
from collections.abc import Iterator

from sqlalchemy.orm import Session, SessionTransaction, sessionmaker


_active_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    'assistant_active_session', default=None,
)


@contextlib.contextmanager
def use_session(session: Session) -> Iterator[None]:
    token = _active_session.set(session)
    try:
        yield
    finally:
        _active_session.reset(token)


@contextlib.contextmanager
def request_savepoint() -> Iterator[SessionTransaction | None]:
    """Nest an all-or-nothing operation inside the active assistant request."""
    active = _active_session.get()
    if active is None:
        yield None
    else:
        with active.begin_nested() as nested:
            yield nested


@contextlib.contextmanager
def database_session(sessions: sessionmaker[Session], *, write: bool = False) -> Iterator[Session]:
    active = _active_session.get()
    if active is not None:
        yield active
    elif write:
        with sessions.begin() as session:
            yield session
    else:
        with sessions() as session:
            yield session
