"""The user of the current request; repositories filter every query by it."""

import contextlib
import contextvars
from collections.abc import Iterator

DEFAULT_USER = 'local'

_current_user: contextvars.ContextVar[str] = contextvars.ContextVar('current_user', default=DEFAULT_USER)


def current_user() -> str:
    return _current_user.get()


@contextlib.contextmanager
def acting_as(user_id: str) -> Iterator[None]:
    token = _current_user.set(user_id)
    try:
        yield
    finally:
        _current_user.reset(token)
