"""Cooperative cancellation before a blocking tool starts external work."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

from handshake.runtime.errors import UserCancelledError

_CURRENT_CANCELLATION: ContextVar[Event | None] = ContextVar(
    "tool_cancellation", default=None
)


@contextmanager
def tool_cancellation(signal: Event) -> Iterator[None]:
    token = _CURRENT_CANCELLATION.set(signal)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION.reset(token)


def check_tool_cancellation() -> None:
    signal = _CURRENT_CANCELLATION.get()
    if signal is not None and signal.is_set():
        raise UserCancelledError("User cancelled tool before execution")


def has_tool_cancellation() -> bool:
    return _CURRENT_CANCELLATION.get() is not None
