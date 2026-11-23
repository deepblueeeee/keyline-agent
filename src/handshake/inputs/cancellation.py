"""Cooperative cancellation while waiting for local input or preparing an action."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

from handshake.runtime.errors import UserCancelledError

_CURRENT_CANCELLATION: ContextVar[Event | None] = ContextVar(
    "cancellation_scope", default=None
)


@contextmanager
def cancellation_scope(signal: Event) -> Iterator[None]:
    token = _CURRENT_CANCELLATION.set(signal)
    try:
        yield
    finally:
        _CURRENT_CANCELLATION.reset(token)


def check_cancelled() -> None:
    signal = _CURRENT_CANCELLATION.get()
    if signal is not None and signal.is_set():
        raise UserCancelledError("User cancelled input before execution")

