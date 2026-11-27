"""Serialized, cancellable terminal input while runtime tasks keep advancing."""

from __future__ import annotations

import asyncio
import getpass
import os
import select
import sys
import termios
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Lock
from typing import TypeVar

from keyline_agent.inputs.cancellation import cancellation_scope, check_cancelled
from keyline_agent.runtime.scheduler import run_blocking

_T = TypeVar("_T")


class TerminalInput:
    """Let exactly one text, Secret, or approval prompt own stdin at a time."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._readers = asyncio.Lock()

    @contextmanager
    def slot(self) -> Iterator[None]:
        while not self._lock.acquire(timeout=0.05):
            check_cancelled()
        try:
            check_cancelled()
            yield
        finally:
            self._lock.release()

    async def read(self, reader: Callable[[], _T]) -> _T:
        cancelled = Event()

        def read() -> _T:
            with cancellation_scope(cancelled), self.slot():
                return reader()

        async with self._readers:
            return await run_blocking(read, on_cancel=cancelled.set)


def read_terminal_line(prompt: str, *, hidden: bool = False) -> str:
    check_cancelled()
    try:
        descriptor = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        # Support embedders with an in-memory input implementation.
        answer = getpass.getpass(prompt) if hidden else input(prompt)
        check_cancelled()
        return answer
    original = termios.tcgetattr(descriptor) if hidden else None
    if original is not None:
        settings = termios.tcgetattr(descriptor)
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(descriptor, termios.TCSANOW, settings)
    print(prompt, end="", flush=True)
    answer_bytes = bytearray()
    try:
        while True:
            check_cancelled()
            readable, _, _ = select.select([descriptor], [], [], 0.05)
            if not readable:
                continue
            character = os.read(descriptor, 1)
            if not character:
                raise EOFError
            if character == b"\n":
                check_cancelled()
                return answer_bytes.decode(sys.stdin.encoding or "utf-8").strip("\r")
            answer_bytes.extend(character)
    finally:
        if original is not None:
            # Discard an unfinished Secret before allowing any ordinary text reader to run.
            termios.tcflush(descriptor, termios.TCIFLUSH)
            termios.tcsetattr(descriptor, termios.TCSANOW, original)
            print()
