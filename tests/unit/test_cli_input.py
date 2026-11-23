import asyncio
import os
import select
import sys
import termios
from threading import Event

import pytest

import handshake.cli_input as cli_input
from handshake.runtime.errors import UserCancelledError
from tests.history_support import run_async


@pytest.mark.parametrize("cancel", (False, True))
@run_async
async def test_secret_input_hides_value_and_restores_terminal(
    monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    master, slave = os.openpty()
    stdin = os.fdopen(slave, "r", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin)
    ready = Event()
    original = termios.tcgetattr(slave)
    set_attributes = termios.tcsetattr

    def set_and_notify(descriptor, when, attributes):
        set_attributes(descriptor, when, attributes)
        if not attributes[3] & termios.ECHO:
            ready.set()

    monkeypatch.setattr(cli_input.termios, "tcsetattr", set_and_notify)
    terminal = cli_input.TerminalInput()
    task = asyncio.create_task(terminal.read(
        lambda: cli_input.read_terminal_line("Secret: ", hidden=True)
    ))
    canary = b"terminal-secret-canary"
    try:
        assert await asyncio.to_thread(ready.wait, 1)
        os.write(master, canary if cancel else canary + b"\n")
        if cancel:
            task.cancel()
            with pytest.raises(UserCancelledError):
                await task
        else:
            assert await asyncio.wait_for(task, 1) == canary.decode()
        assert termios.tcgetattr(slave) == original
        output = bytearray()
        while select.select([master], [], [], 0)[0]:
            output.extend(os.read(master, 1024))
        assert canary not in output
        if cancel:
            next_line = asyncio.create_task(terminal.read(
                lambda: cli_input.read_terminal_line("Text: ")
            ))
            os.write(master, b"safe\n")
            assert await asyncio.wait_for(next_line, 1) == "safe"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        stdin.close()
        os.close(master)
