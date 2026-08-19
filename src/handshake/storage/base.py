"""Small persistence boundary used by the runtime."""

from __future__ import annotations

from typing import Protocol

from handshake.runtime.state import Run, Session


class RunStore(Protocol):
    def create_run(self, session: Session, run: Run) -> None:
        """Persist a newly started run and its initial session message."""
        ...

    def checkpoint_run(self, session: Session, run: Run) -> None:
        """Atomically persist session conversation and one run snapshot."""
        ...
