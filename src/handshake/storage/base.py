"""Small persistence boundary used by the runtime."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from handshake.context.state import ContextCheckpoint, SequencedMessage
from handshake.models.protocol import ModelMessage
from handshake.runtime.state import Run, Session


class RunStore(Protocol):
    def append_messages(
        self, session_id: str, messages: tuple[ModelMessage, ...]
    ) -> tuple[SequencedMessage, ...]:
        """Append messages and return their assigned durable sequences."""
        ...

    def read_messages(
        self, session_id: str, start: int = 0, end: int | None = None
    ) -> Iterator[SequencedMessage]:
        """Stream messages in the half-open sequence range [start, end)."""
        ...

    def message_count(self, session_id: str, run_id: str | None = None) -> int:
        """Return the number of messages in a session transcript."""
        ...

    def has_run_messages(self, session_id: str, run_id: str) -> bool:
        """Return whether a run has any durable protocol messages."""
        ...

    def load_context_checkpoint(self, session_id: str) -> ContextCheckpoint | None:
        """Load the latest model-view recipe without reading raw messages."""
        ...

    def save_context_checkpoint(
        self, session_id: str, checkpoint: ContextCheckpoint
    ) -> ContextCheckpoint:
        """CAS-save a recipe for the exact durable transcript prefix it describes."""
        ...

    def create_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Persist a run and return its newly appended transcript messages."""
        ...

    def checkpoint_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Persist a checkpoint and return its newly appended messages."""
        ...
