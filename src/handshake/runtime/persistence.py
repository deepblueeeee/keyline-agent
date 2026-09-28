"""Runtime-owned persistence contract over domain state."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from handshake.context.state import ContextCheckpoint, SequencedMessage
from handshake.models.protocol import ModelMessage
from handshake.runtime.state import ContextScope, Run, Session


class RunStore(Protocol):
    """Persist and restore domain objects without exposing backend encodings.

    Run checkpoints include their pending messages in one atomic operation.
    Loaded objects are independent snapshots suitable for runtime recovery.
    """

    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        ...

    def load_session(self, session_id: str) -> Session:
        """Restore a session's execution state."""
        ...

    def list_sessions(self) -> tuple[Session, ...]:
        """Return persisted sessions in most-recently-updated order."""
        ...

    def load_context_scope(self, scope_id: str) -> ContextScope:
        """Restore a stream and its current tail run."""
        ...

    def append_messages(
        self, scope_id: str, messages: tuple[ModelMessage, ...]
    ) -> tuple[SequencedMessage, ...]:
        """Append messages to one ContextScope and return durable sequences."""
        ...

    def read_messages(
        self, scope_id: str, start: int = 0, end: int | None = None
    ) -> Iterator[SequencedMessage]:
        """Stream one scope's messages in the half-open range [start, end)."""
        ...

    def message_count(self, scope_id: str) -> int:
        """Return the number of messages in one context stream."""
        ...

    def has_run_messages(self, run_id: str) -> bool:
        """Return whether a run has any durable protocol messages."""
        ...

    def load_context_checkpoint(self, scope_id: str) -> ContextCheckpoint | None:
        """Load the latest model-view recipe without reading raw messages."""
        ...

    def save_context_checkpoint(
        self, scope_id: str, checkpoint: ContextCheckpoint
    ) -> ContextCheckpoint:
        """Persist a context recipe, rejecting a stale revision or transcript prefix."""
        ...

    def create_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Persist a run and append messages to its context scope."""
        ...

    def checkpoint_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Persist a checkpoint and return its newly appended messages."""
        ...

    def load_run(self, run_id: str) -> Run:
        """Restore a complete run state, including work items and dependencies."""
        ...

    def list_runs(self, session_id: str | None = None) -> tuple[Run, ...]:
        """Return runs in session order, or most-recently-updated order globally."""
        ...
