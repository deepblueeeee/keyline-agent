"""Small persistence boundary used by the runtime."""

from __future__ import annotations

from typing import Protocol

from handshake.runtime.state import Session, Task


class TaskStore(Protocol):
    def create_task(self, task: Task) -> None:
        """Persist a newly started task."""
        ...

    def checkpoint_task(self, task: Task) -> None:
        """Atomically replace a task snapshot if its version is still current."""
        ...

    def load_task(self, task_id: str) -> Task:
        """Load one task and its model protocol history."""
        ...

    def list_tasks(self, session_id: str | None = None) -> tuple[Task, ...]:
        """List durable tasks, optionally scoped to one session."""
        ...


class SessionStore(Protocol):
    def create_session(self, session: Session) -> None:
        """Persist a new session."""
        ...

    def load_session(self, session_id: str) -> Session:
        """Load one durable session."""
        ...

    def list_sessions(self) -> tuple[Session, ...]:
        """List sessions by most recent task activity."""
        ...


class StateStore(TaskStore, SessionStore, Protocol):
    """Combined persistence boundary used by the CLI application layer."""

    pass
