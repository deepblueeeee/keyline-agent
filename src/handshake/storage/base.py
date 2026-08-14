"""Small persistence boundary used by the runtime."""

from __future__ import annotations

from typing import Protocol

from handshake.runtime.state import Task


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
