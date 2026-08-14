"""Small persistence boundary used by the runtime."""

from __future__ import annotations

from typing import Protocol

from handshake.runtime.state import Task


class TaskStore(Protocol):
    def create_task(self, task: Task) -> None:
        """Persist a newly started task."""
        ...

    def checkpoint_task(self, task: Task) -> None:
        """Atomically replace the stable state for an existing task."""
        ...

    def load_task(self, task_id: str) -> Task:
        """Load one task and its model protocol history."""
        ...

    def list_tasks(self) -> tuple[Task, ...]:
        """List durable tasks, newest first."""
        ...
