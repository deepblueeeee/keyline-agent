"""Runtime execution state for sessions, runs, tools, and waits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import InvalidStateTransitionError


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkItemStatus(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    COMPLETED = "completed"


class WaitKind(StrEnum):
    TEXT = "text"


class WaitStatus(StrEnum):
    OPEN = "open"
    SATISFIED = "satisfied"


_RUN_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {
        RunStatus.WAITING,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.WAITING: {
        RunStatus.RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}

_WORK_ITEM_TRANSITIONS = {
    WorkItemStatus.WAITING: {WorkItemStatus.READY},
    WorkItemStatus.READY: {WorkItemStatus.COMPLETED},
    WorkItemStatus.COMPLETED: set(),
}

_WAIT_TRANSITIONS = {
    WaitStatus.OPEN: {WaitStatus.SATISFIED},
    WaitStatus.SATISFIED: set(),
}


def utc_now() -> str:
    """Return a sortable UTC timestamp suitable for SQLite text columns."""
    return datetime.now(UTC).isoformat()

_TOOL_RUN_TRANSITIONS = {
    ToolRunStatus.PENDING: {ToolRunStatus.RUNNING},
    ToolRunStatus.RUNNING: {ToolRunStatus.SUCCEEDED, ToolRunStatus.FAILED},
    ToolRunStatus.SUCCEEDED: set(),
    ToolRunStatus.FAILED: set(),
}


@dataclass(slots=True, frozen=True)
class ToolResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


@dataclass(slots=True)
class ToolRun:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    tool_run_id: str = field(default_factory=lambda: str(uuid4()))
    status: ToolRunStatus = ToolRunStatus.PENDING
    result: ToolResult | None = None

    def transition(self, target: ToolRunStatus) -> None:
        if target not in _TOOL_RUN_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition tool run from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class WorkItem:
    description: str
    tool_call_id: str
    work_id: str = field(default_factory=lambda: str(uuid4()))
    status: WorkItemStatus = WorkItemStatus.WAITING

    def transition(self, target: WorkItemStatus) -> None:
        if target not in _WORK_ITEM_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition work item from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class WaitCondition:
    work_id: str
    tool_call_id: str
    prompt: str
    wait_id: str = field(default_factory=lambda: str(uuid4()))
    kind: WaitKind = WaitKind.TEXT
    status: WaitStatus = WaitStatus.OPEN
    response_text: str | None = None
    created_at: str = field(default_factory=utc_now)
    resolved_at: str | None = None

    def transition(self, target: WaitStatus) -> None:
        if target not in _WAIT_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition wait from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class Session:
    workspace: Path
    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[ModelMessage] = field(default_factory=list)
    version: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class Run:
    goal: str
    session_id: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    sequence: int = 1
    previous_run_id: str | None = None
    status: RunStatus = RunStatus.CREATED
    plan: list[str] = field(default_factory=list)
    tool_runs: list[ToolRun] = field(default_factory=list)
    work_items: list[WorkItem] = field(default_factory=list)
    waits: list[WaitCondition] = field(default_factory=list)
    final_summary: str | None = None
    error: str | None = None
    model_name: str = ""
    model_turns: int = 0
    max_steps: int = 50
    version: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def transition(self, target: RunStatus) -> None:
        if target not in _RUN_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition run from {self.status} to {target}"
            )
        self.status = target

    def open_waits(self) -> tuple[WaitCondition, ...]:
        return tuple(wait for wait in self.waits if wait.status is WaitStatus.OPEN)
