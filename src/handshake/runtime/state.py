"""In-memory state for the first Handshake vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from handshake.runtime.errors import InvalidStateTransitionError

ContentBlock = dict[str, Any]


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TASK_TRANSITIONS = {
    TaskStatus.CREATED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}

_TOOL_RUN_TRANSITIONS = {
    ToolRunStatus.PENDING: {ToolRunStatus.RUNNING},
    ToolRunStatus.RUNNING: {ToolRunStatus.SUCCEEDED, ToolRunStatus.FAILED},
    ToolRunStatus.SUCCEEDED: set(),
    ToolRunStatus.FAILED: set(),
}


@dataclass(slots=True, frozen=True)
class ModelMessage:
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": list(self.content)}


@dataclass(slots=True, frozen=True)
class ModelContext:
    system_prompt: str
    goal: str
    workspace: str
    plan: tuple[str, ...]
    messages: tuple[ModelMessage, ...]


@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ModelTurn:
    content: tuple[ContentBlock, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(
            str(block["text"])
            for block in self.content
            if block.get("type") == "text" and block.get("text")
        ).strip()


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
    run_id: str = field(default_factory=lambda: str(uuid4()))
    status: ToolRunStatus = ToolRunStatus.PENDING
    result: ToolResult | None = None

    def transition(self, target: ToolRunStatus) -> None:
        if target not in _TOOL_RUN_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition tool run from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class Task:
    goal: str
    workspace: Path
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.CREATED
    plan: list[str] = field(default_factory=list)
    messages: list[ModelMessage] = field(default_factory=list)
    tool_runs: list[ToolRun] = field(default_factory=list)
    final_summary: str | None = None
    error: str | None = None

    def transition(self, target: TaskStatus) -> None:
        if target not in _TASK_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition task from {self.status} to {target}"
            )
        self.status = target
