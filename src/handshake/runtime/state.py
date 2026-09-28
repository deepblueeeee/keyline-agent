"""Runtime execution state for sessions, runs, work, tools, and dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from secrets import token_urlsafe
from typing import Any
from uuid import uuid4

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
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DependencyKind(StrEnum):
    TEXT = "text"
    SECRET = "secret"


class DependencyStatus(StrEnum):
    OPEN = "open"
    SATISFIED = "satisfied"
    LOST = "lost"


class SecretHandleStatus(StrEnum):
    REQUESTED = "requested"
    AVAILABLE = "available"
    CONSUMED = "consumed"
    LOST = "lost"


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
    WorkItemStatus.READY: {WorkItemStatus.RUNNING, WorkItemStatus.FAILED},
    WorkItemStatus.RUNNING: {
        WorkItemStatus.READY,
        WorkItemStatus.COMPLETED,
        WorkItemStatus.FAILED,
    },
    WorkItemStatus.COMPLETED: set(),
    WorkItemStatus.FAILED: set(),
}

_DEPENDENCY_TRANSITIONS = {
    DependencyStatus.OPEN: {DependencyStatus.SATISFIED, DependencyStatus.LOST},
    DependencyStatus.SATISFIED: {DependencyStatus.LOST},
    DependencyStatus.LOST: set(),
}

_SECRET_HANDLE_TRANSITIONS = {
    SecretHandleStatus.REQUESTED: {
        SecretHandleStatus.AVAILABLE,
        SecretHandleStatus.LOST,
    },
    SecretHandleStatus.AVAILABLE: {
        SecretHandleStatus.CONSUMED,
        SecretHandleStatus.LOST,
    },
    SecretHandleStatus.CONSUMED: set(),
    SecretHandleStatus.LOST: set(),
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
    work_id: str
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
    work_id: str = field(default_factory=lambda: str(uuid4()))
    status: WorkItemStatus = WorkItemStatus.READY

    def transition(self, target: WorkItemStatus) -> None:
        if target not in _WORK_ITEM_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition work item from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class Dependency:
    work_id: str
    tool_call_id: str
    prompt: str
    dependency_id: str = field(default_factory=lambda: str(uuid4()))
    kind: DependencyKind = DependencyKind.TEXT
    status: DependencyStatus = DependencyStatus.OPEN
    created_at: str = field(default_factory=utc_now)
    resolved_at: str | None = None

    def transition(self, target: DependencyStatus) -> None:
        if target not in _DEPENDENCY_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition dependency from {self.status} to {target}"
            )
        self.status = target


@dataclass(slots=True)
class SecretHandle:
    dependency_id: str
    purpose: str
    consumer: str
    environment_variable: str
    secret_handle: str = field(
        default_factory=lambda: f"secret_{token_urlsafe(24)}"
    )
    status: SecretHandleStatus = SecretHandleStatus.REQUESTED
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def transition(self, target: SecretHandleStatus) -> None:
        if target not in _SECRET_HANDLE_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition secret handle from {self.status} to {target}"
            )
        self.status = target
        self.updated_at = utc_now()


@dataclass(slots=True)
class Session:
    workspace: Path
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    context_scope: ContextScope = field(init=False)

    def __post_init__(self) -> None:
        self.context_scope = ContextScope(self.session_id, self.session_id)


@dataclass(frozen=True, slots=True)
class ContextScope:
    """One message stream and the last run that extended it."""

    scope_id: str
    session_id: str
    tail_run_id: str | None = None

    def owns(self, run: Run) -> bool:
        """Whether a run belongs to this session's message stream."""
        return run.session_id == self.session_id and run.context_scope_id == self.scope_id

    def is_tail(self, run: Run) -> bool:
        """Whether the run can continue writing to this stream."""
        return self.owns(run) and self.tail_run_id == run.run_id

    def new_run(
        self,
        goal: str,
        *,
        sequence: int,
        model_name: str,
        max_steps: int,
        agent_spec_id: str,
    ) -> Run:
        """Build the next run in this stream without copying its history."""
        return Run(
            goal=goal,
            session_id=self.session_id,
            sequence=sequence,
            previous_run_id=self.tail_run_id,
            agent_spec_id=agent_spec_id,
            context_scope_id=self.scope_id,
            model_name=model_name,
            max_steps=max_steps,
        )

    def advance(self, run: Run) -> ContextScope:
        """Validate the stream predecessor and return its new durable head."""
        if not self.owns(run):
            raise InvalidStateTransitionError("Run does not belong to this context scope")
        if run.previous_run_id != self.tail_run_id:
            raise InvalidStateTransitionError("Run does not follow the context scope tail")
        return ContextScope(self.scope_id, self.session_id, run.run_id)


@dataclass(slots=True)
class Run:
    goal: str
    session_id: str
    agent_spec_id: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    sequence: int = 1
    previous_run_id: str | None = None
    parent_run_id: str | None = None
    context_scope_id: str = ""
    status: RunStatus = RunStatus.CREATED
    plan: list[str] = field(default_factory=list)
    tool_runs: list[ToolRun] = field(default_factory=list)
    work_items: list[WorkItem] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    secret_handles: list[SecretHandle] = field(default_factory=list)
    final_summary: str | None = None
    error: str | None = None
    model_name: str = ""
    model_turns: int = 0
    max_steps: int = 50
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_spec_id, str) or not self.agent_spec_id.strip():
            raise ValueError("Run must have an agent spec ID")
        if not self.context_scope_id:
            self.context_scope_id = (
                str(uuid4()) if self.parent_run_id is not None else self.session_id
            )
        if not self.work_items:
            self.work_items.append(WorkItem(description=self.goal))

    def transition(self, target: RunStatus) -> None:
        if target not in _RUN_TRANSITIONS[self.status]:
            raise InvalidStateTransitionError(
                f"Cannot transition run from {self.status} to {target}"
            )
        self.status = target

    def work_item_by_id(self, work_id: str) -> WorkItem | None:
        """Return the work item identified within this run, if present."""
        return next(
            (item for item in self.work_items if item.work_id == work_id),
            None,
        )

    def dependency_by_id(self, dependency_id: str) -> Dependency | None:
        """Return the dependency identified within this run, if present."""
        return next(
            (
                dependency
                for dependency in self.dependencies
                if dependency.dependency_id == dependency_id
            ),
            None,
        )

    def secret_handle_by_id(self, secret_handle: str) -> SecretHandle | None:
        """Return opaque Secret metadata owned by this run, if present."""
        return next(
            (
                handle
                for handle in self.secret_handles
                if handle.secret_handle == secret_handle
            ),
            None,
        )

    def add_dependency(
        self, work_item: WorkItem, dependency: Dependency
    ) -> None:
        """Associate a dependency with work owned by this run."""
        owned_work_item = self._require_work_item(work_item)
        if dependency.work_id != owned_work_item.work_id:
            raise InvalidStateTransitionError(
                f"Dependency {dependency.dependency_id} is bound to work item "
                f"{dependency.work_id}, not {owned_work_item.work_id}"
            )
        if self.dependency_by_id(dependency.dependency_id) is not None:
            raise InvalidStateTransitionError(
                f"Dependency {dependency.dependency_id} already belongs to run {self.run_id}"
            )
        self.dependencies.append(dependency)

    def add_tool_run(self, work_item: WorkItem, tool_run: ToolRun) -> None:
        """Associate an executed tool action with work owned by this run."""
        owned_work_item = self._require_work_item(work_item)
        if tool_run.work_id != owned_work_item.work_id:
            raise InvalidStateTransitionError(
                f"Tool run {tool_run.tool_run_id} is bound to work item "
                f"{tool_run.work_id}, not {owned_work_item.work_id}"
            )
        if any(
            existing.tool_run_id == tool_run.tool_run_id
            for existing in self.tool_runs
        ):
            raise InvalidStateTransitionError(
                f"Tool run {tool_run.tool_run_id} already belongs to run {self.run_id}"
            )
        self.tool_runs.append(tool_run)

    def add_secret_handle(
        self, dependency: Dependency, secret_handle: SecretHandle
    ) -> None:
        """Associate opaque Secret metadata with a SECRET dependency."""
        owned_dependency = self.dependency_by_id(dependency.dependency_id)
        if owned_dependency is None:
            raise InvalidStateTransitionError(
                f"Dependency {dependency.dependency_id} does not belong to run {self.run_id}"
            )
        if owned_dependency.kind is not DependencyKind.SECRET:
            raise InvalidStateTransitionError(
                f"Dependency {dependency.dependency_id} is not a secret dependency"
            )
        if secret_handle.dependency_id != owned_dependency.dependency_id:
            raise InvalidStateTransitionError(
                f"Secret handle {secret_handle.secret_handle} is bound to dependency "
                f"{secret_handle.dependency_id}, not {owned_dependency.dependency_id}"
            )
        if self.secret_handle_by_id(secret_handle.secret_handle) is not None:
            raise InvalidStateTransitionError(
                f"Secret handle {secret_handle.secret_handle} already belongs to run "
                f"{self.run_id}"
            )
        if self.secret_handle_for(owned_dependency) is not None:
            raise InvalidStateTransitionError(
                f"Dependency {owned_dependency.dependency_id} already has a secret handle"
            )
        self.secret_handles.append(secret_handle)

    def secret_handle_for(self, dependency: Dependency) -> SecretHandle | None:
        """Return the opaque Secret metadata associated with a dependency."""
        owned_dependency = self.dependency_by_id(dependency.dependency_id)
        if owned_dependency is None:
            raise InvalidStateTransitionError(
                f"Dependency {dependency.dependency_id} does not belong to run {self.run_id}"
            )
        return next(
            (
                handle
                for handle in self.secret_handles
                if handle.dependency_id == owned_dependency.dependency_id
            ),
            None,
        )

    def dependencies_for(self, work_item: WorkItem) -> tuple[Dependency, ...]:
        """Return dependencies associated with work owned by this run."""
        owned_work_item = self._require_work_item(work_item)
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.work_id == owned_work_item.work_id
        )

    def tool_runs_for(self, work_item: WorkItem) -> tuple[ToolRun, ...]:
        """Return tool runs associated with work owned by this run."""
        owned_work_item = self._require_work_item(work_item)
        return tuple(
            tool_run
            for tool_run in self.tool_runs
            if tool_run.work_id == owned_work_item.work_id
        )

    def _require_work_item(self, work_item: WorkItem) -> WorkItem:
        owned_work_item = self.work_item_by_id(work_item.work_id)
        if owned_work_item is None:
            raise InvalidStateTransitionError(
                f"Work item {work_item.work_id} does not belong to run {self.run_id}"
            )
        return owned_work_item

    def open_dependencies(self) -> tuple[Dependency, ...]:
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.status is DependencyStatus.OPEN
        )
