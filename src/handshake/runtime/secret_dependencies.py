"""Runtime transitions for out-of-band Secret dependencies."""

from __future__ import annotations

from typing import Protocol

from handshake.inputs import SecretWriter
from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import SecretInputError
from handshake.runtime.state import (
    Dependency,
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    SecretHandle,
    SecretHandleStatus,
    Session,
    WorkItemStatus,
    utc_now,
)


class SecretAvailability(Protocol):
    def contains_secret(self, secret_handle: str) -> bool:
        """Return whether a raw value is available in this process."""
        ...


def resolve_secret_dependency(
    session: Session,
    run: Run,
    dependency_id: str,
    raw_value: str,
    writer: SecretWriter,
) -> None:
    """Route one raw Secret to its handle without adding it to the transcript."""
    if raw_value == "":
        raise SecretInputError("Secret value must not be empty")
    dependency = run.dependency_by_id(dependency_id)
    if dependency is None:
        raise SecretInputError(
            f"Dependency {dependency_id} does not belong to run {run.run_id}"
        )
    if dependency.kind is not DependencyKind.SECRET:
        raise SecretInputError(f"Dependency {dependency_id} does not accept Secret input")
    if dependency.status is not DependencyStatus.OPEN:
        raise SecretInputError(f"Dependency {dependency_id} is not open")
    if run.status is not RunStatus.WAITING:
        raise SecretInputError(f"Run {run.run_id} is not waiting for input")
    work_item = run.work_item_by_id(dependency.work_id)
    if work_item is None or work_item.status is not WorkItemStatus.READY:
        raise SecretInputError(
            f"Dependency {dependency_id} has no resumable work item"
        )
    handle = run.secret_handle_for(dependency)
    if handle is None or handle.status is not SecretHandleStatus.REQUESTED:
        raise SecretInputError(f"Dependency {dependency_id} has no requested Secret handle")

    writer.put(handle.secret_handle, raw_value)
    handle.transition(SecretHandleStatus.AVAILABLE)
    dependency.transition(DependencyStatus.SATISFIED)
    dependency.resolved_at = utc_now()
    run.transition(RunStatus.RUNNING)
    session.messages.append(
        ModelMessage(
            role="user",
            content=(
                {
                    "type": "text",
                    "text": (
                        "A requested Secret dependency is now available through its "
                        "opaque handle. The Secret value is not included here.\n"
                        f"Dependency ID: {dependency.dependency_id}\n"
                        f"Secret handle: {handle.secret_handle}\n"
                        f"Purpose: {handle.purpose}\n"
                        f"Consumer: {handle.consumer}\n"
                        f"Environment variable: {handle.environment_variable}\n"
                        "Continue the associated work using the designated consumer."
                    ),
                },
            ),
            run_id=run.run_id,
        )
    )


def reissue_lost_secret_dependencies(
    run: Run, availability: SecretAvailability
) -> bool:
    """Replace persisted AVAILABLE handles whose process-local values were lost."""
    if run.status not in {RunStatus.RUNNING, RunStatus.WAITING}:
        return False
    changed = False
    for handle in tuple(run.secret_handles):
        if handle.status is not SecretHandleStatus.AVAILABLE:
            continue
        if availability.contains_secret(handle.secret_handle):
            continue
        dependency = run.dependency_by_id(handle.dependency_id)
        if dependency is None:
            raise SecretInputError(
                f"Secret handle {handle.secret_handle} has no dependency"
            )
        work_item = run.work_item_by_id(dependency.work_id)
        if work_item is None or work_item.status is not WorkItemStatus.READY:
            raise SecretInputError(
                f"Secret handle {handle.secret_handle} has no resumable work item"
            )

        handle.transition(SecretHandleStatus.LOST)
        dependency.transition(DependencyStatus.LOST)
        replacement_dependency = Dependency(
            work_id=work_item.work_id,
            tool_call_id=dependency.tool_call_id,
            prompt=dependency.prompt,
            kind=DependencyKind.SECRET,
        )
        replacement_handle = SecretHandle(
            dependency_id=replacement_dependency.dependency_id,
            purpose=handle.purpose,
            consumer=handle.consumer,
            environment_variable=handle.environment_variable,
        )
        run.add_dependency(work_item, replacement_dependency)
        run.add_secret_handle(replacement_dependency, replacement_handle)
        changed = True

    has_runnable_work = any(
        work_item.status is WorkItemStatus.READY
        and not any(
            dependency.status is DependencyStatus.OPEN
            for dependency in run.dependencies_for(work_item)
        )
        for work_item in run.work_items
    )
    if changed and run.status is RunStatus.RUNNING and not has_runnable_work:
        run.transition(RunStatus.WAITING)
    return changed
