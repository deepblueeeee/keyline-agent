"""Runtime transitions for out-of-band Secret dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from handshake.inputs import SecretWriter
from handshake.models.protocol import ModelMessage, ToolCall
from handshake.runtime.errors import SecretInputError
from handshake.runtime.state import (
    Dependency,
    DependencyKind,
    DependencyStatus,
    ReplayPolicy,
    Run,
    RunStatus,
    SecretHandle,
    SecretHandleStatus,
    Session,
    ToolRunStatus,
    utc_now,
)


class SecretAvailability(Protocol):
    def contains_secret(self, secret_handle: str) -> bool:
        """Return whether a raw value is available in this process."""
        ...


def _reissue_request(run: Run, handle: SecretHandle, *, keep_handle: bool) -> None:
    dependency = run.dependency_by_id(handle.dependency_id)
    if dependency is None:
        raise SecretInputError(f"Secret handle {handle.secret_handle} has no dependency")
    dependency.transition(DependencyStatus.LOST)
    replacement = Dependency(
        tool_call_id=dependency.tool_call_id,
        prompt=dependency.prompt,
        kind=DependencyKind.SECRET,
    )
    run.add_dependency(replacement)
    if keep_handle:
        handle.transition(SecretHandleStatus.REQUESTED)
        handle.dependency_id = replacement.dependency_id
    else:
        handle.transition(SecretHandleStatus.LOST)
        run.add_secret_handle(replacement, SecretHandle(
            dependency_id=replacement.dependency_id,
            purpose=handle.purpose,
            consumer=handle.consumer,
            environment_variable=handle.environment_variable,
        ))


def request_pending_tool_secrets(
    run: Run, calls: Sequence[ToolCall], availability: SecretAvailability
) -> bool:
    """Ask again for Secrets used by commands that have never started."""
    changed = False
    tools = {
        tool.tool_call_id: tool for tool in run.tool_runs if tool.model_turn == run.model_turns
    }
    for call in calls:
        tool = tools.get(call.id)
        if tool is not None:
            if (
                tool.status in {ToolRunStatus.RUNNING, ToolRunStatus.UNKNOWN}
                and tool.replay_policy is ReplayPolicy.MANUAL
            ):
                break
            if tool.status is not ToolRunStatus.PENDING:
                continue
        if call.name != "run_command_with_secret":
            continue
        handle_id = call.arguments.get("secret_handle")
        if not isinstance(handle_id, str):
            continue
        handle = run.secret_handle_by_id(handle_id)
        if (
            handle is not None
            and handle.status is SecretHandleStatus.AVAILABLE
            and not availability.contains_secret(handle_id)
        ):
            # The assistant tool_use still refers to this handle.
            _reissue_request(run, handle, keep_handle=True)
            changed = True
    if changed:
        run.transition(RunStatus.WAITING)
    return changed


def resolve_secret_dependency(
    session: Session,
    run: Run,
    dependency_id: str,
    raw_value: str,
    writer: SecretWriter,
) -> ModelMessage:
    """Route a Secret and return an opaque, uncommitted continuation."""
    if run.session_id != session.session_id:
        raise SecretInputError(f"Run {run.run_id} does not belong to the session")
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
    handle = run.secret_handle_for(dependency)
    if handle is None or handle.status is not SecretHandleStatus.REQUESTED:
        raise SecretInputError(f"Dependency {dependency_id} has no requested Secret handle")

    writer.put(handle.secret_handle, raw_value)
    handle.transition(SecretHandleStatus.AVAILABLE)
    dependency.transition(DependencyStatus.SATISFIED)
    dependency.resolved_at = utc_now()
    if not run.open_dependencies():
        run.transition(RunStatus.RUNNING)
    return ModelMessage(
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
                    "Continue the run using the designated consumer."
                ),
            },
        ),
        run_id=run.run_id,
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
        _reissue_request(run, handle, keep_handle=False)
        changed = True

    if changed and run.status is RunStatus.RUNNING:
        run.transition(RunStatus.WAITING)
    return changed
