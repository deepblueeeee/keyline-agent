"""Runtime resolution of non-sensitive dependencies."""

from __future__ import annotations

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import WaitResolutionError
from handshake.runtime.state import (
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    Session,
    WorkItemStatus,
    utc_now,
)


def resolve_text_dependency(
    session: Session, run: Run, dependency_id: str, response: str
) -> None:
    """Satisfy one text dependency and append a continuation message."""
    normalized = response.strip()
    if not normalized:
        raise WaitResolutionError("Text response must not be empty")
    dependency = run.dependency_by_id(dependency_id)
    if dependency is None:
        raise WaitResolutionError(
            f"Dependency {dependency_id} does not belong to run {run.run_id}"
        )
    if dependency.kind is not DependencyKind.TEXT:
        raise WaitResolutionError(
            f"Dependency {dependency_id} does not accept text input"
        )
    if dependency.status is not DependencyStatus.OPEN:
        raise WaitResolutionError(f"Dependency {dependency_id} is already satisfied")
    if run.status is not RunStatus.WAITING:
        raise WaitResolutionError(f"Run {run.run_id} is not waiting for input")
    work_item = run.work_item_by_id(dependency.work_id)
    if work_item is None or work_item.status is not WorkItemStatus.READY:
        raise WaitResolutionError(
            f"Dependency {dependency_id} has no resumable work item"
        )

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
                        "A requested text dependency has been resolved.\n"
                        f"Dependency ID: {dependency.dependency_id}\n"
                        f"Question: {dependency.prompt}\n"
                        f"User response: {normalized}\n"
                        "Continue the associated work using this information."
                    ),
                },
            ),
            run_id=run.run_id,
        )
    )
