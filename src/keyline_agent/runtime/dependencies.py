"""Runtime resolution of non-sensitive dependencies."""

from __future__ import annotations

from keyline_agent.models.protocol import ModelMessage
from keyline_agent.runtime.errors import WaitResolutionError
from keyline_agent.runtime.state import (
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    Session,
    utc_now,
)


def resolve_text_dependency(
    session: Session, run: Run, dependency_id: str, response: str
) -> ModelMessage:
    """Satisfy one text dependency and return its uncommitted continuation."""
    if run.session_id != session.session_id:
        raise WaitResolutionError(f"Run {run.run_id} does not belong to the session")
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
                    "A requested text dependency has been resolved.\n"
                    f"Dependency ID: {dependency.dependency_id}\n"
                    f"Question: {dependency.prompt}\n"
                    f"User response: {normalized}\n"
                    "Continue the run using this information."
                ),
            },
        ),
        run_id=run.run_id,
    )
