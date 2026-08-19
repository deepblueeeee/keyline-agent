"""Run creation and ordering within a durable session."""

from __future__ import annotations

from collections.abc import Sequence

from handshake.runtime.errors import AgentLoopError
from handshake.runtime.state import Run, RunStatus, Session

TERMINAL_RUN_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


def create_initial_run(
    session: Session,
    goal: str,
    *,
    model_name: str,
    max_steps: int,
) -> Run:
    """Create the first run in a session without persisting it."""
    _validate_goal(goal)
    return Run(
        goal=goal.strip(),
        session_id=session.session_id,
        sequence=1,
        model_name=model_name,
        max_steps=max_steps,
    )


def create_followup_run(
    session: Session,
    runs: Sequence[Run],
    goal: str,
) -> Run:
    """Create a run after the current session tail has terminated."""
    _validate_goal(goal)
    latest = _latest_run(session, runs)
    if latest.status not in TERMINAL_RUN_STATUSES:
        raise AgentLoopError(
            f"Session {session.session_id} already has an active run "
            f"{latest.run_id} ({latest.status.value})"
        )
    return Run(
        goal=goal.strip(),
        session_id=session.session_id,
        sequence=latest.sequence + 1,
        previous_run_id=latest.run_id,
        model_name=latest.model_name,
        max_steps=latest.max_steps,
    )


def latest_run(session: Session, runs: Sequence[Run]) -> Run:
    """Return the current tail run for a non-empty session."""
    return _latest_run(session, runs)


def _latest_run(session: Session, runs: Sequence[Run]) -> Run:
    _validate_session_runs(session, runs)
    if not runs:
        raise AgentLoopError(f"Session {session.session_id} has no runs")
    return max(runs, key=lambda item: item.sequence)


def _validate_session_runs(session: Session, runs: Sequence[Run]) -> None:
    if any(run.session_id != session.session_id for run in runs):
        raise AgentLoopError(f"Runs do not all belong to session {session.session_id}")
    sequences = [run.sequence for run in runs]
    if len(sequences) != len(set(sequences)):
        raise AgentLoopError(f"Session {session.session_id} contains duplicate run sequences")


def _validate_goal(goal: str) -> None:
    if not goal.strip():
        raise AgentLoopError("Run goal must not be empty")
