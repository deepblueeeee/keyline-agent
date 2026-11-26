"""Run creation and ordering within a durable session."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from handshake.runtime.errors import AgentLoopError
from handshake.runtime.state import Run, RunStatus, Session

TERMINAL_RUN_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class ChildRunResult:
    """The durable outcome handed back to a delegating model turn."""

    run_id: str
    agent_spec_id: str
    status: RunStatus
    summary: str | None
    error: str | None

    @classmethod
    def from_run(cls, run: Run) -> ChildRunResult:
        if run.status not in TERMINAL_RUN_STATUSES:
            raise AgentLoopError("A child result requires a terminal run")
        return cls(run.run_id, run.agent_spec_id, run.status, run.final_summary, run.error)


def create_child_run(
    session: Session,
    runs: Sequence[Run],
    parent: Run,
    goal: str,
    *,
    agent_spec_id: str,
    model_name: str,
    max_steps: int,
    tool_call_id: str,
) -> Run:
    """Create one isolated child for a checkpointed delegation tool call."""
    _validate_goal(goal)
    _validate_session_runs(session, runs)
    if parent.session_id != session.session_id or parent.run_id not in {run.run_id for run in runs}:
        raise AgentLoopError("Delegating parent must belong to this session")
    if parent.status is not RunStatus.WAITING:
        raise AgentLoopError("Delegating parent must be waiting for its child")
    if not agent_spec_id or not tool_call_id or max_steps < 0:
        raise AgentLoopError("Child needs an agent, tool call, and non-negative step budget")
    if any(
        run.parent_run_id == parent.run_id and run.delegation_tool_call_id == tool_call_id
        for run in runs
    ):
        raise AgentLoopError("Delegation tool call already has a child run")
    return Run(
        goal=goal.strip(),
        session_id=session.session_id,
        sequence=max(run.sequence for run in runs) + 1,
        parent_run_id=parent.run_id,
        delegation_tool_call_id=tool_call_id,
        agent_spec_id=agent_spec_id,
        model_name=model_name,
        max_steps=max_steps,
    )


def create_initial_run(
    session: Session,
    goal: str,
    *,
    model_name: str,
    max_steps: int,
    agent_spec_id: str,
) -> Run:
    """Create the first run in a session without persisting it."""
    _validate_goal(goal)
    return session.context_scope.new_run(
        goal.strip(),
        sequence=1,
        model_name=model_name,
        max_steps=max_steps,
        agent_spec_id=agent_spec_id,
    )


def create_followup_run(
    session: Session,
    runs: Sequence[Run],
    goal: str,
    *,
    max_steps: int = 0,
) -> Run:
    """Create a root run after every existing run has terminated."""
    _validate_goal(goal)
    latest = _latest_run(session, runs)
    active = next((run for run in runs if run.status not in TERMINAL_RUN_STATUSES), None)
    if active is not None:
        raise AgentLoopError(
            f"Session {session.session_id} already has an active run "
            f"{active.run_id} ({active.status.value})"
        )
    previous_root = max(
        (run for run in runs if run.parent_run_id is None),
        key=lambda run: run.sequence,
    )
    if session.context_scope.tail_run_id != previous_root.run_id:
        raise AgentLoopError("Session context scope tail does not match the latest root run")
    return session.context_scope.new_run(
        goal.strip(),
        sequence=latest.sequence + 1,
        agent_spec_id=previous_root.agent_spec_id,
        model_name=previous_root.model_name,
        max_steps=max_steps,
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
