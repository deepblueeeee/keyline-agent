"""Session-level task creation and sanitized cross-task context."""

from __future__ import annotations

from collections.abc import Sequence

from handshake.runtime.errors import AgentLoopError
from handshake.runtime.state import (
    Session,
    SessionHistoryEntry,
    Task,
    TaskStatus,
)

_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
}


def create_initial_task(
    session: Session,
    goal: str,
    *,
    model_name: str,
    max_steps: int,
) -> Task:
    """Create the first task in a session without persisting it."""
    _validate_goal(goal)
    return Task(
        goal=goal,
        workspace=session.workspace,
        session_id=session.session_id,
        sequence=1,
        model_name=model_name,
        max_steps=max_steps,
    )


def create_followup_task(
    session: Session,
    tasks: Sequence[Task],
    goal: str,
) -> Task:
    """Create the next task after the current session tail has terminated."""
    _validate_goal(goal)
    latest = _latest_task(session, tasks)
    if latest.status not in _TERMINAL_TASK_STATUSES:
        raise AgentLoopError(
            f"Session {session.session_id} already has an active task "
            f"{latest.task_id} ({latest.status.value})"
        )
    return Task(
        goal=goal,
        workspace=session.workspace,
        session_id=session.session_id,
        sequence=latest.sequence + 1,
        previous_task_id=latest.task_id,
        model_name=latest.model_name,
        max_steps=latest.max_steps,
    )


def build_session_history(
    session: Session,
    tasks: Sequence[Task],
    *,
    before_sequence: int,
) -> tuple[SessionHistoryEntry, ...]:
    """Return sanitized terminal task outcomes preceding one task."""
    _validate_session_tasks(session, tasks)
    history: list[SessionHistoryEntry] = []
    for task in sorted(tasks, key=lambda item: item.sequence):
        if task.sequence >= before_sequence:
            continue
        if task.status not in _TERMINAL_TASK_STATUSES:
            raise AgentLoopError(
                f"Prior task {task.task_id} is not terminal ({task.status.value})"
            )
        outcome = task.final_summary if task.status is TaskStatus.COMPLETED else task.error
        history.append(
            SessionHistoryEntry(
                sequence=task.sequence,
                goal=task.goal,
                status=task.status,
                outcome=outcome,
            )
        )
    return tuple(history)


def latest_task(session: Session, tasks: Sequence[Task]) -> Task:
    """Return the current tail task for a non-empty session."""
    return _latest_task(session, tasks)


def _latest_task(session: Session, tasks: Sequence[Task]) -> Task:
    _validate_session_tasks(session, tasks)
    if not tasks:
        raise AgentLoopError(f"Session {session.session_id} has no tasks")
    return max(tasks, key=lambda item: item.sequence)


def _validate_session_tasks(session: Session, tasks: Sequence[Task]) -> None:
    if any(task.session_id != session.session_id for task in tasks):
        raise AgentLoopError(f"Tasks do not all belong to session {session.session_id}")
    sequences = [task.sequence for task in tasks]
    if len(sequences) != len(set(sequences)):
        raise AgentLoopError(f"Session {session.session_id} contains duplicate task sequences")


def _validate_goal(goal: str) -> None:
    if not goal.strip():
        raise AgentLoopError("Task goal must not be empty")
