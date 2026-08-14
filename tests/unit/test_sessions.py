from pathlib import Path

import pytest

from handshake.runtime.errors import AgentLoopError, ConcurrentSessionUpdateError
from handshake.runtime.sessions import (
    build_session_history,
    create_followup_task,
    create_initial_task,
)
from handshake.runtime.state import ModelMessage, Session, Task, TaskStatus
from handshake.storage.sqlite import SQLiteStateStore


def _completed_initial_task(session: Session) -> Task:
    task = create_initial_task(
        session,
        "first goal",
        model_name="deepseek-v4-pro",
        max_steps=17,
    )
    task.transition(TaskStatus.RUNNING)
    task.final_summary = "first result"
    task.transition(TaskStatus.COMPLETED)
    return task


def test_followup_task_inherits_config_and_history_uses_only_outcome(
    tmp_path: Path,
) -> None:
    session = Session(workspace=tmp_path)
    first = _completed_initial_task(session)
    first.messages.append(  # deliberately excluded from cross-task history
        ModelMessage(
            role="assistant",
            content=({"type": "thinking", "thinking": "private protocol"},),
        )
    )

    second = create_followup_task(session, [first], "second goal")
    history = build_session_history(session, [first], before_sequence=second.sequence)

    assert second.sequence == 2
    assert second.previous_task_id == first.task_id
    assert second.model_name == "deepseek-v4-pro"
    assert second.max_steps == 17
    assert second.model_turns == 0
    assert history[0].goal == "first goal"
    assert history[0].outcome == "first result"
    assert "private protocol" not in repr(history)


def test_active_task_blocks_followup(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    active = create_initial_task(
        session,
        "active goal",
        model_name="deepseek-v4-flash",
        max_steps=50,
    )
    active.transition(TaskStatus.RUNNING)

    with pytest.raises(AgentLoopError, match="already has an active task"):
        create_followup_task(session, [active], "must wait")


@pytest.mark.parametrize("status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
def test_failed_or_cancelled_task_allows_followup(
    tmp_path: Path, status: TaskStatus
) -> None:
    session = Session(workspace=tmp_path)
    task = create_initial_task(
        session,
        "first attempt",
        model_name="deepseek-v4-flash",
        max_steps=50,
    )
    task.transition(TaskStatus.RUNNING)
    task.error = "stopped"
    task.transition(status)

    followup = create_followup_task(session, [task], "try another approach")

    assert followup.previous_task_id == task.task_id
    assert followup.sequence == 2


def test_concurrent_followup_append_rejects_stale_session_tail(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    first = _completed_initial_task(session)
    store.create_task(first)
    first_snapshot = store.load_task(first.task_id)
    candidate_a = create_followup_task(session, [first_snapshot], "candidate A")
    candidate_b = create_followup_task(session, [first_snapshot], "candidate B")

    store.create_task(candidate_a)

    with pytest.raises(ConcurrentSessionUpdateError, match="changed before task"):
        store.create_task(candidate_b)

    tasks = store.list_tasks(session.session_id)
    assert [task.goal for task in tasks] == ["first goal", "candidate A"]
