from pathlib import Path

import pytest

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import AgentLoopError
from handshake.runtime.runs import create_followup_run, create_initial_run
from handshake.runtime.state import Run, RunStatus, Session, WorkItemStatus
from handshake.storage.errors import StorageError
from handshake.storage.sqlite import SQLiteStateStore


def _completed_initial_run(session: Session) -> tuple[Run, tuple[ModelMessage, ...]]:
    run = create_initial_run(
        session,
        "first goal",
        model_name="deepseek-v4-pro",
        max_steps=17,
    )
    run.transition(RunStatus.RUNNING)
    run.work_items[0].transition(WorkItemStatus.RUNNING)
    messages = (
        ModelMessage(
            role="user",
            content=({"type": "text", "text": run.goal},),
            run_id=run.run_id,
        ),
        ModelMessage(
            role="assistant",
            content=({"type": "text", "text": "first result"},),
            run_id=run.run_id,
        ),
    )
    run.final_summary = "first result"
    run.work_items[0].transition(WorkItemStatus.COMPLETED)
    run.transition(RunStatus.COMPLETED)
    return run, messages


def test_followup_run_inherits_config_without_owning_messages(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    first, messages = _completed_initial_run(session)

    second = create_followup_run(session, [first], "second goal")

    assert second.sequence == 2
    assert second.previous_run_id == first.run_id
    assert second.model_name == "deepseek-v4-pro"
    assert second.max_steps == 17
    assert second.model_turns == 0
    assert not hasattr(second, "messages")
    assert [message.run_id for message in messages] == [
        first.run_id,
        first.run_id,
    ]


def test_active_run_blocks_followup(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    active = create_initial_run(
        session,
        "active goal",
        model_name="deepseek-v4-flash",
        max_steps=50,
    )
    active.transition(RunStatus.RUNNING)
    active.work_items[0].transition(WorkItemStatus.RUNNING)

    with pytest.raises(AgentLoopError, match="already has an active run"):
        create_followup_run(session, [active], "must wait")


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.CANCELLED])
def test_failed_or_cancelled_run_allows_followup(
    tmp_path: Path, status: RunStatus
) -> None:
    session = Session(workspace=tmp_path)
    run = create_initial_run(
        session,
        "first attempt",
        model_name="deepseek-v4-flash",
        max_steps=50,
    )
    run.transition(RunStatus.RUNNING)
    run.work_items[0].transition(WorkItemStatus.RUNNING)
    run.error = "stopped"
    run.work_items[0].transition(WorkItemStatus.FAILED)
    run.transition(status)

    followup = create_followup_run(session, [run], "try another approach")

    assert followup.previous_run_id == run.run_id
    assert followup.sequence == 2


def test_competing_followup_run_cannot_fork_session_history(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    first, messages = _completed_initial_run(session)
    store.create_run(session, first, messages)

    session_a = store.load_session(session.session_id)
    session_b = store.load_session(session.session_id)
    first_snapshot = store.load_run(first.run_id)
    candidate_a = create_followup_run(session_a, [first_snapshot], "candidate A")
    candidate_b = create_followup_run(session_b, [first_snapshot], "candidate B")
    pending: dict[str, tuple[ModelMessage, ...]] = {}
    for candidate in (candidate_a, candidate_b):
        candidate.transition(RunStatus.RUNNING)
        candidate.work_items[0].transition(WorkItemStatus.RUNNING)
        pending[candidate.run_id] = (
            ModelMessage(
                role="user",
                content=({"type": "text", "text": candidate.goal},),
                run_id=candidate.run_id,
            ),
        )

    store.create_run(session_a, candidate_a, pending[candidate_a.run_id])

    with pytest.raises(StorageError, match="is not the next run"):
        store.create_run(session_b, candidate_b, pending[candidate_b.run_id])

    runs = store.list_runs(session.session_id)
    assert [run.goal for run in runs] == ["first goal", "candidate A"]
