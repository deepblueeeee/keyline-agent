import sqlite3
from pathlib import Path

import pytest

from handshake.context import AgentContextManager, ContextCheckpoint
from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import AgentLoopError
from handshake.runtime.persistence import RunStore
from handshake.runtime.runs import create_followup_run
from handshake.runtime.state import Run, RunStatus, Session
from handshake.storage.errors import StorageError
from tests.history_support import fresh_store


def _message(run: Run, text: str) -> ModelMessage:
    return ModelMessage("user", ({"type": "text", "text": text},), run.run_id)


def _restore(manager: AgentContextManager, store: RunStore, run: Run) -> None:
    manager.restore(
        store.load_context_scope(run.context_scope_id),
        store.load_context_checkpoint(run.context_scope_id),
        store.message_count(run.context_scope_id),
        lambda start, end: store.read_messages(run.context_scope_id, start, end),
    )


def test_child_run_has_empty_private_history_and_checkpoint(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    parent = Run(goal="parent goal", session_id=session.session_id, agent_spec_id="planner")
    store.create_run(session, parent, (_message(parent, "parent private history"),))
    parent_checkpoint = store.save_context_checkpoint(
        parent.context_scope_id,
        ContextCheckpoint(source_message_count=1),
    )
    child = Run(
        goal="child goal",
        session_id=session.session_id,
        sequence=2,
        parent_run_id=parent.run_id,
        agent_spec_id="reader",
    )
    store.create_run(session, child, (_message(child, "child goal"),))

    assert child.previous_run_id is None
    assert store.load_context_scope(child.context_scope_id).is_tail(child)
    assert store.load_context_scope(parent.context_scope_id).is_tail(parent)
    assert store.load_session(session.session_id).context_scope.tail_run_id == parent.run_id
    assert store.load_run(child.run_id).parent_run_id == parent.run_id
    assert store.load_run(child.run_id).agent_spec_id == "reader"
    assert store.message_count(child.context_scope_id) == 1
    assert store.load_context_checkpoint(child.context_scope_id) is None
    child_messages = tuple(store.read_messages(child.context_scope_id))
    assert [item.message.content[0]["text"] for item in child_messages] == [
        "child goal"
    ]

    store.checkpoint_run(session, parent, (_message(parent, "parent continued"),))
    store.checkpoint_run(session, child, (_message(child, "child continued"),))
    child_checkpoint = store.save_context_checkpoint(
        child.context_scope_id,
        ContextCheckpoint(source_message_count=2),
    )
    assert store.message_count(parent.context_scope_id) == 2
    assert store.message_count(child.context_scope_id) == 2
    assert store.load_context_checkpoint(parent.context_scope_id) == parent_checkpoint
    assert store.load_context_checkpoint(child.context_scope_id) == child_checkpoint

    manager = AgentContextManager()
    _restore(manager, store, parent)
    _restore(manager, store, child)
    assert manager.is_initialized(store.load_context_scope(parent.context_scope_id))
    assert manager.is_initialized(store.load_context_scope(child.context_scope_id))
    assert [item.content[0]["text"] for item in manager.build(session, parent).messages] == [
        "parent private history",
        "parent continued",
    ]
    assert [item.content[0]["text"] for item in manager.build(session, child).messages] == [
        "child goal",
        "child continued",
    ]


def test_followup_reuses_main_context_scope_without_copying(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    first = Run(goal="first", session_id=session.session_id, agent_spec_id="worker")
    first.transition(RunStatus.RUNNING)
    first.transition(RunStatus.COMPLETED)
    store.create_run(session, first, (_message(first, "first history"),))
    checkpoint = store.save_context_checkpoint(
        first.context_scope_id,
        ContextCheckpoint(source_message_count=1),
    )
    second = Run(
        goal="second",
        session_id=session.session_id,
        sequence=2,
        previous_run_id=first.run_id,
        agent_spec_id="worker",
    )
    store.create_run(session, second, (_message(second, "second goal"),))

    assert first.context_scope_id == second.context_scope_id == session.context_scope.scope_id
    assert store.load_run(second.run_id).previous_run_id == first.run_id
    assert store.load_context_scope(second.context_scope_id) == session.context_scope
    assert store.load_session(session.session_id).context_scope.tail_run_id == second.run_id
    assert store.message_count(first.context_scope_id) == 2
    assert store.message_count(second.context_scope_id) == 2
    assert store.load_context_checkpoint(second.context_scope_id) == checkpoint
    assert [item.message.run_id for item in store.read_messages(second.context_scope_id)] == [
        first.run_id,
        second.run_id,
    ]
    store.checkpoint_run(session, second, (_message(second, "second continued"),))
    assert store.message_count(first.context_scope_id) == 3
    assert store.message_count(second.context_scope_id) == 3
    assert store.load_context_checkpoint(first.context_scope_id) == checkpoint
    with pytest.raises(StorageError, match="tail run"):
        store.checkpoint_run(session, first, (_message(first, "late write"),))
    assert store.has_run_messages(first.run_id)
    assert store.has_run_messages(second.run_id)
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM context_scopes").fetchone()[0] == 1

    manager = AgentContextManager()
    _restore(manager, store, first)
    assert manager.is_initialized(store.load_context_scope(second.context_scope_id))
    assert manager.message_count(store.load_context_scope(second.context_scope_id)) == 3
    assert [item.content[0]["text"] for item in manager.build(session, second).messages] == [
        "first history",
        "second goal",
        "second continued",
    ]


def test_wrong_predecessor_does_not_advance_scope_or_write_messages(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    first = Run(goal="first", session_id=session.session_id)
    store.create_run(session, first, (_message(first, "first"),))
    wrong = Run(
        goal="wrong",
        session_id=session.session_id,
        sequence=2,
        previous_run_id="some-other-run",
    )

    with pytest.raises(StorageError, match="context scope tail"):
        store.create_run(session, wrong, (_message(wrong, "wrong"),))

    assert store.load_context_scope(session.context_scope.scope_id).tail_run_id == first.run_id
    assert store.list_runs(session.session_id) == (store.load_run(first.run_id),)
    assert store.message_count(session.context_scope.scope_id) == 1


def test_child_must_refer_to_parent_in_same_session(tmp_path: Path) -> None:
    first_session = Session(workspace=tmp_path)
    second_session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, first_session)
    store.create_session(second_session)
    parent = Run(goal="parent", session_id=first_session.session_id)
    store.create_run(first_session, parent)
    child = Run(
        goal="child",
        session_id=second_session.session_id,
        parent_run_id=parent.run_id,
    )
    with pytest.raises(StorageError, match="Parent run must belong"):
        store.create_run(second_session, child)
    assert store.list_runs(second_session.session_id) == ()


def test_user_followup_after_child_copies_root_not_child(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    root = Run(goal="root", session_id=session.session_id, agent_spec_id="planner")
    root.transition(RunStatus.RUNNING)
    root.transition(RunStatus.COMPLETED)
    store.create_run(session, root, (_message(root, "root history"),))
    child = Run(
        goal="child",
        session_id=session.session_id,
        sequence=2,
        parent_run_id=root.run_id,
        agent_spec_id="reader",
    )
    child.transition(RunStatus.RUNNING)
    child.transition(RunStatus.COMPLETED)
    store.create_run(session, child, (_message(child, "child private history"),))

    followup = create_followup_run(session, store.list_runs(session.session_id), "next goal")
    assert followup.context_scope_id == root.context_scope_id
    assert followup.previous_run_id == root.run_id
    assert followup.agent_spec_id == "planner"
    store.create_run(session, followup, (_message(followup, "next goal"),))
    messages = tuple(store.read_messages(followup.context_scope_id))
    assert [item.message.content[0]["text"] for item in messages] == [
        "root history",
        "next goal",
    ]


def test_user_followup_waits_for_active_parent_even_if_child_terminated(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    root = Run(goal="root", session_id=session.session_id)
    root.transition(RunStatus.RUNNING)
    store.create_run(session, root)
    child = Run(
        goal="child",
        session_id=session.session_id,
        sequence=2,
        parent_run_id=root.run_id,
    )
    child.transition(RunStatus.RUNNING)
    child.transition(RunStatus.COMPLETED)
    store.create_run(session, child)
    with pytest.raises(AgentLoopError, match="active run"):
        create_followup_run(session, store.list_runs(session.session_id), "too early")
