import sqlite3
from pathlib import Path

import pytest

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import (
    ConcurrentRunUpdateError,
    ConcurrentSessionUpdateError,
    RunNotFoundError,
    StorageError,
)
from handshake.runtime.state import (
    Dependency,
    Run,
    RunStatus,
    Session,
    WorkItemStatus,
)
from handshake.storage.sqlite import SQLiteStateStore


def _waiting_run(session: Session) -> Run:
    run = Run(
        goal="选择发布颜色",
        session_id=session.session_id,
        model_name="deepseek-v4-flash",
        max_steps=7,
        model_turns=2,
    )
    run.transition(RunStatus.RUNNING)
    work = run.work_items[0]
    work.transition(WorkItemStatus.RUNNING)
    session.messages.extend(
        [
            ModelMessage(
                role="user",
                content=({"type": "text", "text": run.goal},),
                run_id=run.run_id,
            ),
            ModelMessage(
                role="assistant",
                content=(
                    {"type": "thinking", "thinking": "需要用户决定"},
                    {
                        "type": "tool_use",
                        "id": "ask-1",
                        "name": "wait_for_text",
                        "input": {"prompt": "发布按钮使用什么颜色？"},
                    },
                ),
                run_id=run.run_id,
            ),
            ModelMessage(
                role="user",
                content=(
                    {
                        "type": "tool_result",
                        "tool_use_id": "ask-1",
                        "content": "dependency registered",
                        "is_error": False,
                    },
                ),
                run_id=run.run_id,
            ),
        ]
    )
    run.add_dependency(
        work,
        Dependency(
            work_id=work.work_id,
            tool_call_id="ask-1",
            prompt="发布按钮使用什么颜色？",
        ),
    )
    work.transition(WorkItemStatus.READY)
    run.transition(RunStatus.WAITING)
    return run


def test_schema_initialization_and_session_run_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "state" / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run = _waiting_run(session)
    store.create_run(session, run)

    reopened = SQLiteStateStore(database)
    loaded_session = reopened.load_session(session.session_id)
    loaded_run = reopened.load_run(run.run_id)

    assert loaded_run.goal == "选择发布颜色"
    assert loaded_run.status is RunStatus.WAITING
    assert loaded_run.model_turns == 2
    assert loaded_run.max_steps == 7
    assert loaded_session.messages[1].content[0]["thinking"] == "需要用户决定"
    assert loaded_session.messages[1].run_id == run.run_id
    assert loaded_run.dependencies[0].tool_call_id == "ask-1"
    assert loaded_run.work_items[0].status is WorkItemStatus.READY
    assert loaded_run.session_id == session.session_id
    with sqlite3.connect(database) as connection:
        dependency_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(dependencies)")
        }
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "response_text" not in dependency_columns
    assert "dependencies" in tables
    assert "wait_conditions" not in tables

    with pytest.raises(RunNotFoundError):
        store.load_run("missing")


def test_incompatible_development_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 5")

    with pytest.raises(StorageError, match="incompatible with current development schema"):
        SQLiteStateStore(database)


def test_environment_api_key_is_not_written_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary-key-that-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    database = tmp_path / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    store.create_run(session, _waiting_run(session))

    assert secret.encode() not in database.read_bytes()


def test_stale_session_checkpoint_cannot_overwrite_conversation(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run = _waiting_run(session)
    store.create_run(session, run)

    current_session = store.load_session(session.session_id)
    stale_session = store.load_session(session.session_id)
    current_run = store.load_run(run.run_id)
    stale_run = store.load_run(run.run_id)
    current_session.messages.append(
        ModelMessage(
            role="user",
            content=({"type": "text", "text": "newer message"},),
            run_id=run.run_id,
        )
    )
    current_run.plan = ["newer plan"]
    store.checkpoint_run(current_session, current_run)

    stale_session.messages.append(
        ModelMessage(
            role="user",
            content=({"type": "text", "text": "stale message"},),
            run_id=run.run_id,
        )
    )
    stale_run.plan = ["stale plan"]
    with pytest.raises(ConcurrentSessionUpdateError):
        store.checkpoint_run(stale_session, stale_run)

    persisted_session = store.load_session(session.session_id)
    persisted_run = store.load_run(run.run_id)
    assert persisted_run.plan == ["newer plan"]
    assert persisted_session.messages[-1].content[0]["text"] == "newer message"


def test_stale_run_checkpoint_rolls_back_session_version(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run = _waiting_run(session)
    store.create_run(session, run)
    stale_run = store.load_run(run.run_id)

    current_session = store.load_session(session.session_id)
    current_run = store.load_run(run.run_id)
    current_run.plan = ["current"]
    store.checkpoint_run(current_session, current_run)

    fresh_session = store.load_session(session.session_id)
    original_session_version = fresh_session.version
    stale_run.plan = ["stale"]
    with pytest.raises(ConcurrentRunUpdateError):
        store.checkpoint_run(fresh_session, stale_run)

    assert store.load_session(session.session_id).version == original_session_version
    assert store.load_run(run.run_id).plan == ["current"]


def test_run_goal_cannot_change_after_creation(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run = _waiting_run(session)
    store.create_run(session, run)
    loaded_session = store.load_session(session.session_id)
    loaded_run = store.load_run(run.run_id)
    loaded_run.goal = "different goal"

    with pytest.raises(StorageError, match="invalid session membership"):
        store.checkpoint_run(loaded_session, loaded_run)

    assert store.load_run(run.run_id).goal == "选择发布颜色"
