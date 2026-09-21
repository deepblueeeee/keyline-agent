import sqlite3
from pathlib import Path

import pytest

from handshake.models.protocol import ModelMessage
from handshake.runtime.state import (
    Dependency,
    Run,
    RunStatus,
    Session,
    WorkItemStatus,
)
from handshake.storage.errors import RunNotFoundError, StorageError
from handshake.storage.sqlite import SQLiteStateStore


def _waiting_run(session: Session) -> tuple[Run, list[ModelMessage]]:
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
    messages = [
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
    return run, messages


def test_schema_initialization_and_session_run_round_trip(tmp_path: Path) -> None:
    database = tmp_path / "state" / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run, messages = _waiting_run(session)
    store.create_run(session, run, tuple(messages))

    reopened = SQLiteStateStore(database)
    loaded_session = reopened.load_session(session.session_id)
    loaded_run = reopened.load_run(run.run_id)

    assert loaded_run.goal == "选择发布颜色"
    assert loaded_run.status is RunStatus.WAITING
    assert loaded_run.model_turns == 2
    assert loaded_run.max_steps == 7
    loaded_messages = tuple(store.read_messages(session.session_id))
    assert loaded_messages[1].message.content[0]["thinking"] == "需要用户决定"
    assert loaded_messages[1].message.run_id == run.run_id
    assert loaded_run.dependencies[0].tool_call_id == "ask-1"
    assert loaded_run.work_items[0].status is WorkItemStatus.READY
    assert loaded_run.session_id == session.session_id
    assert not hasattr(loaded_session, "messages")
    with sqlite3.connect(database) as connection:
        session_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")
        }
        run_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")
        }
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
    assert "version" not in session_columns
    assert "version" not in run_columns
    assert not hasattr(loaded_session, "version")
    assert not hasattr(loaded_run, "version")
    assert "dependencies" in tables
    assert "wait_conditions" not in tables

    with pytest.raises(RunNotFoundError):
        store.load_run("missing")


def test_messages_are_append_only_and_support_range_queries(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run, messages = _waiting_run(session)
    store.create_run(session, run)

    first = tuple(messages[:2])
    first_appended = store.append_messages(session.session_id, first)
    with sqlite3.connect(database) as connection:
        original_rows = connection.execute(
            "SELECT sequence, content_json FROM messages ORDER BY sequence"
        ).fetchall()

    second_appended = store.append_messages(session.session_id, (messages[2],))

    assert [item.sequence for item in first_appended] == [0, 1]
    assert [item.message for item in first_appended] == list(first)
    assert [item.sequence for item in second_appended] == [2]
    assert second_appended[0].message == messages[2]
    assert store.message_count(session.session_id) == 3
    assert store.message_count(session.session_id, run.run_id) == 3
    assert store.has_run_messages(session.session_id, run.run_id)
    assert not store.has_run_messages(session.session_id, "missing-run")
    ranged = tuple(store.read_messages(session.session_id, 1, 3))
    assert [item.sequence for item in ranged] == [1, 2]
    assert [item.message for item in ranged] == messages[1:]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT sequence, content_json FROM messages WHERE sequence < 2 ORDER BY sequence"
        ).fetchall() == original_rows


def test_incompatible_development_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 7")

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
    run, messages = _waiting_run(session)
    store.create_run(session, run, tuple(messages))

    assert secret.encode() not in database.read_bytes()


def test_checkpoint_failure_rolls_back_the_complete_snapshot(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run, messages = _waiting_run(session)
    store.create_run(session, run, tuple(messages))

    loaded_session = store.load_session(session.session_id)
    loaded_run = store.load_run(run.run_id)
    original_messages = tuple(store.read_messages(session.session_id))
    loaded_run.plan = ["must roll back"]
    pending = (
        ModelMessage(
            role="user",
            content=({"type": "text", "text": "invalid foreign key"},),
            run_id="missing-run",
        ),
    )
    with pytest.raises(StorageError, match="SQLite state storage failed"):
        store.checkpoint_run(loaded_session, loaded_run, pending)

    assert tuple(store.read_messages(session.session_id)) == original_messages
    assert store.load_run(run.run_id).plan == []


def test_run_goal_cannot_change_after_creation(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run, messages = _waiting_run(session)
    store.create_run(session, run, tuple(messages))
    loaded_session = store.load_session(session.session_id)
    loaded_run = store.load_run(run.run_id)
    loaded_run.goal = "different goal"

    with pytest.raises(StorageError, match="invalid session membership"):
        store.checkpoint_run(loaded_session, loaded_run)

    assert store.load_run(run.run_id).goal == "选择发布颜色"
