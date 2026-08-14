import sqlite3
from pathlib import Path

import pytest

from handshake.runtime.errors import (
    ConcurrentTaskUpdateError,
    StorageError,
    TaskNotFoundError,
)
from handshake.runtime.state import (
    ModelMessage,
    Session,
    Task,
    TaskStatus,
    WaitCondition,
    WorkItem,
)
from handshake.storage.sqlite import SQLiteStateStore


def _waiting_task(tmp_path: Path, session_id: str) -> Task:
    task = Task(
        goal="选择发布颜色",
        workspace=tmp_path,
        session_id=session_id,
        model_name="deepseek-v4-flash",
        max_steps=7,
        model_turns=2,
    )
    task.transition(TaskStatus.RUNNING)
    task.messages.extend(
        [
            ModelMessage(role="user", content=({"type": "text", "text": task.goal},)),
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
            ),
        ]
    )
    work = WorkItem(description="选择颜色", tool_call_id="ask-1")
    task.work_items.append(work)
    task.waits.append(
        WaitCondition(
            work_id=work.work_id,
            tool_call_id="ask-1",
            prompt="发布按钮使用什么颜色？",
        )
    )
    task.transition(TaskStatus.WAITING)
    return task


def test_schema_initialization_is_idempotent_and_task_round_trips(tmp_path: Path) -> None:
    database = tmp_path / "state" / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    task = _waiting_task(tmp_path, session.session_id)
    store.create_task(task)

    loaded = SQLiteStateStore(database).load_task(task.task_id)

    assert loaded.goal == "选择发布颜色"
    assert loaded.status is TaskStatus.WAITING
    assert loaded.model_turns == 2
    assert loaded.max_steps == 7
    assert loaded.messages[1].content[0]["thinking"] == "需要用户决定"
    assert loaded.waits[0].tool_call_id == "ask-1"
    assert loaded.session_id == session.session_id

    with pytest.raises(TaskNotFoundError):
        store.load_task("missing")


def test_incompatible_development_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 3")

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
    store.create_task(_waiting_task(tmp_path, session.session_id))

    assert secret.encode() not in database.read_bytes()


def test_stale_checkpoint_is_rejected_without_overwriting_newer_state(
    tmp_path: Path,
) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    task = _waiting_task(tmp_path, session.session_id)
    store.create_task(task)
    first = store.load_task(task.task_id)
    stale = store.load_task(task.task_id)

    first.plan = ["newer plan"]
    first.messages.append(
        ModelMessage(role="user", content=({"type": "text", "text": "newer message"},))
    )
    store.checkpoint_task(first)
    stale.plan = ["stale plan"]
    stale.messages.append(
        ModelMessage(role="user", content=({"type": "text", "text": "stale message"},))
    )

    with pytest.raises(ConcurrentTaskUpdateError, match="changed since it was loaded"):
        store.checkpoint_task(stale)

    persisted = store.load_task(task.task_id)
    assert first.version == 1
    assert stale.version == 0
    assert persisted.version == 1
    assert persisted.plan == ["newer plan"]
    assert persisted.messages[-1].content[0]["text"] == "newer message"
