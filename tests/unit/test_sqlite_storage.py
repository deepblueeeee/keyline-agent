import sqlite3
from pathlib import Path

import pytest

from handshake.runtime.errors import ConcurrentTaskUpdateError, TaskNotFoundError
from handshake.runtime.state import (
    ModelMessage,
    Task,
    TaskStatus,
    WaitCondition,
    WorkItem,
)
from handshake.storage.sqlite import SQLiteTaskStore


def _waiting_task(tmp_path: Path) -> Task:
    task = Task(
        goal="选择发布颜色",
        workspace=tmp_path,
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


def test_migration_is_idempotent_and_task_round_trips(tmp_path: Path) -> None:
    database = tmp_path / "state" / "handshake.db"
    store = SQLiteTaskStore(database)
    task = _waiting_task(tmp_path)
    store.create_task(task)

    loaded = SQLiteTaskStore(database).load_task(task.task_id)

    assert loaded.goal == "选择发布颜色"
    assert loaded.status is TaskStatus.WAITING
    assert loaded.model_turns == 2
    assert loaded.max_steps == 7
    assert loaded.messages[1].content[0]["thinking"] == "需要用户决定"
    assert loaded.waits[0].tool_call_id == "ask-1"

    with pytest.raises(TaskNotFoundError):
        store.load_task("missing")


def test_v1_database_migrates_to_optimistic_lock_version(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 1")

    SQLiteTaskStore(database)
    SQLiteTaskStore(database)

    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(tasks)")
        }
    assert schema_version == 2
    assert columns["version"][3] == 1
    assert columns["version"][4] == "0"


def test_environment_api_key_is_not_written_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary-key-that-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    database = tmp_path / "handshake.db"
    store = SQLiteTaskStore(database)
    store.create_task(_waiting_task(tmp_path))

    assert secret.encode() not in database.read_bytes()


def test_stale_checkpoint_is_rejected_without_overwriting_newer_state(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    task = _waiting_task(tmp_path)
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
