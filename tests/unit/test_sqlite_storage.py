from pathlib import Path

import pytest

from handshake.runtime.errors import TaskNotFoundError
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


def test_environment_api_key_is_not_written_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary-key-that-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    database = tmp_path / "handshake.db"
    store = SQLiteTaskStore(database)
    store.create_task(_waiting_task(tmp_path))

    assert secret.encode() not in database.read_bytes()
