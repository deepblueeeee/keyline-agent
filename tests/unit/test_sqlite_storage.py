from pathlib import Path

import pytest

from handshake.runtime.errors import TaskNotFoundError, WaitResolutionError
from handshake.runtime.state import (
    ModelMessage,
    Task,
    TaskStatus,
    WaitCondition,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
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


def test_resolving_wait_appends_linked_tool_result_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    task = _waiting_task(tmp_path)
    store.create_task(task)
    wait_id = task.waits[0].wait_id

    resumed = store.resolve_text_wait(task.task_id, wait_id, "  深蓝色  ")

    assert resumed.status is TaskStatus.RUNNING
    assert resumed.waits[0].status is WaitStatus.SATISFIED
    assert resumed.waits[0].response_text == "深蓝色"
    assert resumed.work_items[0].status is WorkItemStatus.COMPLETED
    result = resumed.messages[-1].content[0]
    assert result["tool_use_id"] == "ask-1"
    assert "深蓝色" in result["content"]
    with pytest.raises(WaitResolutionError, match="already satisfied"):
        store.resolve_text_wait(task.task_id, wait_id, "绿色")


def test_invalid_task_wait_and_empty_response_are_rejected(tmp_path: Path) -> None:
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    task = _waiting_task(tmp_path)
    store.create_task(task)

    with pytest.raises(TaskNotFoundError):
        store.load_task("missing")
    with pytest.raises(WaitResolutionError, match="must not be empty"):
        store.resolve_text_wait(task.task_id, task.waits[0].wait_id, "  ")
    with pytest.raises(WaitResolutionError, match="does not belong"):
        store.resolve_text_wait(task.task_id, "missing", "blue")


def test_environment_api_key_is_not_written_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary-key-that-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    database = tmp_path / "handshake.db"
    store = SQLiteTaskStore(database)
    store.create_task(_waiting_task(tmp_path))

    assert secret.encode() not in database.read_bytes()
