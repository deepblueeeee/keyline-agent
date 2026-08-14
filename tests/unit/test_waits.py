from pathlib import Path

import pytest

from handshake.runtime.errors import WaitResolutionError
from handshake.runtime.state import (
    ModelMessage,
    Task,
    TaskStatus,
    WaitCondition,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
)
from handshake.runtime.waits import resolve_text_wait


def _waiting_task(tmp_path: Path) -> Task:
    task = Task(goal="选择发布颜色", workspace=tmp_path, session_id="test-session")
    task.transition(TaskStatus.RUNNING)
    task.messages.append(
        ModelMessage(
            role="assistant",
            content=(
                {
                    "type": "tool_use",
                    "id": "ask-1",
                    "name": "wait_for_text",
                    "input": {"prompt": "发布按钮使用什么颜色？"},
                },
            ),
        )
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


def test_resolving_wait_updates_runtime_state_and_links_tool_result(
    tmp_path: Path,
) -> None:
    task = _waiting_task(tmp_path)
    wait_id = task.waits[0].wait_id

    resolve_text_wait(task, wait_id, "  深蓝色  ")

    assert task.status is TaskStatus.RUNNING
    assert task.waits[0].status is WaitStatus.SATISFIED
    assert task.waits[0].response_text == "深蓝色"
    assert task.work_items[0].status is WorkItemStatus.COMPLETED
    result = task.messages[-1].content[0]
    assert result["tool_use_id"] == "ask-1"
    assert "深蓝色" in result["content"]
    with pytest.raises(WaitResolutionError, match="already satisfied"):
        resolve_text_wait(task, wait_id, "绿色")


def test_invalid_wait_and_empty_response_are_rejected(tmp_path: Path) -> None:
    task = _waiting_task(tmp_path)

    with pytest.raises(WaitResolutionError, match="must not be empty"):
        resolve_text_wait(task, task.waits[0].wait_id, "  ")
    with pytest.raises(WaitResolutionError, match="does not belong"):
        resolve_text_wait(task, "missing", "blue")
