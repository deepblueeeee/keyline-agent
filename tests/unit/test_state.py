from pathlib import Path

import pytest

from handshake.runtime.errors import InvalidStateTransitionError
from handshake.runtime.state import (
    Task,
    TaskStatus,
    ToolRun,
    ToolRunStatus,
    WaitCondition,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
)


def test_task_accepts_only_declared_transitions(tmp_path: Path) -> None:
    task = Task(goal="test", workspace=tmp_path)
    task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        task.transition(TaskStatus.RUNNING)


def test_tool_run_accepts_only_declared_transitions() -> None:
    run = ToolRun(tool_call_id="call", tool_name="read_file", arguments={})
    run.transition(ToolRunStatus.RUNNING)
    run.transition(ToolRunStatus.FAILED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(ToolRunStatus.SUCCEEDED)


def test_waiting_task_and_bound_work_follow_declared_transitions(tmp_path: Path) -> None:
    task = Task(goal="test", workspace=tmp_path)
    work = WorkItem(description="Choose a color", tool_call_id="ask-1")
    wait = WaitCondition(
        work_id=work.work_id,
        tool_call_id="ask-1",
        prompt="Which color?",
    )

    task.transition(TaskStatus.RUNNING)
    task.transition(TaskStatus.WAITING)
    wait.transition(WaitStatus.SATISFIED)
    work.transition(WorkItemStatus.READY)
    work.transition(WorkItemStatus.COMPLETED)
    task.transition(TaskStatus.RUNNING)

    with pytest.raises(InvalidStateTransitionError):
        wait.transition(WaitStatus.SATISFIED)
