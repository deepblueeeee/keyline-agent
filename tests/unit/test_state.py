from pathlib import Path

import pytest

from handshake.runtime.errors import InvalidStateTransitionError
from handshake.runtime.state import Task, TaskStatus, ToolRun, ToolRunStatus


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
