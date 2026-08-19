import pytest

from handshake.runtime.errors import InvalidStateTransitionError
from handshake.runtime.state import (
    Run,
    RunStatus,
    ToolRun,
    ToolRunStatus,
    WaitCondition,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
)


def test_run_accepts_only_declared_transitions() -> None:
    run = Run(goal="test", session_id="test-session")
    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(RunStatus.RUNNING)


def test_tool_run_accepts_only_declared_transitions() -> None:
    run = ToolRun(tool_call_id="call", tool_name="read_file", arguments={})
    run.transition(ToolRunStatus.RUNNING)
    run.transition(ToolRunStatus.FAILED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(ToolRunStatus.SUCCEEDED)


def test_waiting_run_and_bound_work_follow_declared_transitions() -> None:
    run = Run(goal="test", session_id="test-session")
    work = WorkItem(description="Choose a color", tool_call_id="ask-1")
    wait = WaitCondition(
        work_id=work.work_id,
        tool_call_id="ask-1",
        prompt="Which color?",
    )

    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.WAITING)
    wait.transition(WaitStatus.SATISFIED)
    work.transition(WorkItemStatus.READY)
    work.transition(WorkItemStatus.COMPLETED)
    run.transition(RunStatus.RUNNING)

    with pytest.raises(InvalidStateTransitionError):
        wait.transition(WaitStatus.SATISFIED)
