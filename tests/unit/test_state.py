import pytest

from handshake.runtime.errors import InvalidStateTransitionError
from handshake.runtime.state import (
    Dependency,
    DependencyStatus,
    Run,
    RunStatus,
    ToolRun,
    ToolRunStatus,
    WorkItem,
    WorkItemStatus,
)


def test_run_accepts_only_declared_transitions() -> None:
    run = Run(goal="test", session_id="test-session", agent_spec_id="test")
    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(RunStatus.RUNNING)


@pytest.mark.parametrize("spec_id", ["", "   "])
def test_run_requires_agent_spec_id(spec_id: str) -> None:
    with pytest.raises(ValueError, match="must have an agent spec ID"):
        Run(goal="test", session_id="test-session", agent_spec_id=spec_id)


def test_tool_run_accepts_only_declared_transitions() -> None:
    run = ToolRun(
        work_id="work",
        tool_call_id="call",
        tool_name="read_file",
        arguments={},
    )
    run.transition(ToolRunStatus.RUNNING)
    run.transition(ToolRunStatus.FAILED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(ToolRunStatus.SUCCEEDED)


def test_dependency_blocks_ready_work_without_a_duplicate_waiting_status() -> None:
    run = Run(goal="test", session_id="test-session", agent_spec_id="test")
    work = run.work_items[0]
    dependency = Dependency(
        work_id=work.work_id,
        tool_call_id="ask-1",
        prompt="Which color?",
    )
    run.add_dependency(work, dependency)

    run.transition(RunStatus.RUNNING)
    work.transition(WorkItemStatus.RUNNING)
    work.transition(WorkItemStatus.READY)
    run.transition(RunStatus.WAITING)
    dependency.transition(DependencyStatus.SATISFIED)
    run.transition(RunStatus.RUNNING)
    work.transition(WorkItemStatus.RUNNING)
    work.transition(WorkItemStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        dependency.transition(DependencyStatus.SATISFIED)


def test_run_owns_work_item_relationships() -> None:
    run = Run(goal="test", session_id="test-session", agent_spec_id="test")
    work = run.work_items[0]
    dependency = Dependency(
        work_id=work.work_id,
        tool_call_id="ask-1",
        prompt="Which color?",
    )
    tool_run = ToolRun(
        work_id=work.work_id,
        tool_call_id="read-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    run.add_dependency(work, dependency)
    run.add_tool_run(work, tool_run)

    assert run.dependencies_for(work) == (dependency,)
    assert run.tool_runs_for(work) == (tool_run,)
    assert run.dependency_by_id(dependency.dependency_id) is dependency
    assert run.work_item_by_id(work.work_id) is work

    foreign_work = WorkItem(description="foreign")
    with pytest.raises(InvalidStateTransitionError, match="does not belong"):
        run.dependencies_for(foreign_work)
    with pytest.raises(InvalidStateTransitionError, match="is bound to work item"):
        run.add_dependency(
            work,
            Dependency(
                work_id=foreign_work.work_id,
                tool_call_id="ask-2",
                prompt="Which size?",
            ),
        )
    with pytest.raises(InvalidStateTransitionError, match="already belongs"):
        run.add_tool_run(work, tool_run)
