import pytest

from handshake.runtime.errors import InvalidStateTransitionError
from handshake.runtime.state import (
    Dependency,
    DependencyStatus,
    Run,
    RunStatus,
    ToolRun,
    ToolRunStatus,
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
        tool_call_id="call",
        tool_name="read_file",
        arguments={},
    )
    run.transition(ToolRunStatus.RUNNING)
    run.transition(ToolRunStatus.FAILED)

    with pytest.raises(InvalidStateTransitionError):
        run.transition(ToolRunStatus.SUCCEEDED)


def test_dependency_blocks_run_without_a_duplicate_waiting_status() -> None:
    run = Run(goal="test", session_id="test-session", agent_spec_id="test")
    dependency = Dependency(
        tool_call_id="ask-1",
        prompt="Which color?",
    )
    run.add_dependency(dependency)

    run.transition(RunStatus.RUNNING)
    run.transition(RunStatus.WAITING)
    assert run.open_dependencies() == (dependency,)
    dependency.transition(DependencyStatus.SATISFIED)
    run.transition(RunStatus.RUNNING)
    assert run.open_dependencies() == ()

    with pytest.raises(InvalidStateTransitionError):
        dependency.transition(DependencyStatus.SATISFIED)


def test_run_owns_dependency_and_tool_run_relationships() -> None:
    run = Run(goal="test", session_id="test-session", agent_spec_id="test")
    dependency = Dependency(
        tool_call_id="ask-1",
        prompt="Which color?",
    )
    tool_run = ToolRun(
        tool_call_id="read-1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    run.add_dependency(dependency)
    run.add_tool_run(tool_run)

    assert run.dependencies == [dependency]
    assert run.tool_runs == [tool_run]
    assert run.dependency_by_id(dependency.dependency_id) is dependency

    with pytest.raises(InvalidStateTransitionError, match="already belongs"):
        run.add_dependency(dependency)
    with pytest.raises(InvalidStateTransitionError, match="already belongs"):
        run.add_tool_run(tool_run)
