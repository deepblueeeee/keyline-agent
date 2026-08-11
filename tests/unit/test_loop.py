from pathlib import Path

import pytest

from handshake.models.scripted import ScriptedModel
from handshake.runtime.errors import (
    ModelProtocolError,
    ModelProviderError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.loop import run_agent
from handshake.runtime.state import ModelTurn, Task, TaskStatus, ToolCall
from handshake.tools import create_default_registry


def tool_turn(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(
        content=tuple(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
            for call in calls
        ),
        tool_calls=calls,
        stop_reason="tool_use",
    )


def final_turn(text: str = "done") -> ModelTurn:
    return ModelTurn(
        content=({"type": "text", "text": text},),
        stop_reason="end_turn",
    )


def test_multiple_tools_execute_in_order_and_results_return_to_model(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
    model = ScriptedModel(
        [
            tool_turn(
                ToolCall("read", "read_file", {"path": "a.txt"}),
                ToolCall("write", "write_file", {"path": "b.txt", "content": "new"}),
            ),
            final_turn("finished"),
        ]
    )
    task = Task(goal="copy knowledge", workspace=tmp_path)

    run_agent(task, model, create_default_registry(lambda _cwd, _argv: True))

    assert task.status is TaskStatus.COMPLETED
    assert [run.tool_name for run in task.tool_runs] == ["read_file", "write_file"]
    second_context = model.contexts[1]
    tool_results = second_context.messages[-1].content
    assert [block["tool_use_id"] for block in tool_results] == ["read", "write"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new"


def test_unknown_tool_is_structured_failure_and_loop_can_finish(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool_turn(ToolCall("missing", "not_registered", {})), final_turn()]
    )
    task = Task(goal="recover", workspace=tmp_path)

    run_agent(task, model, create_default_registry(lambda _cwd, _argv: True))

    assert task.status is TaskStatus.COMPLETED
    assert task.tool_runs[0].result is not None
    assert not task.tool_runs[0].result.ok
    assert "Unknown tool" in task.tool_runs[0].result.stderr


def test_mismatched_tool_blocks_fail_task(tmp_path: Path) -> None:
    bad_turn = ModelTurn(
        content=({"type": "tool_use", "id": "a", "name": "read_file", "input": {}},),
        tool_calls=(ToolCall("b", "read_file", {}),),
        stop_reason="tool_use",
    )
    task = Task(goal="bad", workspace=tmp_path)

    with pytest.raises(ModelProtocolError):
        run_agent(
            task,
            ScriptedModel([bad_turn]),
            create_default_registry(lambda _cwd, _argv: True),
        )

    assert task.status is TaskStatus.FAILED


def test_script_exhaustion_fails_task(tmp_path: Path) -> None:
    task = Task(goal="bad", workspace=tmp_path)

    with pytest.raises(ModelProviderError):
        run_agent(
            task,
            ScriptedModel([]),
            create_default_registry(lambda _cwd, _argv: True),
        )

    assert task.status is TaskStatus.FAILED


def test_step_limit_fails_task(tmp_path: Path) -> None:
    task = Task(goal="too long", workspace=tmp_path)
    model = ScriptedModel(
        [tool_turn(ToolCall("plan", "update_plan", {"steps": ["continue"]}))]
    )

    with pytest.raises(StepLimitExceededError):
        run_agent(
            task,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            max_steps=1,
        )

    assert task.status is TaskStatus.FAILED


def test_cancelled_approval_cancels_task(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    task = Task(goal="cancel", workspace=tmp_path)
    model = ScriptedModel(
        [tool_turn(ToolCall("shell", "run_shell", {"argv": ["true"]}))]
    )

    with pytest.raises(UserCancelledError):
        run_agent(task, model, create_default_registry(cancel))

    assert task.status is TaskStatus.CANCELLED
