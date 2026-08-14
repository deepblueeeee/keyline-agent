from pathlib import Path

import pytest

from handshake.models.scripted import ScriptedModel
from handshake.runtime.errors import (
    ModelProtocolError,
    ModelProviderError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.loop import continue_agent, run_agent
from handshake.runtime.state import ModelTurn, Task, TaskStatus, ToolCall
from handshake.storage.sqlite import SQLiteTaskStore
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


def test_text_wait_round_trips_thinking_and_resumes_with_original_call_id(
    tmp_path: Path,
) -> None:
    wait_turn = ModelTurn(
        content=(
            {"type": "thinking", "thinking": "The repository cannot answer this."},
            {
                "type": "tool_use",
                "id": "ask-color",
                "name": "wait_for_text",
                "input": {"prompt": "Which color should the button use?"},
            },
        ),
        tool_calls=(
            ToolCall(
                "ask-color",
                "wait_for_text",
                {"prompt": "Which color should the button use?"},
            ),
        ),
        stop_reason="tool_use",
    )
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    first_model = ScriptedModel([wait_turn])
    task = Task(goal="style the button", workspace=tmp_path)

    run_agent(
        task,
        first_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert task.status is TaskStatus.WAITING
    wait_tool = next(tool for tool in first_model.tool_sets[0] if tool.name == "wait_for_text")
    assert wait_tool.input_schema["required"] == ["prompt"]
    assert "API keys" in wait_tool.description

    resumed = store.resolve_text_wait(task.task_id, task.waits[0].wait_id, "navy")
    second_model = ScriptedModel([final_turn("used navy")])
    continue_agent(
        resumed,
        second_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    context = second_model.contexts[0]
    assert context.messages[-2].content[0]["thinking"] == (
        "The repository cannot answer this."
    )
    assert context.messages[-1].content[0]["tool_use_id"] == "ask-color"
    assert resumed.status is TaskStatus.COMPLETED
    assert store.load_task(task.task_id).final_summary == "used navy"


def test_mixed_wait_turn_rejects_every_call_without_side_effect(tmp_path: Path) -> None:
    mixed = tool_turn(
        ToolCall("write", "write_file", {"path": "created.txt", "content": "bad"}),
        ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}),
    )
    model = ScriptedModel([mixed, final_turn("corrected")])
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    task = Task(goal="safe", workspace=tmp_path)

    run_agent(
        task,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert not (tmp_path / "created.txt").exists()
    assert task.status is TaskStatus.COMPLETED
    errors = model.contexts[1].messages[-1].content
    assert len(errors) == 2
    assert all(block["is_error"] for block in errors)


def test_model_turn_budget_is_preserved_across_wait(tmp_path: Path) -> None:
    store = SQLiteTaskStore(tmp_path / "handshake.db")
    task = Task(goal="ask once", workspace=tmp_path)
    model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}))]
    )
    run_agent(
        task,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        max_steps=1,
        store=store,
    )
    resumed = store.resolve_text_wait(task.task_id, task.waits[0].wait_id, "A")

    with pytest.raises(StepLimitExceededError):
        continue_agent(
            resumed,
            ScriptedModel([final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
        )

    assert resumed.status is TaskStatus.FAILED
