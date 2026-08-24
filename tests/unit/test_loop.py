import sys
from pathlib import Path

import pytest

from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.errors import (
    ModelProtocolError,
    ModelProviderError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.loop import continue_run, start_run
from handshake.runtime.runs import create_followup_run
from handshake.runtime.state import Run, RunStatus, Session
from handshake.runtime.waits import resolve_text_wait
from handshake.storage.sqlite import SQLiteStateStore
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


def _session_run(tmp_path: Path, goal: str, *, max_steps: int = 50) -> tuple[Session, Run]:
    session = Session(workspace=tmp_path)
    run = Run(
        goal=goal,
        session_id=session.session_id,
        model_name="scripted",
        max_steps=max_steps,
    )
    return session, run


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
    session, run = _session_run(tmp_path, "copy knowledge")

    start_run(session, run, model, create_default_registry(lambda _cwd, _argv: True))

    assert run.status is RunStatus.COMPLETED
    assert [tool_run.tool_name for tool_run in run.tool_runs] == [
        "read_file",
        "write_file",
    ]
    tool_results = model.contexts[1].messages[-1].content
    assert [block["tool_use_id"] for block in tool_results] == ["read", "write"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new"


def test_unknown_tool_is_structured_failure_and_loop_can_finish(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool_turn(ToolCall("missing", "not_registered", {})), final_turn()]
    )
    session, run = _session_run(tmp_path, "recover")

    start_run(session, run, model, create_default_registry(lambda _cwd, _argv: True))

    assert run.status is RunStatus.COMPLETED
    assert run.tool_runs[0].result is not None
    assert not run.tool_runs[0].result.ok
    assert "Unknown tool" in run.tool_runs[0].result.stderr


def test_mismatched_tool_blocks_fail_run(tmp_path: Path) -> None:
    bad_turn = ModelTurn(
        content=({"type": "tool_use", "id": "a", "name": "read_file", "input": {}},),
        tool_calls=(ToolCall("b", "read_file", {}),),
        stop_reason="tool_use",
    )
    session, run = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProtocolError):
        start_run(
            session,
            run,
            ScriptedModel([bad_turn]),
            create_default_registry(lambda _cwd, _argv: True),
        )

    assert run.status is RunStatus.FAILED


def test_script_exhaustion_fails_run(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProviderError):
        start_run(
            session,
            run,
            ScriptedModel([]),
            create_default_registry(lambda _cwd, _argv: True),
        )

    assert run.status is RunStatus.FAILED


def test_step_limit_is_owned_by_run(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path, "too long", max_steps=1)
    model = ScriptedModel(
        [
            tool_turn(
                ToolCall(
                    "search",
                    "search_files",
                    {"query": "missing", "path": "."},
                )
            )
        ]
    )

    with pytest.raises(StepLimitExceededError):
        start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
        )

    assert run.status is RunStatus.FAILED


def test_simple_run_can_edit_and_validate_without_a_plan(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("old", encoding="utf-8")
    session, run = _session_run(tmp_path, "replace one value")
    model = ScriptedModel(
        [
            tool_turn(ToolCall("read", "read_file", {"path": "value.txt"})),
            tool_turn(
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "value.txt", "content": "new"},
                )
            ),
            tool_turn(
                ToolCall(
                    "test",
                    "run_tests",
                    {
                        "argv": [
                            sys.executable,
                            "-c",
                            "assert open('value.txt').read() == 'new'",
                        ]
                    },
                )
            ),
            final_turn("updated and verified"),
        ]
    )

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
    )

    assert run.status is RunStatus.COMPLETED
    assert run.plan == []
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "new"
    assert run.tool_runs[-1].result is not None
    assert run.tool_runs[-1].result.ok


def test_prompt_and_plan_tool_describe_adaptive_planning(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path, "small request")
    model = ScriptedModel([final_turn()])

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
    )

    prompt = model.contexts[0].system_prompt
    assert "Optional current plan:\n(none)" in prompt
    assert "Investigate enough repository context" in prompt
    assert "Simple requests may be completed without creating a plan." in prompt
    assert "Treat the plan as provisional" in prompt
    assert "Keep a short current plan" not in prompt
    plan_tool = next(tool for tool in model.tool_sets[0] if tool.name == "update_plan")
    assert "non-trivial, multi-step work" in plan_tool.description
    assert "Do not call this tool for simple work" in plan_tool.description


def test_cancelled_tool_closes_protocol_before_run_terminates(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run = _session_run(tmp_path, "cancel")
    model = ScriptedModel(
        [tool_turn(ToolCall("shell", "run_shell", {"argv": ["true"]}))]
    )

    with pytest.raises(UserCancelledError):
        start_run(session, run, model, create_default_registry(cancel))

    assert run.status is RunStatus.CANCELLED
    result = session.messages[-1].content[0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "shell"
    assert result["is_error"] is True


def test_interrupted_multi_tool_turn_closes_every_call(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run = _session_run(tmp_path, "several tools")
    calls = (
        ToolCall("write-a", "write_file", {"path": "a.txt", "content": "done"}),
        ToolCall("shell", "run_shell", {"argv": ["true"]}),
        ToolCall("write-b", "write_file", {"path": "b.txt", "content": "skipped"}),
    )

    with pytest.raises(UserCancelledError):
        start_run(
            session,
            run,
            ScriptedModel([tool_turn(*calls)]),
            create_default_registry(cancel),
        )

    results = session.messages[-1].content
    assert [block["tool_use_id"] for block in results] == [
        "write-a",
        "shell",
        "write-b",
    ]
    assert results[0]["is_error"] is False
    assert results[1]["is_error"] is True
    assert results[2]["is_error"] is True
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "done"
    assert not (tmp_path / "b.txt").exists()


def test_followup_run_receives_complete_prior_protocol_history(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("value", encoding="utf-8")
    session, first = _session_run(tmp_path, "inspect")
    first_model = ScriptedModel(
        [
            ModelTurn(
                content=(
                    {"type": "thinking", "thinking": "read first"},
                    {
                        "type": "tool_use",
                        "id": "read",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    },
                ),
                tool_calls=(ToolCall("read", "read_file", {"path": "a.txt"}),),
                stop_reason="tool_use",
            ),
            final_turn("first complete"),
        ]
    )
    registry = create_default_registry(lambda _cwd, _argv: True)
    start_run(session, first, first_model, registry)
    second = create_followup_run(session, [first], "continue")
    second_model = ScriptedModel([final_turn("second complete")])

    start_run(session, second, second_model, registry)

    messages = second_model.contexts[0].messages
    assert [message.run_id for message in messages] == [
        first.run_id,
        first.run_id,
        first.run_id,
        first.run_id,
        second.run_id,
    ]
    assert messages[1].content[0]["type"] == "thinking"
    assert messages[1].content[1]["id"] == "read"
    assert messages[2].content[0]["tool_use_id"] == "read"
    assert messages[3].content[0]["text"] == "first complete"


def test_text_wait_round_trips_thinking_and_original_call_id(tmp_path: Path) -> None:
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
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session, run = _session_run(tmp_path, "style the button")
    store.create_session(session)
    first_model = ScriptedModel([wait_turn])

    start_run(
        session,
        run,
        first_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert run.status is RunStatus.WAITING
    wait_tool = next(tool for tool in first_model.tool_sets[0] if tool.name == "wait_for_text")
    assert wait_tool.input_schema["required"] == ["prompt"]
    assert "API keys" in wait_tool.description

    resumed_session = store.load_session(session.session_id)
    resumed_run = store.load_run(run.run_id)
    resolve_text_wait(
        resumed_session,
        resumed_run,
        resumed_run.waits[0].wait_id,
        "navy",
    )
    store.checkpoint_run(resumed_session, resumed_run)
    second_model = ScriptedModel([final_turn("used navy")])
    continue_run(
        resumed_session,
        resumed_run,
        second_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    context = second_model.contexts[0]
    assert context.messages[-2].content[0]["thinking"] == (
        "The repository cannot answer this."
    )
    assert context.messages[-1].content[0]["tool_use_id"] == "ask-color"
    assert resumed_run.status is RunStatus.COMPLETED
    assert store.load_run(run.run_id).final_summary == "used navy"


def test_mixed_wait_turn_rejects_every_call_without_side_effect(tmp_path: Path) -> None:
    mixed = tool_turn(
        ToolCall("write", "write_file", {"path": "created.txt", "content": "bad"}),
        ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}),
    )
    model = ScriptedModel([mixed, final_turn("corrected")])
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session, run = _session_run(tmp_path, "safe")
    store.create_session(session)

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert not (tmp_path / "created.txt").exists()
    assert run.status is RunStatus.COMPLETED
    errors = model.contexts[1].messages[-1].content
    assert len(errors) == 2
    assert all(block["is_error"] for block in errors)


def test_model_turn_budget_is_preserved_across_wait(tmp_path: Path) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session, run = _session_run(tmp_path, "ask once", max_steps=1)
    store.create_session(session)
    model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}))]
    )
    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )
    resumed_session = store.load_session(session.session_id)
    resumed_run = store.load_run(run.run_id)
    resolve_text_wait(
        resumed_session, resumed_run, resumed_run.waits[0].wait_id, "A"
    )
    store.checkpoint_run(resumed_session, resumed_run)

    with pytest.raises(StepLimitExceededError):
        continue_run(
            resumed_session,
            resumed_run,
            ScriptedModel([final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
        )

    assert resumed_run.status is RunStatus.FAILED
