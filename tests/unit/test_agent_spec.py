import pytest

from keyline_agent.context import AgentContextManager, ContextViewError
from keyline_agent.context.budget import ContextBudgetPolicy
from keyline_agent.models.errors import ModelProtocolError
from keyline_agent.models.protocol import ModelTurn, ToolCall
from keyline_agent.models.scripted import ScriptedModel
from keyline_agent.runtime.agents import (
    AgentSpec,
    demo_agent_spec,
    main_agent_spec,
    model_name_for_run,
)
from keyline_agent.runtime.dependencies import resolve_text_dependency
from keyline_agent.runtime.errors import AgentLoopError
from keyline_agent.runtime.loop import continue_run, start_run
from keyline_agent.runtime.runs import create_initial_run
from keyline_agent.runtime.state import RunStatus, Session
from keyline_agent.storage.sqlite import SQLiteStateStore
from keyline_agent.tools import create_default_registry
from tests.history_support import run_async


def _tool_turn(call: ToolCall) -> ModelTurn:
    return ModelTurn(
        content=({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments},),
        tool_calls=(call,),
        stop_reason="tool_use",
    )


def _final_turn() -> ModelTurn:
    return ModelTurn(content=({"type": "text", "text": "done"},), stop_reason="end_turn")


def test_agent_spec_holds_stable_configuration() -> None:
    spec = AgentSpec(
        spec_id="reader",
        model_name="deepseek-v4-flash",
        instructions="Read the code.",
        allowed_tools=frozenset({"read_file", "search_files"}),
    )
    assert spec.spec_id == "reader"
    assert spec.instructions == "Read the code."
    assert spec.default_max_steps == 0
    assert spec.allowed_tools == {"read_file", "search_files"}


def test_named_specs_define_their_own_instructions() -> None:
    budget = ContextBudgetPolicy()
    main = main_agent_spec("model", ("read_file", "run_tests"), budget)
    demo = demo_agent_spec(("run_tests",))

    assert main.spec_id == "main"
    assert main.instructions.startswith(
        "You are Keyline, a local coding agent operating on one workspace."
    )
    assert "main coding agent" not in main.instructions
    assert "delegate" not in main.instructions
    assert main.default_budget is budget
    assert main.allowed_tools == {"read_file", "run_tests"}
    assert demo.spec_id == "demo"
    assert "isolated coding demo" in demo.instructions
    assert demo.instructions != main.instructions


def test_delegation_guidance_requires_the_tool() -> None:
    main = main_agent_spec("model", ("read_file", "delegate"), ContextBudgetPolicy())

    assert "explorer" in main.instructions
    assert "reviewer" in main.instructions


def test_agent_spec_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        AgentSpec("", "model", "prompt")
    with pytest.raises(ValueError):
        AgentSpec("reader", "model", "prompt", default_max_steps=-1)


def test_model_selection_uses_current_spec_for_bound_runs(tmp_path) -> None:
    session = Session(workspace=tmp_path)
    bound = create_initial_run(
        session, "Continue", model_name="previous-model", max_steps=5,
        agent_spec_id="main",
    )
    current_spec = AgentSpec("main", "current-model", "Continue the work.")

    assert model_name_for_run(bound, current_spec) == "current-model"
    assert bound.model_name == "previous-model"
    wrong_spec = AgentSpec("reviewer", "wrong-model", "Review the work.")
    with pytest.raises(AgentLoopError, match="Agent spec main is unavailable"):
        model_name_for_run(bound, wrong_spec)


@run_async
async def test_resumed_run_uses_current_agent_configuration(tmp_path) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = create_initial_run(
        session,
        "Choose a color",
        model_name="old-model",
        max_steps=5,
        agent_spec_id="main",
    )
    registry = create_default_registry(lambda _cwd, _argv: True)
    original_spec = AgentSpec(
        "main", "old-model", "Old instructions", allowed_tools={"wait_for_text"}
    )
    first_model = ScriptedModel(
        [_tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}))]
    )

    await start_run(session, run, first_model, registry, store=store, agent_spec=original_spec)

    assert run.status is RunStatus.WAITING
    assert "Old instructions" in first_model.contexts[0].system_prompt
    assert "wait_for_text" in first_model.contexts[0].system_prompt
    assert "read_file" not in first_model.contexts[0].system_prompt
    assert {tool.name for tool in first_model.tool_sets[0]} == {"wait_for_text"}

    restored_session = store.load_session(session.session_id)
    restored_run = store.load_run(run.run_id)
    continuation = resolve_text_dependency(
        restored_session, restored_run, restored_run.dependencies[0].dependency_id, "blue"
    )
    current_spec = AgentSpec("main", "new-model", "New instructions", allowed_tools={"read_file"})
    second_model = ScriptedModel([_final_turn()])
    await continue_run(
        restored_session,
        restored_run,
        second_model,
        registry,
        store=store,
        pending_messages=(continuation,),
        agent_spec=current_spec,
    )

    assert restored_run.status is RunStatus.COMPLETED
    assert store.load_run(run.run_id).model_name == "new-model"
    assert "New instructions" in second_model.contexts[0].system_prompt
    assert "Old instructions" not in second_model.contexts[0].system_prompt
    assert "read_file" in second_model.contexts[0].system_prompt
    assert "wait_for_text" not in second_model.contexts[0].system_prompt
    assert {tool.name for tool in second_model.tool_sets[0]} == {"read_file"}


@run_async
async def test_agent_tool_allowance_blocks_unexpected_tool_call(tmp_path) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = create_initial_run(
        session,
        "Read the workspace",
        model_name="scripted",
        max_steps=5,
        agent_spec_id="reader",
    )
    model = ScriptedModel(
        [_tool_turn(ToolCall("write", "write_file", {"path": "bad.txt", "content": "bad"}))]
    )
    spec = AgentSpec("reader", "scripted", "Read only.", allowed_tools={"read_file"})

    with pytest.raises(ModelProtocolError, match="cannot use tool write_file"):
        await start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            agent_spec=spec,
        )

    assert run.status is RunStatus.FAILED
    assert not (tmp_path / "bad.txt").exists()


@run_async
async def test_run_rejects_mismatched_agent_spec(tmp_path) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = create_initial_run(
        session,
        "Run",
        model_name="scripted",
        max_steps=5,
        agent_spec_id="missing",
    )

    with pytest.raises(AgentLoopError, match="Agent spec missing is unavailable"):
        await start_run(
            session,
            run,
            ScriptedModel([_final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            agent_spec=AgentSpec("other", "scripted", "Do the work."),
        )

    assert run.status is RunStatus.CREATED
    assert store.list_runs(session.session_id) == ()


@run_async
async def test_run_rejects_context_manager_bound_to_another_scope(tmp_path) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = create_initial_run(
        session,
        "Run",
        model_name="scripted",
        max_steps=5,
        agent_spec_id="main",
    )

    with pytest.raises(ContextViewError, match="belongs to scope"):
        await start_run(
            session,
            run,
            ScriptedModel([_final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            context_manager=AgentContextManager(scope_id="another-scope"),
            agent_spec=AgentSpec("main", "scripted", "Do the work."),
        )

    assert run.status is RunStatus.CREATED
    assert store.list_runs(session.session_id) == ()
