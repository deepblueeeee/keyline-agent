import json
from dataclasses import replace
from pathlib import Path

import pytest

import keyline_agent.runtime.delegation as runtime_delegation
from keyline_agent.context import ActiveContextManagers
from keyline_agent.context.budget import ContextBudgetPolicy
from keyline_agent.models.protocol import ModelTurn, ToolCall
from keyline_agent.models.scripted import ScriptedModel
from keyline_agent.runtime.agents import AgentSpec, child_agent_specs
from keyline_agent.runtime.delegation import DelegationRuntime, continue_delegation
from keyline_agent.runtime.dependencies import resolve_text_dependency
from keyline_agent.runtime.loop import continue_run, start_run
from keyline_agent.runtime.recovery import resume_run
from keyline_agent.runtime.state import Run, RunStatus, Session
from keyline_agent.storage.sqlite import SQLiteStateStore
from keyline_agent.tools import create_default_registry
from tests.history_support import run_async


def _tool_turn(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(
        content=tuple(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            for call in calls
        ),
        tool_calls=calls,
        stop_reason="tool_use",
    )


def _final(text: str) -> ModelTurn:
    return ModelTurn(content=({"type": "text", "text": text},), stop_reason="end_turn")


def _setup(tmp_path: Path, parent_turns: list[ModelTurn], child_turns: list[ModelTurn]):
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    parent = Run("Parent goal", session.session_id, "main")
    main_spec = AgentSpec(
        "main", "scripted", "Manage the task.", allowed_tools={"delegate", "read_file"}
    )
    child_spec = AgentSpec(
        "explorer",
        "scripted",
        "Explore only.",
        allowed_tools={"read_file", "wait_for_text"},
    )
    parent_model = ScriptedModel(parent_turns)
    child_model = ScriptedModel(child_turns)
    registry = create_default_registry(lambda _cwd, _argv: True)
    delegation = DelegationRuntime(
        {"main": main_spec, "explorer": child_spec},
        lambda _spec: child_model,
        ActiveContextManagers(),
    )
    return (
        session,
        store,
        parent,
        main_spec,
        child_spec,
        parent_model,
        child_model,
        registry,
        delegation,
    )


def _child_result(store: SQLiteStateStore, parent: Run) -> dict[str, object]:
    messages = tuple(store.read_messages(parent.context_scope_id))
    block = next(
        message.message.content[0]
        for message in messages
        if message.message.role == "user"
        and message.message.content[0].get("type") == "tool_result"
    )
    return json.loads(block["content"])


@run_async
async def test_delegation_runs_child_to_completion_with_private_context(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall(
                    "delegate-1",
                    "delegate",
                    {"agent_spec_id": "explorer", "goal": "Find the parser"},
                )
            ),
            _final("Parent finished"),
        ],
        [_final("Parser is in parser.py")],
    )
    session, store, parent, main_spec, _, parent_model, child_model, registry, delegation = setup

    await start_run(
        session,
        parent,
        parent_model,
        registry,
        store=store,
        agent_spec=main_spec,
        delegation=delegation,
    )

    children = [
        run for run in store.list_runs(session.session_id) if run.parent_run_id == parent.run_id
    ]
    assert len(children) == 1
    child = children[0]
    assert parent.status is RunStatus.COMPLETED
    assert child.status is RunStatus.COMPLETED
    assert child.delegation_tool_call_id == "delegate-1"
    assert child.context_scope_id != parent.context_scope_id
    assert child.previous_run_id is None
    assert [
        message.message.content[0]["text"]
        for message in store.read_messages(child.context_scope_id)
    ] == ["Find the parser", "Parser is in parser.py"]
    assert "Explore only." in child_model.contexts[0].system_prompt
    assert "delegate" not in {tool.name for tool in child_model.tool_sets[0]}
    assert "Manage the task." not in child_model.contexts[0].system_prompt
    assert "Find the parser" not in parent_model.contexts[1].system_prompt
    assert "Parser is in parser.py" not in parent_model.contexts[1].system_prompt
    assert _child_result(store, parent) == {
        "run_id": child.run_id,
        "agent_spec_id": "explorer",
        "status": "completed",
        "summary": "Parser is in parser.py",
        "error": None,
    }


@run_async
async def test_batch_creates_all_children_before_running_and_joins_once(
    tmp_path: Path,
) -> None:
    calls = (
        ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
    )
    setup = _setup(
        tmp_path,
        [_tool_turn(*calls), _final("Used both results")],
        [],
    )
    (
        session, store, parent, main_spec, explorer_spec,
        parent_model, _, registry, delegation,
    ) = setup
    reviewer_spec = AgentSpec(
        "reviewer", "scripted", "Review only.", allowed_tools={"read_file"}
    )
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["reviewer"] = reviewer_spec
    explorer_model = ScriptedModel(
        [
            _tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which file?"})),
            _final("Exploration done"),
        ]
    )
    reviewer_model = ScriptedModel([_final("Review done")])

    def model_for_spec(spec: AgentSpec) -> ScriptedModel:
        assert len(store.list_runs(session.session_id)) == 3
        return explorer_model if spec.spec_id == "explorer" else reviewer_model

    delegation.model_for_spec = model_for_spec
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    children = store.list_runs(session.session_id)[1:]
    assert [child.delegation_tool_call_id for child in children] == ["explore", "review"]
    assert [child.status for child in children] == [RunStatus.WAITING, RunStatus.COMPLETED]
    assert parent.status is RunStatus.WAITING
    assert len(tuple(store.read_messages(parent.context_scope_id))) == 2

    explorer = children[0]
    continuation = resolve_text_dependency(
        session, explorer, explorer.open_dependencies()[0].dependency_id, "parser.py"
    )
    await continue_run(
        session, explorer, explorer_model, registry, store=store,
        pending_messages=(continuation,), agent_spec=explorer_spec,
    )
    await continue_delegation(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )

    assert parent.status is RunStatus.COMPLETED
    messages = tuple(store.read_messages(parent.context_scope_id))
    assert len(messages) == 4
    result_message = messages[2].message
    assert result_message.role == "user"
    assert [block["tool_use_id"] for block in result_message.content] == [
        "explore", "review",
    ]
    assert [
        json.loads(block["content"])["summary"] for block in result_message.content
    ] == ["Exploration done", "Review done"]


@run_async
async def test_failed_child_does_not_prevent_sibling_or_batch_join(tmp_path: Path) -> None:
    calls = (
        ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("Handled both")], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["reviewer"] = AgentSpec(
        "reviewer", "scripted", "Review only.", allowed_tools={"read_file"}
    )
    models = {
        "explorer": ScriptedModel([ModelTurn(content=(), stop_reason="end_turn")]),
        "reviewer": ScriptedModel([_final("Review done")]),
    }
    delegation.model_for_spec = lambda spec: models[spec.spec_id]
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    children = store.list_runs(session.session_id)[1:]
    assert [child.status for child in children] == [RunStatus.FAILED, RunStatus.COMPLETED]
    assert parent.status is RunStatus.COMPLETED
    results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
    assert [block["is_error"] for block in results] == [True, False]


@run_async
async def test_invalid_batch_creates_no_children_and_returns_all_results(
    tmp_path: Path,
) -> None:
    calls = (
        ToolCall("valid", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("invalid", "delegate", {"agent_spec_id": "missing", "goal": "Review"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("Handled invalid batch")], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    assert parent.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 1
    results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
    assert [block["tool_use_id"] for block in results] == ["valid", "invalid"]
    assert all(block["is_error"] for block in results)


@run_async
async def test_two_children_of_same_spec_keep_independent_contexts(tmp_path: Path) -> None:
    calls = (
        ToolCall("first", "delegate", {"agent_spec_id": "explorer", "goal": "Find A"}),
        ToolCall("second", "delegate", {"agent_spec_id": "explorer", "goal": "Find B"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("Combined")], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    models: list[ScriptedModel] = []

    def model_for_spec(_spec: AgentSpec) -> ScriptedModel:
        model = ScriptedModel([_final(f"Result {len(models) + 1}")])
        models.append(model)
        return model

    delegation.model_for_spec = model_for_spec
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    first, second = store.list_runs(session.session_id)[1:]
    assert first.context_scope_id != second.context_scope_id
    assert [model.contexts[0].goal for model in models] == ["Find A", "Find B"]
    results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
    assert [json.loads(block["content"])["summary"] for block in results] == [
        "Result 1", "Result 2",
    ]


@run_async
async def test_waiting_child_hands_off_and_resumes_parent_after_restart(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall(
                    "delegate-1", "delegate", {"agent_spec_id": "explorer", "goal": "Investigate"}
                )
            ),
        ],
        [
            _tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which module?"})),
            _final("Found module A"),
        ],
    )
    (
        session,
        store,
        parent,
        main_spec,
        child_spec,
        parent_model,
        child_model,
        registry,
        delegation,
    ) = setup
    await start_run(
        session,
        parent,
        parent_model,
        registry,
        store=store,
        agent_spec=main_spec,
        delegation=delegation,
    )
    child = store.list_runs(session.session_id)[1]
    assert parent.status is RunStatus.WAITING
    assert parent.open_dependencies() == ()
    assert child.status is RunStatus.WAITING
    parent_messages = tuple(store.read_messages(parent.context_scope_id))
    assert len(parent_messages) == 2
    assert parent_messages[-1].message.role == "assistant"

    restored_session = store.load_session(session.session_id)
    child = store.load_run(child.run_id)
    continuation = resolve_text_dependency(
        restored_session, child, child.open_dependencies()[0].dependency_id, "A"
    )
    await continue_run(
        restored_session,
        child,
        child_model,
        registry,
        store=store,
        pending_messages=(continuation,),
        agent_spec=child_spec,
    )
    assert child.status is RunStatus.COMPLETED

    restored_parent = store.load_run(parent.run_id)
    resumed_model = ScriptedModel([_final("Parent used child result")])
    restored_delegation = DelegationRuntime(
        delegation.agent_specs, lambda _spec: child_model, ActiveContextManagers()
    )
    await continue_delegation(
        restored_session,
        restored_parent,
        resumed_model,
        registry,
        store=store,
        agent_spec=main_spec,
        delegation=restored_delegation,
    )
    assert restored_parent.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 2
    assert _child_result(store, restored_parent)["summary"] == "Found module A"
    assert "Found module A" in resumed_model.contexts[0].messages[-1].content[0]["content"]


@run_async
async def test_child_failure_is_a_structured_tool_error_for_parent(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall("delegate-1", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"})
            ),
            _final("Handled child failure"),
        ],
        [ModelTurn(content=(), stop_reason="end_turn")],
    )
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    await start_run(
        session,
        parent,
        parent_model,
        registry,
        store=store,
        agent_spec=main_spec,
        delegation=delegation,
    )
    child = store.list_runs(session.session_id)[1]
    assert child.status is RunStatus.FAILED
    assert parent.status is RunStatus.COMPLETED
    result = _child_result(store, parent)
    assert result["status"] == "failed"
    assert result["error"]


@run_async
async def test_child_setup_failure_is_a_structured_tool_error_for_parent(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall("delegate-1", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"})
            ),
            _final("Handled child setup failure"),
        ],
        [],
    )
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup

    def unavailable_model(_spec: AgentSpec) -> ScriptedModel:
        raise RuntimeError("provider unavailable")

    delegation.model_for_spec = unavailable_model
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    child = store.list_runs(session.session_id)[1]
    assert child.status is RunStatus.FAILED
    assert parent.status is RunStatus.COMPLETED
    assert _child_result(store, parent)["error"] == "Unexpected child failure (RuntimeError)"


def test_builtin_child_profiles_cannot_delegate() -> None:
    for spec in child_agent_specs("scripted", ContextBudgetPolicy()):
        assert "delegate" not in spec.allowed_tools


@run_async
async def test_delegation_rejects_a_spec_that_can_delegate(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall("delegate-1", "delegate", {"agent_spec_id": "main", "goal": "Recurse"})
            ),
            _final("Handled rejected child"),
        ],
        [],
    )
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    assert parent.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 1


@run_async
async def test_child_context_pressure_is_handed_back_to_parent(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall("delegate-1", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"})
            ),
            _final("Handled child context pressure"),
        ],
        [],
    )
    session, store, parent, main_spec, child_spec, parent_model, _, registry, delegation = setup
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["explorer"] = replace(
        child_spec,
        default_budget=ContextBudgetPolicy(
            context_window_tokens=2,
            reserved_output_tokens=1,
        ),
    )
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    child = store.list_runs(session.session_id)[1]
    assert child.status is RunStatus.FAILED
    assert parent.status is RunStatus.COMPLETED
    assert _child_result(store, parent)["status"] == "failed"


@run_async
async def test_mixed_delegation_turn_rejects_all_tools(tmp_path: Path) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall(
                    "delegate-1", "delegate",
                    {"agent_spec_id": "explorer", "goal": "Explore"},
                ),
                ToolCall("read-1", "read_file", {"path": "missing.txt"}),
            ),
            _final("Recovered"),
        ],
        [],
    )
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    assert parent.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 1
    assert parent.tool_runs == []
    result_message = tuple(store.read_messages(parent.context_scope_id))[2].message
    assert all(block["is_error"] for block in result_message.content)


@run_async
async def test_checkpointed_delegation_creates_child_once_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall("delegate-1", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"})
            )
        ],
        [_final("Found answer")],
    )
    session, store, parent, main_spec, _, parent_model, child_model, registry, delegation = setup
    original = runtime_delegation._run_child

    async def interrupt(*_args: object, **_kwargs: object) -> Run:
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_delegation, "_run_child", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session,
            parent,
            parent_model,
            registry,
            store=store,
            agent_spec=main_spec,
            delegation=delegation,
        )
    assert store.load_run(parent.run_id).status is RunStatus.WAITING
    assert len(store.list_runs(session.session_id)) == 2
    assert store.list_runs(session.session_id)[1].status is RunStatus.RUNNING
    monkeypatch.setattr(runtime_delegation, "_run_child", original)

    resumed = store.load_run(parent.run_id)
    resumed_model = ScriptedModel([_final("Done")])
    restored_delegation = DelegationRuntime(
        delegation.agent_specs, lambda _spec: child_model, ActiveContextManagers()
    )
    await continue_delegation(
        store.load_session(session.session_id),
        resumed,
        resumed_model,
        registry,
        store=store,
        agent_spec=main_spec,
        delegation=restored_delegation,
    )
    assert resumed.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 2
    assert _child_result(store, resumed)["summary"] == "Found answer"


@run_async
async def test_interrupted_batch_resumes_precreated_children_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = (
        ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls)], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["reviewer"] = AgentSpec(
        "reviewer", "scripted", "Review only.", allowed_tools={"read_file"}
    )
    original = runtime_delegation._run_child

    async def interrupt(*_args: object, **_kwargs: object) -> Run:
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_delegation, "_run_child", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session, parent, parent_model, registry, store=store,
            agent_spec=main_spec, delegation=delegation,
        )
    assert [run.status for run in store.list_runs(session.session_id)] == [
        RunStatus.WAITING, RunStatus.RUNNING, RunStatus.RUNNING,
    ]
    monkeypatch.setattr(runtime_delegation, "_run_child", original)

    restored = store.load_run(parent.run_id)
    restored_delegation = DelegationRuntime(
        delegation.agent_specs,
        lambda spec: ScriptedModel([_final(f"{spec.spec_id} done")]),
        ActiveContextManagers(),
    )
    await continue_delegation(
        store.load_session(session.session_id), restored,
        ScriptedModel([_final("Done")]), registry, store=store,
        agent_spec=main_spec, delegation=restored_delegation,
    )
    assert restored.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 3
    results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
    assert [block["tool_use_id"] for block in results] == ["explore", "review"]


@run_async
async def test_interrupted_batch_creation_restores_only_missing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = (
        ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls)], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["reviewer"] = AgentSpec(
        "reviewer", "scripted", "Review only.", allowed_tools={"read_file"}
    )
    original_create = store.create_run

    def interrupt_second_child(
        current_session: Session,
        run: Run,
        messages: tuple = (),
    ) -> tuple:
        if run.delegation_tool_call_id == "review":
            raise KeyboardInterrupt
        return original_create(current_session, run, messages)

    monkeypatch.setattr(store, "create_run", interrupt_second_child)
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session, parent, parent_model, registry, store=store,
            agent_spec=main_spec, delegation=delegation,
        )
    assert [run.delegation_tool_call_id for run in store.list_runs(session.session_id)] == [
        None, "explore",
    ]
    monkeypatch.setattr(store, "create_run", original_create)
    restored = store.load_run(parent.run_id)
    restored_delegation = DelegationRuntime(
        delegation.agent_specs,
        lambda spec: ScriptedModel([_final(f"{spec.spec_id} done")]),
        ActiveContextManagers(),
    )
    await continue_delegation(
        store.load_session(session.session_id), restored,
        ScriptedModel([_final("Done")]), registry, store=store,
        agent_spec=main_spec, delegation=restored_delegation,
    )
    assert restored.status is RunStatus.COMPLETED
    assert len(store.list_runs(session.session_id)) == 3


@run_async
async def test_joined_batch_resumes_parent_without_duplicate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = (
        ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
        ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls)], [])
    session, store, parent, main_spec, _, parent_model, _, registry, delegation = setup
    assert isinstance(delegation.agent_specs, dict)
    delegation.agent_specs["reviewer"] = AgentSpec(
        "reviewer", "scripted", "Review only.", allowed_tools={"read_file"}
    )
    original_child = runtime_delegation._run_child

    async def interrupt(*_args: object, **_kwargs: object) -> Run:
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_delegation, "_run_child", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session, parent, parent_model, registry, store=store,
            agent_spec=main_spec, delegation=delegation,
        )
    monkeypatch.setattr(runtime_delegation, "_run_child", original_child)
    original_turns = runtime_delegation._run_turns
    monkeypatch.setattr(runtime_delegation, "_run_turns", interrupt)
    restored = store.load_run(parent.run_id)
    restored_delegation = DelegationRuntime(
        delegation.agent_specs,
        lambda spec: ScriptedModel([_final(f"{spec.spec_id} done")]),
        ActiveContextManagers(),
    )
    with pytest.raises(KeyboardInterrupt):
        await continue_delegation(
            store.load_session(session.session_id), restored,
            ScriptedModel([]), registry, store=store,
            agent_spec=main_spec, delegation=restored_delegation,
        )
    monkeypatch.setattr(runtime_delegation, "_run_turns", original_turns)
    persisted = store.load_run(parent.run_id)
    assert persisted.status is RunStatus.RUNNING
    messages = tuple(store.read_messages(parent.context_scope_id))
    assert len(messages[2].message.content) == 2

    await resume_run(
        store.load_session(session.session_id), persisted,
        ScriptedModel([_final("Done")]), registry, store=store,
        agent_spec=main_spec, delegation=restored_delegation,
    )
    assert persisted.status is RunStatus.COMPLETED
    assert len(tuple(store.read_messages(parent.context_scope_id))) == 4


@run_async
async def test_committed_child_result_resumes_without_duplicate_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup(
        tmp_path,
        [
            _tool_turn(
                ToolCall(
                    "delegate-1", "delegate",
                    {"agent_spec_id": "explorer", "goal": "Explore"},
                )
            ),
        ],
        [
            _tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which module?"})),
            _final("Found answer"),
        ],
    )
    (
        session, store, parent, main_spec, child_spec,
        parent_model, child_model, registry, delegation,
    ) = setup
    await start_run(
        session, parent, parent_model, registry, store=store,
        agent_spec=main_spec, delegation=delegation,
    )
    child = store.list_runs(session.session_id)[1]
    continuation = resolve_text_dependency(
        session, child, child.open_dependencies()[0].dependency_id, "A"
    )
    await continue_run(
        session, child, child_model, registry, store=store,
        pending_messages=(continuation,), agent_spec=child_spec,
    )
    assert child.status is RunStatus.COMPLETED

    async def interrupt(*_args: object, **_kwargs: object) -> Run:
        raise KeyboardInterrupt

    restored_parent = store.load_run(parent.run_id)
    restored_delegation = DelegationRuntime(
        delegation.agent_specs, lambda _spec: child_model, ActiveContextManagers()
    )
    original_turns = runtime_delegation._run_turns
    monkeypatch.setattr(runtime_delegation, "_run_turns", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await continue_delegation(
            store.load_session(session.session_id), restored_parent,
            ScriptedModel([]), registry, store=store,
            agent_spec=main_spec, delegation=restored_delegation,
        )
    monkeypatch.setattr(runtime_delegation, "_run_turns", original_turns)
    assert store.load_run(parent.run_id).status is RunStatus.RUNNING

    resumed = store.load_run(parent.run_id)
    await continue_delegation(
        store.load_session(session.session_id), resumed,
        ScriptedModel([_final("Done")]), registry, store=store,
        agent_spec=main_spec, delegation=restored_delegation,
    )
    assert resumed.status is RunStatus.COMPLETED
    result_blocks = [
        block
        for item in store.read_messages(parent.context_scope_id)
        for block in item.message.content
        if block.get("type") == "tool_result"
    ]
    assert len(result_blocks) == 1


@run_async
async def test_concurrent_children_finish_out_of_order_and_join_in_call_order(
    tmp_path: Path,
) -> None:
    import asyncio

    calls = tuple(
        ToolCall(str(index), "delegate", {"agent_spec_id": "explorer", "goal": str(index)})
        for index in range(3)
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("joined")], [])
    session, store, parent, main_spec, _, model, _, registry, delegation = setup
    starts: asyncio.Queue[str] = asyncio.Queue()
    releases = {str(index): asyncio.Event() for index in range(3)}
    finishes: list[str] = []

    class ConcurrentModel:
        async def complete(self, context, _tools) -> ModelTurn:
            assert len(store.list_runs(session.session_id)) == 4
            starts.put_nowait(context.goal)
            await releases[context.goal].wait()
            finishes.append(context.goal)
            return _final(f"result {context.goal}")

    delegation.model_for_spec = lambda _spec: ConcurrentModel()
    task = asyncio.create_task(start_run(
        session, parent, model, registry, store=store, agent_spec=main_spec,
        delegation=delegation,
    ))
    try:
        assert [await asyncio.wait_for(starts.get(), 1) for _ in range(3)] == ["0", "1", "2"]
        assert len(model.contexts) == 1
        assert len(tuple(store.read_messages(parent.context_scope_id))) == 2
        releases["2"].set()
        releases["1"].set()
        await asyncio.sleep(0)
        assert finishes == ["2", "1"]
        assert len(model.contexts) == 1
        releases["0"].set()
        await task
    finally:
        for release in releases.values():
            release.set()
        await task
    assert parent.status is RunStatus.COMPLETED
    result_message = tuple(store.read_messages(parent.context_scope_id))[2].message
    assert [block["tool_use_id"] for block in result_message.content] == ["0", "1", "2"]
    assert [json.loads(block["content"])["summary"] for block in result_message.content] == [
        "result 0", "result 1", "result 2",
    ]


@run_async
async def test_cancellation_preserves_completed_child_and_resumes_only_interrupted_child(
    tmp_path: Path,
) -> None:
    import asyncio

    calls = (
        ToolCall("done", "delegate", {"agent_spec_id": "explorer", "goal": "done"}),
        ToolCall("pending", "delegate", {"agent_spec_id": "explorer", "goal": "pending"}),
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("joined")], [])
    session, store, parent, main_spec, _, model, _, registry, delegation = setup
    started = asyncio.Event()
    cancelled = asyncio.Event()
    requests: list[str] = []

    class InterruptibleModel:
        async def complete(self, context, _tools) -> ModelTurn:
            requests.append(context.goal)
            if context.goal == "pending":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return _final("completed")

    delegation.model_for_spec = lambda _spec: InterruptibleModel()
    task = asyncio.create_task(start_run(
        session, parent, model, registry, store=store, agent_spec=main_spec,
        delegation=delegation,
    ))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    restored_session = store.load_session(session.session_id)
    restored_parent = store.load_run(parent.run_id)
    assert restored_parent.status is RunStatus.WAITING
    children = [run for run in store.list_runs(session.session_id) if run.parent_run_id]
    assert [child.status for child in children] == [RunStatus.COMPLETED, RunStatus.RUNNING]
    resumed = ScriptedModel([_final("recovered")])
    recovered_delegation = DelegationRuntime(
        delegation.agent_specs, lambda _spec: resumed, ActiveContextManagers()
    )
    await continue_delegation(
        restored_session, restored_parent, model, registry, store=store,
        agent_spec=main_spec, delegation=recovered_delegation,
    )
    assert restored_parent.status is RunStatus.COMPLETED
    assert [context.goal for context in resumed.contexts] == ["pending"]
    results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
    assert [json.loads(block["content"])["summary"] for block in results] == [
        "completed", "recovered",
    ]
    assert requests == ["done", "pending"]


@run_async
async def test_storage_failure_interrupts_inflight_sibling_without_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from keyline_agent.storage.errors import StorageError

    calls = tuple(
        ToolCall(goal, "delegate", {"agent_spec_id": "explorer", "goal": goal})
        for goal in ("blocked", "fail")
    )
    setup = _setup(tmp_path, [_tool_turn(*calls)], [])
    session, store, parent, main_spec, _, model, _, registry, delegation = setup
    started = asyncio.Event()
    cancelled = asyncio.Event()
    checkpoint = store.checkpoint_run

    def fail_checkpoint(current_session, run, messages=()):
        if run.goal == "fail":
            raise StorageError("disk unavailable")
        return checkpoint(current_session, run, messages)

    monkeypatch.setattr(store, "checkpoint_run", fail_checkpoint)

    class FailingBatchModel:
        async def complete(self, context, _tools) -> ModelTurn:
            if context.goal == "blocked":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            await started.wait()
            return _final("result")

    delegation.model_for_spec = lambda _spec: FailingBatchModel()
    with pytest.raises(StorageError, match="disk unavailable"):
        await asyncio.wait_for(start_run(
            session, parent, model, registry, store=store, agent_spec=main_spec,
            delegation=delegation,
        ), 1)
    assert cancelled.is_set()
    assert store.load_run(parent.run_id).status is RunStatus.WAITING
    assert len(tuple(store.read_messages(parent.context_scope_id))) == 2
    assert all(child.status is RunStatus.RUNNING for child in store.list_runs(session.session_id)
               if child.parent_run_id)


@pytest.mark.parametrize("outcome", ("completed", "failed", "waiting"))
@run_async
async def test_child_cleanup_failure_keeps_outcome_and_sibling_progress(
    tmp_path: Path, outcome: str, caplog: pytest.LogCaptureFixture,
) -> None:
    import asyncio

    from keyline_agent.models.errors import ModelProviderError

    calls = tuple(
        ToolCall(goal, "delegate", {"agent_spec_id": "explorer", "goal": goal})
        for goal in ("first", "second")
    )
    setup = _setup(tmp_path, [_tool_turn(*calls), _final("joined")], [])
    session, store, parent, main_spec, _, model, _, registry, delegation = setup
    closed = asyncio.Event()
    secret = "cleanup-exception-canary"

    class ClosingModel:
        async def complete(self, context, _tools) -> ModelTurn:
            if context.goal == "first":
                if outcome == "failed":
                    raise ModelProviderError("Model failed")
                if outcome == "waiting":
                    return _tool_turn(ToolCall("wait", "wait_for_text", {"prompt": "Question"}))
                return _final("first result")
            await asyncio.wait_for(closed.wait(), 1)
            return _final("second result")

        async def aclose(self) -> None:
            closed.set()
            raise RuntimeError(secret)

    delegation.model_for_spec = lambda _spec: ClosingModel()
    await start_run(
        session, parent, model, registry, store=store, agent_spec=main_spec,
        delegation=delegation,
    )
    children = [run for run in store.list_runs(session.session_id) if run.parent_run_id]
    assert children[0].status.value == outcome
    assert children[1].status is RunStatus.COMPLETED
    assert parent.status is (
        RunStatus.WAITING if outcome == "waiting" else RunStatus.COMPLETED
    )
    assert secret not in caplog.text
    assert "Model resource cleanup failed (RuntimeError)" in caplog.text
    if outcome != "waiting":
        results = tuple(store.read_messages(parent.context_scope_id))[2].message.content
        assert json.loads(results[0]["content"])["status"] == outcome
        assert json.loads(results[1]["content"])["summary"] == "second result"


@run_async
async def test_continue_delegation_cannot_reenter_a_running_parent(tmp_path: Path) -> None:
    import asyncio

    from keyline_agent.runtime.errors import AgentLoopError

    call = ToolCall("child", "delegate", {"agent_spec_id": "explorer", "goal": "child"})
    setup = _setup(tmp_path, [_tool_turn(call), _final("joined")], [])
    session, store, parent, main_spec, _, model, _, registry, delegation = setup
    started = asyncio.Event()
    release = asyncio.Event()

    class WaitingModel:
        async def complete(self, _context, _tools) -> ModelTurn:
            started.set()
            await release.wait()
            return _final("child result")

    delegation.model_for_spec = lambda _spec: WaitingModel()
    task = asyncio.create_task(start_run(
        session, parent, model, registry, store=store, agent_spec=main_spec,
        delegation=delegation,
    ))
    await asyncio.wait_for(started.wait(), 1)
    try:
        with pytest.raises(AgentLoopError, match="already has an active run"):
            await continue_delegation(
                session, store.load_run(parent.run_id), model, registry,
                store=store, agent_spec=main_spec, delegation=delegation,
            )
        assert len(model.contexts) == 1
    finally:
        release.set()
        await task
    assert parent.status is RunStatus.COMPLETED
