import sys
from pathlib import Path

import pytest

import handshake.runtime.loop as runtime_loop
from handshake.context import SequencedMessage
from handshake.inputs import EphemeralSecretStore
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import ModelMessage, ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.dependencies import resolve_text_dependency
from handshake.runtime.errors import (
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.loop import continue_run, start_run
from handshake.runtime.persistence import RunStore
from handshake.runtime.runs import create_followup_run
from handshake.runtime.secret_dependencies import resolve_secret_dependency
from handshake.runtime.state import (
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    Session,
)
from handshake.storage.errors import StorageError
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry
from handshake.tools.secret_command import SecretCommandRunner
from tests.history_support import TEST_AGENT_SPEC, fresh_store, history, run_async


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


def _session_run(
    tmp_path: Path, goal: str, *, max_steps: int = 50
) -> tuple[Session, Run, RunStore]:
    session = Session(workspace=tmp_path)
    run = Run(
        goal=goal,
        session_id=session.session_id,
        model_name="scripted",
        max_steps=max_steps,
        agent_spec_id="test",
    )
    return session, run, fresh_store(tmp_path, session)


class _RecordingStore(SQLiteStateStore):
    def __init__(
        self, events: list[str], database: Path, *, fail_checkpoint: bool = False
    ) -> None:
        super().__init__(database)
        self.events = events
        self.fail_checkpoint = fail_checkpoint
        self.snapshots: list[
            tuple[RunStatus, DependencyStatus | None, int]
        ] = []

    def create_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        self.events.append("create-run")
        return super().create_run(session, run, messages)

    def checkpoint_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        dependency_status = (
            run.dependencies[-1].status if run.dependencies else None
        )
        self.snapshots.append(
            (
                run.status,
                dependency_status,
                self.message_count(run.context_scope_id) + len(messages),
            )
        )
        self.events.append(f"checkpoint:{run.status.value}")
        if self.fail_checkpoint:
            raise StorageError("checkpoint unavailable")
        return super().checkpoint_run(session, run, messages)


class _FailingIngress:
    def ingest_messages(
        self,
        _run: Run,
        _messages: tuple[SequencedMessage, ...],
    ) -> None:
        raise RuntimeError("view ingress failed")


def test_checkpoint_clears_committed_pending_before_view_ingress(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "persist once")
    store.create_run(session, run)
    pending = [
        ModelMessage(
            role="user",
            content=({"type": "text", "text": run.goal},),
            run_id=run.run_id,
        )
    ]

    with pytest.raises(RuntimeError, match="view ingress failed"):
        runtime_loop._checkpoint(
            session,
            run,
            store,
            _FailingIngress(),  # type: ignore[arg-type]
            pending,
        )

    assert pending == []
    assert store.message_count(run.context_scope_id) == 1


@run_async
async def test_multiple_tools_execute_in_order_and_results_return_to_model(tmp_path: Path) -> None:
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
    session, run, store = _session_run(tmp_path, "copy knowledge")

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.COMPLETED
    assert [tool_run.tool_name for tool_run in run.tool_runs] == [
        "read_file",
        "write_file",
    ]
    tool_results = model.contexts[1].messages[-1].content
    assert [block["tool_use_id"] for block in tool_results] == ["read", "write"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new"


@run_async
async def test_unknown_tool_is_structured_failure_and_loop_can_finish(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool_turn(ToolCall("missing", "not_registered", {})), final_turn()]
    )
    session, run, store = _session_run(tmp_path, "recover")

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.tool_runs[0].result is not None
    assert not run.tool_runs[0].result.ok
    assert "Unknown tool" in run.tool_runs[0].result.stderr


@run_async
async def test_mismatched_tool_blocks_fail_run(tmp_path: Path) -> None:
    bad_turn = ModelTurn(
        content=({"type": "tool_use", "id": "a", "name": "read_file", "input": {}},),
        tool_calls=(ToolCall("b", "read_file", {}),),
        stop_reason="tool_use",
    )
    session, run, store = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProtocolError):
        await start_run(
            session,
            run,
            ScriptedModel([bad_turn]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )

    assert run.status is RunStatus.FAILED


@run_async
async def test_script_exhaustion_fails_run(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProviderError):
        await start_run(
            session,
            run,
            ScriptedModel([]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )

    assert run.status is RunStatus.FAILED


@run_async
async def test_step_limit_is_owned_by_run(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "too long", max_steps=1)
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
        await start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )

    assert run.status is RunStatus.FAILED


@run_async
async def test_simple_run_can_edit_and_validate_without_a_plan(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("old", encoding="utf-8")
    session, run, store = _session_run(tmp_path, "replace one value")
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

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.plan == []
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "new"
    assert run.tool_runs[-1].result is not None
    assert run.tool_runs[-1].result.ok


@run_async
async def test_plan_tool_describes_adaptive_planning(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "small request")
    model = ScriptedModel([final_turn()])

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    plan_tool = next(tool for tool in model.tool_sets[0] if tool.name == "update_plan")
    assert "non-trivial, multi-step work" in plan_tool.description
    assert "Do not call this tool for simple work" in plan_tool.description


@run_async
async def test_cancelled_tool_closes_protocol_before_run_terminates(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run, store = _session_run(tmp_path, "cancel")
    model = ScriptedModel(
        [tool_turn(ToolCall("shell", "run_shell", {"argv": ["true"]}))]
    )

    with pytest.raises(UserCancelledError):
        await start_run(
            session, run, model, create_default_registry(cancel),
            store=store, agent_spec=TEST_AGENT_SPEC,
        )

    assert run.status is RunStatus.CANCELLED
    result = history(store, session)[-1].content[0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "shell"
    assert result["is_error"] is True


@run_async
async def test_interrupted_multi_tool_turn_closes_every_call(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run, store = _session_run(tmp_path, "several tools")
    calls = (
        ToolCall("write-a", "write_file", {"path": "a.txt", "content": "done"}),
        ToolCall("shell", "run_shell", {"argv": ["true"]}),
        ToolCall("write-b", "write_file", {"path": "b.txt", "content": "skipped"}),
    )

    with pytest.raises(UserCancelledError):
        await start_run(
            session,
            run,
            ScriptedModel([tool_turn(*calls)]),
            create_default_registry(cancel),
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )

    results = history(store, session)[-1].content
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


@run_async
async def test_followup_run_receives_complete_prior_protocol_history(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("value", encoding="utf-8")
    session, first, store = _session_run(tmp_path, "inspect")
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
    await start_run(
        session, first, first_model, registry, store=store, agent_spec=TEST_AGENT_SPEC
    )
    second = create_followup_run(session, [first], "continue")
    second_model = ScriptedModel([final_turn("second complete")])

    await start_run(
        session, second, second_model, registry, store=store, agent_spec=TEST_AGENT_SPEC
    )

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


@run_async
async def test_text_dependency_closes_tool_call_before_continuation(tmp_path: Path) -> None:
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
    session, run, store = _session_run(tmp_path, "style the button")
    first_model = ScriptedModel([wait_turn])

    await start_run(
        session,
        run,
        first_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.WAITING
    assert run.dependencies[0].status is DependencyStatus.OPEN
    immediate_result = history(store, session)[-1].content[0]
    assert immediate_result["type"] == "tool_result"
    assert immediate_result["tool_use_id"] == "ask-color"
    assert immediate_result["is_error"] is False
    assert run.dependencies[0].dependency_id in immediate_result["content"]
    wait_tool = next(tool for tool in first_model.tool_sets[0] if tool.name == "wait_for_text")
    assert wait_tool.input_schema["required"] == ["prompt"]
    assert "API keys" in wait_tool.description

    resumed_session = store.load_session(session.session_id)
    resumed_run = store.load_run(run.run_id)
    continuation = resolve_text_dependency(
        resumed_session,
        resumed_run,
        resumed_run.dependencies[0].dependency_id,
        "navy",
    )
    second_model = ScriptedModel([final_turn("used navy")])
    await continue_run(
        resumed_session,
        resumed_run,
        second_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        pending_messages=(continuation,),
        agent_spec=TEST_AGENT_SPEC,
    )

    context = second_model.contexts[0]
    assert context.messages[-3].content[0]["thinking"] == (
        "The repository cannot answer this."
    )
    assert context.messages[-2].content[0]["tool_use_id"] == "ask-color"
    continuation = context.messages[-1].content[0]
    assert continuation["type"] == "text"
    assert "navy" in continuation["text"]
    assert resumed_run.dependencies[0].dependency_id in continuation["text"]
    assert sum(
        block.get("tool_use_id") == "ask-color"
        for message in context.messages
        for block in message.content
    ) == 1
    assert resumed_run.status is RunStatus.COMPLETED
    assert store.load_run(run.run_id).final_summary == "used navy"


@run_async
async def test_multiple_waits_resume_only_after_every_dependency_is_resolved(
    tmp_path: Path,
) -> None:
    calls = (
        ToolCall("ask-color", "wait_for_text", {"prompt": "Which color?"}),
        ToolCall(
            "ask-key",
            "wait_for_secret",
            {
                "prompt": "Integration key?",
                "purpose": "run a local test",
                "environment_variable": "TEST_API_KEY",
            },
        ),
        ToolCall("ask-size", "wait_for_text", {"prompt": "Which size?"}),
    )
    session, run, store = _session_run(tmp_path, "configure the test")
    secret_store = EphemeralSecretStore()
    runner = SecretCommandRunner(
        secret_store.consumer(), lambda _cwd, _argv: True
    )
    await start_run(
        session,
        run,
        ScriptedModel([tool_turn(*calls)]),
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.WAITING
    assert [dependency.kind for dependency in run.dependencies] == [
        DependencyKind.TEXT,
        DependencyKind.SECRET,
        DependencyKind.TEXT,
    ]
    assert [block["tool_use_id"] for block in history(store, session)[-1].content] == [
        call.id for call in calls
    ]
    assert all(not block["is_error"] for block in history(store, session)[-1].content)

    loaded_session = store.load_session(session.session_id)
    loaded_run = store.load_run(run.run_id)
    first_answer = resolve_text_dependency(
        loaded_session, loaded_run, loaded_run.dependencies[0].dependency_id, "navy"
    )
    assert loaded_run.status is RunStatus.WAITING
    await continue_run(
        loaded_session,
        loaded_run,
        ScriptedModel([]),
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        pending_messages=(first_answer,),
        agent_spec=TEST_AGENT_SPEC,
    )
    assert store.load_run(run.run_id).status is RunStatus.WAITING
    assert len(store.load_run(run.run_id).open_dependencies()) == 2

    key_value = "multi-wait-canary"
    loaded_run = store.load_run(run.run_id)
    key_answer = resolve_secret_dependency(
        loaded_session,
        loaded_run,
        loaded_run.dependencies[1].dependency_id,
        key_value,
        secret_store.writer(),
    )
    assert loaded_run.status is RunStatus.WAITING
    await continue_run(
        loaded_session,
        loaded_run,
        ScriptedModel([]),
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        pending_messages=(key_answer,),
        secret_command_runner=runner,
        agent_spec=TEST_AGENT_SPEC,
    )
    assert key_value.encode() not in store.database_path.read_bytes()
    assert len(store.load_run(run.run_id).open_dependencies()) == 1

    loaded_run = store.load_run(run.run_id)
    last_answer = resolve_text_dependency(
        loaded_session, loaded_run, loaded_run.dependencies[2].dependency_id, "large"
    )
    assert loaded_run.status is RunStatus.RUNNING
    final_model = ScriptedModel([final_turn("configured")])
    await continue_run(
        loaded_session,
        loaded_run,
        final_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        pending_messages=(last_answer,),
        secret_command_runner=runner,
        agent_spec=TEST_AGENT_SPEC,
    )
    assert loaded_run.status is RunStatus.COMPLETED
    assert loaded_run.open_dependencies() == ()
    assert [dependency.status for dependency in loaded_run.dependencies] == [
        DependencyStatus.SATISFIED
    ] * 3
    assert len(final_model.contexts) == 1
    assert all(
        answer in repr(final_model.contexts[0].messages)
        for answer in ("navy", "large")
    )
    assert key_value not in repr(final_model.contexts[0].messages)


@run_async
async def test_invalid_wait_rejects_entire_wait_batch(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            tool_turn(
                ToolCall("valid", "wait_for_text", {"prompt": "Which color?"}),
                ToolCall("invalid", "wait_for_text", {"prompt": "  "}),
            ),
            final_turn("recovered"),
        ]
    )
    session, run, store = _session_run(tmp_path, "choose color")

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.dependencies == []
    rejected = model.contexts[1].messages[-1].content
    assert [block["tool_use_id"] for block in rejected] == ["valid", "invalid"]
    assert all(block["is_error"] for block in rejected)


@run_async
async def test_wait_and_resolution_are_checkpointed_before_they_are_reported(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    session, run, _unused_store = _session_run(tmp_path, "choose a color")
    store = _RecordingStore(events, tmp_path / "recording.db")
    store.create_session(session)
    wait_model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}))]
    )

    await start_run(
        session,
        run,
        wait_model,
        create_default_registry(lambda _cwd, _argv: True),
        emit=events.append,
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert events.index("checkpoint:waiting") < events.index(
        "Waiting for text: Which color?"
    )
    continuation = resolve_text_dependency(
        session, run, run.dependencies[0].dependency_id, "navy"
    )
    await continue_run(
        session,
        run,
        ScriptedModel([final_turn("used navy")]),
        create_default_registry(lambda _cwd, _argv: True),
        emit=events.append,
        store=store,
        pending_messages=(continuation,),
        agent_spec=TEST_AGENT_SPEC,
    )

    assert store.snapshots == [
        (RunStatus.WAITING, DependencyStatus.OPEN, 3),
        (RunStatus.RUNNING, DependencyStatus.SATISFIED, 4),
        (RunStatus.COMPLETED, DependencyStatus.SATISFIED, 5),
    ]
    assert events.index("checkpoint:running") < events.index(
        "Input received; resuming run"
    )
    assert events.index("checkpoint:completed") < events.index("Run completed")


@run_async
async def test_checkpoint_failure_is_not_reclassified_as_run_failure(tmp_path: Path) -> None:
    events: list[str] = []
    session, run, _unused_store = _session_run(tmp_path, "choose a color")
    store = _RecordingStore(
        events, tmp_path / "recording-failure.db", fail_checkpoint=True
    )
    store.create_session(session)
    model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}))]
    )

    with pytest.raises(StorageError, match="checkpoint unavailable"):
        await start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            emit=events.append,
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )

    assert run.status is RunStatus.WAITING
    assert run.error is None
    assert "Waiting for text: Which color?" not in events


@run_async
async def test_mixed_wait_turn_rejects_every_call_without_side_effect(tmp_path: Path) -> None:
    mixed = tool_turn(
        ToolCall("write", "write_file", {"path": "created.txt", "content": "bad"}),
        ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}),
    )
    model = ScriptedModel([mixed, final_turn("corrected")])
    session, run, store = _session_run(tmp_path, "safe")

    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )

    assert not (tmp_path / "created.txt").exists()
    assert run.status is RunStatus.COMPLETED
    errors = model.contexts[1].messages[-1].content
    assert len(errors) == 2
    assert all(block["is_error"] for block in errors)


@run_async
async def test_model_turn_budget_is_preserved_across_wait(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "ask once", max_steps=1)
    model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}))]
    )
    await start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        agent_spec=TEST_AGENT_SPEC,
    )
    resumed_session = store.load_session(session.session_id)
    resumed_run = store.load_run(run.run_id)
    continuation = resolve_text_dependency(
        resumed_session,
        resumed_run,
        resumed_run.dependencies[0].dependency_id,
        "A",
    )

    with pytest.raises(StepLimitExceededError):
        await continue_run(
            resumed_session,
            resumed_run,
            ScriptedModel([final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            pending_messages=(continuation,),
            agent_spec=TEST_AGENT_SPEC,
        )

    assert resumed_run.status is RunStatus.FAILED
