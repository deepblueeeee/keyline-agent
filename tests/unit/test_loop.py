import sys
from pathlib import Path

import pytest

import handshake.runtime.loop as runtime_loop
from handshake.context import SequencedMessage
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import ModelMessage, ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.dependencies import resolve_text_dependency
from handshake.runtime.errors import (
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.loop import continue_run, start_run
from handshake.runtime.runs import create_followup_run
from handshake.runtime.state import (
    DependencyStatus,
    Run,
    RunStatus,
    Session,
    WorkItemStatus,
)
from handshake.storage.errors import StorageError
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry
from tests.history_support import fresh_store, history


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
) -> tuple[Session, Run, SQLiteStateStore]:
    session = Session(workspace=tmp_path)
    run = Run(
        goal=goal,
        session_id=session.session_id,
        model_name="scripted",
        max_steps=max_steps,
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
            tuple[RunStatus, WorkItemStatus, DependencyStatus | None, int]
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
                run.work_items[0].status,
                dependency_status,
                self.message_count(session.session_id) + len(messages),
            )
        )
        self.events.append(f"checkpoint:{run.status.value}")
        if self.fail_checkpoint:
            raise StorageError("checkpoint unavailable")
        return super().checkpoint_run(session, run, messages)


class _FailingIngress:
    def ingest_messages(
        self,
        _session: Session,
        _messages: tuple[SequencedMessage, ...],
    ) -> None:
        raise RuntimeError("view ingress failed")


def test_checkpoint_clears_committed_pending_before_view_ingress(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "persist once")
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
            create=True,
        )

    assert pending == []
    assert store.message_count(session.session_id) == 1


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
    session, run, store = _session_run(tmp_path, "copy knowledge")

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert run.status is RunStatus.COMPLETED
    assert [tool_run.tool_name for tool_run in run.tool_runs] == [
        "read_file",
        "write_file",
    ]
    assert {tool_run.work_id for tool_run in run.tool_runs} == {
        run.work_items[0].work_id
    }
    assert run.work_items[0].status is WorkItemStatus.COMPLETED
    tool_results = model.contexts[1].messages[-1].content
    assert [block["tool_use_id"] for block in tool_results] == ["read", "write"]
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "new"


def test_unknown_tool_is_structured_failure_and_loop_can_finish(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool_turn(ToolCall("missing", "not_registered", {})), final_turn()]
    )
    session, run, store = _session_run(tmp_path, "recover")

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

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
    session, run, store = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProtocolError):
        start_run(
            session,
            run,
            ScriptedModel([bad_turn]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
        )

    assert run.status is RunStatus.FAILED


def test_script_exhaustion_fails_run(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "bad")

    with pytest.raises(ModelProviderError):
        start_run(
            session,
            run,
            ScriptedModel([]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
        )

    assert run.status is RunStatus.FAILED


def test_step_limit_is_owned_by_run(tmp_path: Path) -> None:
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
        start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
        )

    assert run.status is RunStatus.FAILED


def test_simple_run_can_edit_and_validate_without_a_plan(tmp_path: Path) -> None:
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

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.plan == []
    assert (tmp_path / "value.txt").read_text(encoding="utf-8") == "new"
    assert run.tool_runs[-1].result is not None
    assert run.tool_runs[-1].result.ok


def test_plan_tool_describes_adaptive_planning(tmp_path: Path) -> None:
    session, run, store = _session_run(tmp_path, "small request")
    model = ScriptedModel([final_turn()])

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    plan_tool = next(tool for tool in model.tool_sets[0] if tool.name == "update_plan")
    assert "non-trivial, multi-step work" in plan_tool.description
    assert "Do not call this tool for simple work" in plan_tool.description


def test_cancelled_tool_closes_protocol_before_run_terminates(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run, store = _session_run(tmp_path, "cancel")
    model = ScriptedModel(
        [tool_turn(ToolCall("shell", "run_shell", {"argv": ["true"]}))]
    )

    with pytest.raises(UserCancelledError):
        start_run(session, run, model, create_default_registry(cancel), store=store)

    assert run.status is RunStatus.CANCELLED
    result = history(store, session)[-1].content[0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "shell"
    assert result["is_error"] is True


def test_interrupted_multi_tool_turn_closes_every_call(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    session, run, store = _session_run(tmp_path, "several tools")
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
            store=store,
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


def test_followup_run_receives_complete_prior_protocol_history(tmp_path: Path) -> None:
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
    start_run(session, first, first_model, registry, store=store)
    second = create_followup_run(session, [first], "continue")
    second_model = ScriptedModel([final_turn("second complete")])

    start_run(session, second, second_model, registry, store=store)

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


def test_text_dependency_closes_tool_call_before_continuation(tmp_path: Path) -> None:
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

    start_run(
        session,
        run,
        first_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
    )

    assert run.status is RunStatus.WAITING
    assert run.work_items[0].status is WorkItemStatus.READY
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
    continue_run(
        resumed_session,
        resumed_run,
        second_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=store,
        pending_messages=(continuation,),
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
    assert resumed_run.work_items[0].status is WorkItemStatus.COMPLETED
    assert store.load_run(run.run_id).final_summary == "used navy"


def test_wait_and_resolution_are_checkpointed_before_they_are_reported(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    session, run, _unused_store = _session_run(tmp_path, "choose a color")
    store = _RecordingStore(events, tmp_path / "recording.db")
    store.create_session(session)
    wait_model = ScriptedModel(
        [tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}))]
    )

    start_run(
        session,
        run,
        wait_model,
        create_default_registry(lambda _cwd, _argv: True),
        emit=events.append,
        store=store,
    )

    assert events.index("checkpoint:waiting") < events.index(
        "Waiting for text: Which color?"
    )
    continuation = resolve_text_dependency(
        session, run, run.dependencies[0].dependency_id, "navy"
    )
    continue_run(
        session,
        run,
        ScriptedModel([final_turn("used navy")]),
        create_default_registry(lambda _cwd, _argv: True),
        emit=events.append,
        store=store,
        pending_messages=(continuation,),
    )

    assert store.snapshots == [
        (RunStatus.WAITING, WorkItemStatus.READY, DependencyStatus.OPEN, 3),
        (RunStatus.RUNNING, WorkItemStatus.READY, DependencyStatus.SATISFIED, 4),
        (RunStatus.COMPLETED, WorkItemStatus.COMPLETED, DependencyStatus.SATISFIED, 5),
    ]
    assert events.index("checkpoint:running") < events.index(
        "Input received; resuming run"
    )
    assert events.index("checkpoint:completed") < events.index("Run completed")


def test_checkpoint_failure_is_not_reclassified_as_run_failure(tmp_path: Path) -> None:
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
        start_run(
            session,
            run,
            model,
            create_default_registry(lambda _cwd, _argv: True),
            emit=events.append,
            store=store,
        )

    assert run.status is RunStatus.WAITING
    assert run.work_items[0].status is WorkItemStatus.READY
    assert run.error is None
    assert "Waiting for text: Which color?" not in events


def test_mixed_wait_turn_rejects_every_call_without_side_effect(tmp_path: Path) -> None:
    mixed = tool_turn(
        ToolCall("write", "write_file", {"path": "created.txt", "content": "bad"}),
        ToolCall("ask", "wait_for_text", {"prompt": "Choose?"}),
    )
    model = ScriptedModel([mixed, final_turn("corrected")])
    session, run, store = _session_run(tmp_path, "safe")

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
    session, run, store = _session_run(tmp_path, "ask once", max_steps=1)
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
    continuation = resolve_text_dependency(
        resumed_session,
        resumed_run,
        resumed_run.dependencies[0].dependency_id,
        "A",
    )

    with pytest.raises(StepLimitExceededError):
        continue_run(
            resumed_session,
            resumed_run,
            ScriptedModel([final_turn()]),
            create_default_registry(lambda _cwd, _argv: True),
            store=store,
            pending_messages=(continuation,),
        )

    assert resumed_run.status is RunStatus.FAILED
