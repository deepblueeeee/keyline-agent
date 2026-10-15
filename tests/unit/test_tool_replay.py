from pathlib import Path

import pytest

from handshake.models.base import ToolDefinition
from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.agents import AgentSpec
from handshake.runtime.errors import AgentLoopError, ToolOutcomeUnknownError
from handshake.runtime.loop import start_run
from handshake.runtime.recovery import resume_run
from handshake.runtime.state import ReplayPolicy, Run, RunStatus, Session, ToolResult, ToolRunStatus
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools.registry import ToolBinding, ToolRegistry
from tests.history_support import run_async


def _turn(call: ToolCall) -> ModelTurn:
    return ModelTurn(
        content=({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments},),
        tool_calls=(call,),
        stop_reason="tool_use",
    )


def _final() -> ModelTurn:
    return ModelTurn(content=({"type": "text", "text": "done"},), stop_reason="end_turn")


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_calls"),
    [
        (ReplayPolicy.AUTO, RunStatus.COMPLETED, 2),
        (ReplayPolicy.MANUAL, RunStatus.FAILED, 1),
    ],
)
@run_async
async def test_recovery_replays_only_safe_unknown_tool(
    tmp_path: Path,
    policy: ReplayPolicy,
    expected_status: RunStatus,
    expected_calls: int,
) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = Run("test replay", session.session_id, "test")
    spec = AgentSpec("test", "scripted", "Test replay.", allowed_tools={"custom"})
    count = 0

    def handler(_workspace: Path, _arguments: object) -> ToolResult:
        nonlocal count
        count += 1
        if count == 1:
            raise KeyboardInterrupt
        return ToolResult(ok=True, stdout="completed")

    registry = ToolRegistry()
    registry.register(
        ToolBinding(
            ToolDefinition("custom", "Test a tool", {"type": "object"}),
            handler,
            replay_policy=policy,
        )
    )
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session, run, ScriptedModel([_turn(ToolCall("call", "custom", {}))]),
            registry, store=store, agent_spec=spec,
        )
    persisted = store.load_run(run.run_id)
    assert persisted.status is RunStatus.RUNNING
    assert persisted.tool_runs[0].status is ToolRunStatus.RUNNING
    assert len(tuple(store.read_messages(run.context_scope_id))) == 2

    restored_session = store.load_session(session.session_id)
    if policy is ReplayPolicy.MANUAL:
        with pytest.raises(ToolOutcomeUnknownError):
            await resume_run(
                restored_session, persisted, ScriptedModel([]), registry,
                store=store, agent_spec=spec,
            )
        assert persisted.tool_runs[0].status is ToolRunStatus.UNKNOWN
        assert "not retried automatically" in (persisted.error or "")
    else:
        await resume_run(
            restored_session, persisted, ScriptedModel([_final()]), registry,
            store=store, agent_spec=spec,
        )
        assert persisted.tool_runs[0].status is ToolRunStatus.SUCCEEDED
    assert count == expected_calls
    assert persisted.status is expected_status
    assert store.load_run(run.run_id).status is expected_status


@run_async
async def test_completed_tool_result_survives_interruption_before_protocol_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = Run("test completed result", session.session_id, "test")
    spec = AgentSpec("test", "scripted", "Test replay.", allowed_tools={"custom"})
    count = 0

    def handler(_workspace: Path, _arguments: object) -> ToolResult:
        nonlocal count
        count += 1
        return ToolResult(ok=True, stdout="once")

    registry = ToolRegistry()
    registry.register(
        ToolBinding(
            ToolDefinition("custom", "Test a tool", {"type": "object"}),
            handler,
            replay_policy=ReplayPolicy.MANUAL,
        )
    )
    import handshake.runtime.loop as runtime_loop

    original_append = runtime_loop._append_tool_results

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_loop, "_append_tool_results", interrupt)
    with pytest.raises(KeyboardInterrupt):
        await start_run(
            session, run, ScriptedModel([_turn(ToolCall("call", "custom", {}))]),
            registry, store=store, agent_spec=spec,
        )
    monkeypatch.setattr(runtime_loop, "_append_tool_results", original_append)
    persisted = store.load_run(run.run_id)
    assert persisted.tool_runs[0].status is ToolRunStatus.SUCCEEDED
    await resume_run(
        store.load_session(session.session_id), persisted,
        ScriptedModel([_final()]), registry, store=store, agent_spec=spec,
    )
    assert count == 1
    assert persisted.status is RunStatus.COMPLETED


@run_async
async def test_handler_exception_keeps_tool_outcome_unknown(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = Run("test handler exception", session.session_id, "test")
    spec = AgentSpec("test", "scripted", "Test tools.", allowed_tools={"custom"})

    def handler(_workspace: Path, _arguments: object) -> ToolResult:
        raise RuntimeError("handler failed after an uncertain side effect")

    registry = ToolRegistry()
    registry.register(
        ToolBinding(ToolDefinition("custom", "Test a tool", {"type": "object"}), handler)
    )
    with pytest.raises(AgentLoopError):
        await start_run(
            session, run, ScriptedModel([_turn(ToolCall("call", "custom", {}))]),
            registry, store=store, agent_spec=spec,
        )
    persisted = store.load_run(run.run_id)
    assert persisted.status is RunStatus.FAILED
    assert persisted.tool_runs[0].status is ToolRunStatus.UNKNOWN


@pytest.mark.parametrize("policy", (ReplayPolicy.AUTO, ReplayPolicy.MANUAL))
@run_async
async def test_cancelled_tool_saves_its_outcome_before_recovery(
    tmp_path: Path, policy: ReplayPolicy,
) -> None:
    import asyncio
    import threading

    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = Run("cancel a tool", session.session_id, "test")
    spec = AgentSpec("test", "scripted", "Test cancellation.", allowed_tools={"custom"})
    started = threading.Event()
    release = threading.Event()
    calls = 0
    checkpoint_threads: list[int] = []
    checkpoint = store.checkpoint_run

    def record_checkpoint(*args, **kwargs):
        checkpoint_threads.append(threading.get_ident())
        return checkpoint(*args, **kwargs)

    store.checkpoint_run = record_checkpoint

    def handler(_workspace, _arguments) -> ToolResult:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        return ToolResult(ok=True)

    registry = ToolRegistry()
    registry.register(ToolBinding(
        ToolDefinition("custom", "Test cancellation", {"type": "object"}), handler,
        replay_policy=policy,
    ))
    task = asyncio.create_task(start_run(
        session, run, ScriptedModel([_turn(ToolCall("call", "custom", {}))]),
        registry, store=store, agent_spec=spec,
    ))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    restored = store.load_run(run.run_id)
    assert restored.status is RunStatus.RUNNING
    assert restored.tool_runs[0].status is ToolRunStatus.SUCCEEDED
    assert len(tuple(store.read_messages(run.context_scope_id))) == 2
    await resume_run(
        store.load_session(session.session_id), restored, ScriptedModel([_final()]),
        registry, store=store, agent_spec=spec,
    )
    assert restored.status is RunStatus.COMPLETED
    assert calls == 1
    assert set(checkpoint_threads) == {threading.get_ident()}


@run_async
async def test_cancelling_queued_tool_does_not_persist_an_execution_intent(tmp_path: Path) -> None:
    import asyncio

    from handshake.runtime.scheduler import CooperativeScheduler

    session = Session(workspace=tmp_path)
    store = SQLiteStateStore(tmp_path / "state.db")
    store.create_session(session)
    run = Run("queued tool", session.session_id, "test")
    spec = AgentSpec("test", "scripted", "Test queuing.", allowed_tools={"custom"})
    registry = ToolRegistry()
    calls = 0

    def handler(_workspace, _arguments) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(ok=True)

    registry.register(ToolBinding(
        ToolDefinition("custom", "Test queuing", {"type": "object"}), handler,
    ))
    scheduler = CooperativeScheduler()
    async with scheduler.tool_slot():
        task = asyncio.create_task(start_run(
            session, run, ScriptedModel([_turn(ToolCall("call", "custom", {}))]),
            registry, store=store, agent_spec=spec, scheduler=scheduler,
        ))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    restored = store.load_run(run.run_id)
    assert not restored.tool_runs
    assert calls == 0
    await resume_run(
        store.load_session(session.session_id), restored, ScriptedModel([_final()]),
        registry, store=store, agent_spec=spec,
    )
    assert calls == 1
    assert restored.status is RunStatus.COMPLETED
