import asyncio
import threading
from pathlib import Path

import pytest

from keyline_agent.context.condenser import LLMHistoryCondenser
from keyline_agent.models.protocol import ModelContext, ModelTurn
from keyline_agent.runtime.errors import AgentLoopError
from keyline_agent.runtime.scheduler import CooperativeScheduler, run_blocking
from keyline_agent.runtime.state import Run, RunStatus, Session, ToolResult
from keyline_agent.tools.registry import ToolRegistry
from tests.history_support import run_async


def _context(goal: str) -> ModelContext:
    return ModelContext("instructions", goal, "/tmp", (), ())


class ControlledModel:
    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.releases: dict[str, asyncio.Event] = {}
        self.active = 0
        self.peak = 0

    async def complete(self, context, _tools) -> ModelTurn:
        goal = context.goal
        release = self.releases.setdefault(goal, asyncio.Event())
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.put_nowait(goal)
        try:
            await release.wait()
            return ModelTurn(
                content=({"type": "text", "text": '{"episodes":[]}'},),
                stop_reason="end_turn",
            )
        finally:
            self.active -= 1


@run_async
async def test_model_limit_is_shared_with_history_condensation() -> None:
    scheduler = CooperativeScheduler(max_model_requests=2)
    model = ControlledModel()
    agent = scheduler.model(model)
    condenser = LLMHistoryCondenser(scheduler.model(model))
    tasks = [
        asyncio.create_task(agent.complete(_context("first"), ())),
        asyncio.create_task(condenser.compact_raw((), ())),
        asyncio.create_task(agent.complete(_context("third"), ())),
    ]
    try:
        assert await asyncio.wait_for(model.started.get(), 1) == "first"
        assert await asyncio.wait_for(model.started.get(), 1) == ""
        assert model.active == 2
        assert model.started.empty()
        model.releases["first"].set()
        assert await asyncio.wait_for(model.started.get(), 1) == "third"
        assert model.peak == 2
    finally:
        for release in model.releases.values():
            release.set()
        await asyncio.gather(*tasks)
    assert model.active == 0


@run_async
async def test_cancelled_model_request_releases_its_capacity() -> None:
    scheduler = CooperativeScheduler(max_model_requests=1)
    model = ControlledModel()
    bound = scheduler.model(model)
    first = asyncio.create_task(bound.complete(_context("first"), ()))
    second = asyncio.create_task(bound.complete(_context("second"), ()))
    try:
        assert await asyncio.wait_for(model.started.get(), 1) == "first"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(model.started.get(), 1) == "second"
        model.releases["second"].set()
        await second
        assert model.peak == 1
    finally:
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@run_async
async def test_one_context_scope_has_only_one_active_run(tmp_path: Path) -> None:
    session = Session(workspace=tmp_path)
    run = Run("goal", session.session_id, "test")
    run.transition(RunStatus.RUNNING)
    scheduler = CooperativeScheduler()
    started = asyncio.Event()
    release = asyncio.Event()

    async def resume(current: Run) -> Run:
        started.set()
        await release.wait()
        return current

    first = asyncio.create_task(scheduler.advance_runnable(run, resume))
    await asyncio.wait_for(started.wait(), 1)
    try:
        with pytest.raises(AgentLoopError, match="already has an active run"):
            await scheduler.advance_runnable(run, resume)
    finally:
        release.set()
        await first
    await scheduler.advance_runnable(run, resume)


@run_async
async def test_blocking_tools_are_serial_and_do_not_block_model_requests() -> None:
    scheduler = CooperativeScheduler()
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    calls: list[str] = []

    def first_tool() -> ToolResult:
        calls.append("first")
        started.set()
        assert release.wait(2)
        return ToolResult(ok=True)

    def second_tool() -> ToolResult:
        calls.append("second")
        second_started.set()
        return ToolResult(ok=True)

    async def execute(tool) -> ToolResult:
        async with scheduler.tool_slot():
            return await run_blocking(tool)

    first = asyncio.create_task(execute(first_tool))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        second = asyncio.create_task(execute(second_tool))
        model = ControlledModel()
        request = asyncio.create_task(scheduler.model(model).complete(_context("model"), ()))
        assert await asyncio.wait_for(model.started.get(), 1) == "model"
        model.releases["model"].set()
        await request
        assert not second_started.is_set()
    finally:
        release.set()
        await first
        if second is not None:
            await second
    assert calls == ["first", "second"]


@run_async
async def test_cancelling_tool_keeps_slot_until_thread_finishes() -> None:
    scheduler = CooperativeScheduler()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    next_started = threading.Event()

    def tool() -> ToolResult:
        started.set()
        assert release.wait(2)
        finished.set()
        return ToolResult(ok=True)

    def next_tool() -> ToolResult:
        assert finished.is_set()
        next_started.set()
        return ToolResult(ok=True)

    async def execute(handler) -> ToolResult:
        async with scheduler.tool_slot():
            return await run_blocking(handler)

    first = asyncio.create_task(execute(tool))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        second = asyncio.create_task(execute(next_tool))
        await asyncio.sleep(0)
        assert not first.done()
        assert not next_started.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        if second is not None:
            await second
    assert next_started.is_set()


@pytest.mark.parametrize("entry", ("start", "resume", "continue"))
@pytest.mark.parametrize("scheduler_mode", ("shared", "new", "default"))
@run_async
async def test_runtime_entry_claims_scope_before_mutation(
    tmp_path: Path, entry: str, scheduler_mode: str,
) -> None:
    from dataclasses import replace

    from keyline_agent.models.scripted import ScriptedModel
    from keyline_agent.runtime.loop import continue_run, start_run
    from keyline_agent.runtime.recovery import resume_run
    from tests.history_support import TEST_AGENT_SPEC, fresh_store

    session = Session(workspace=tmp_path)
    store = fresh_store(tmp_path, session)
    run = Run("first", session.session_id, "test")
    created_copy = replace(run)
    scheduler = CooperativeScheduler()
    first_model = ControlledModel()
    first = asyncio.create_task(start_run(
        session, run, first_model, ToolRegistry(), store=store,
        agent_spec=TEST_AGENT_SPEC, scheduler=scheduler,
    ))
    await asyncio.wait_for(first_model.started.get(), 1)
    second_scheduler = (
        scheduler if scheduler_mode == "shared"
        else CooperativeScheduler() if scheduler_mode == "new" else None
    )
    second_model = ScriptedModel([])
    try:
        second_run = created_copy if entry == "start" else store.load_run(run.run_id)
        common = dict(store=store, agent_spec=TEST_AGENT_SPEC, scheduler=second_scheduler)
        with pytest.raises(AgentLoopError, match="already has an active run"):
            if entry == "start":
                await start_run(session, second_run, second_model, ToolRegistry(), **common)
            elif entry == "resume":
                await resume_run(session, second_run, second_model, ToolRegistry(), **common)
            else:
                await continue_run(
                    session, second_run, second_model, ToolRegistry(),
                    pending_messages=(), **common,
                )
        assert not second_model.contexts
        assert store.message_count(run.context_scope_id) == 1
    finally:
        first_model.releases["first"].set()
        await first
    assert store.message_count(run.context_scope_id) == 2
