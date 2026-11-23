"""Concurrent model requests and serial workspace tools for durable runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TypeVar
from weakref import WeakKeyDictionary

from handshake.models.base import ModelAdapter, ToolDefinition, close_model
from handshake.models.protocol import ModelContext, ModelTurn
from handshake.runtime.errors import AgentLoopError, UserCancelledError
from handshake.runtime.progress import run_identity
from handshake.runtime.runs import TERMINAL_RUN_STATUSES
from handshake.runtime.state import Run, RunStatus

_T = TypeVar("_T")

_SCOPE_OWNERS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Task[object]]
] = WeakKeyDictionary()


class CooperativeScheduler:
    """Run children cooperatively, limiting requests and serializing workspace tools."""

    def __init__(self, max_model_requests: int = 4) -> None:
        if type(max_model_requests) is not int or max_model_requests < 1:
            raise ValueError("max_model_requests must be an integer >= 1")
        self._requests = asyncio.Semaphore(max_model_requests)
        self._tools = asyncio.Lock()

    def model(self, model: ModelAdapter) -> ModelAdapter:
        """Apply the same request limit to agent turns and history condensation."""
        if isinstance(model, _ScheduledModel):
            if model.scheduler is self:
                return model
            model = model.delegate
        return _ScheduledModel(model, self)

    def runnable(self, children: Sequence[Run]) -> tuple[Run, ...]:
        return tuple(child for child in children if child.status is RunStatus.RUNNING)

    async def advance(
        self,
        children: Sequence[Run],
        resume: Callable[[Run], Awaitable[Run]],
        *, include_waiting: bool = False,
    ) -> tuple[Run, ...]:
        outcomes = {child.run_id: child for child in children}

        async def advance_child(child: Run) -> Run | BaseException:
            # Return global interruptions to the coordinator so it can stop and drain siblings.
            try:
                if child.status is RunStatus.WAITING:
                    with self.run_scope(child):
                        return await resume(child)
                return await self.advance_runnable(child, resume)
            except BaseException as exc:
                return exc

        tasks = {
            asyncio.create_task(advance_child(child), name=f"run:{child.run_id}"): child
            for child in children
            if child.status is RunStatus.RUNNING or (
                include_waiting and child.status is RunStatus.WAITING and child.open_dependencies()
            )
        }
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    outcome = task.result()
                    if isinstance(outcome, BaseException):
                        raise outcome
                    outcomes[tasks[task].run_id] = outcome
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        return tuple(outcomes[child.run_id] for child in children)

    def ready_to_join(self, children: Sequence[Run]) -> bool:
        return bool(children) and all(
            child.status in TERMINAL_RUN_STATUSES for child in children
        )

    async def advance_runnable(
        self, run: Run, resume: Callable[[Run], Awaitable[Run]]
    ) -> Run:
        if run.status is not RunStatus.RUNNING:
            raise AgentLoopError("Only a runnable run can advance")
        with self.run_scope(run):
            return await resume(run)

    @contextmanager
    def run_scope(self, run: Run) -> Iterator[None]:
        """Give one task exclusive ownership, including nested runtime entry points."""
        task = asyncio.current_task()
        if task is None:
            raise AgentLoopError("Run execution requires an asyncio task")
        owners = _SCOPE_OWNERS.setdefault(asyncio.get_running_loop(), {})
        scope_id = run.context_scope_id
        owner = owners.get(scope_id)
        if owner is not None:
            if owner is not task:
                raise AgentLoopError("Context scope already has an active run")
            with run_identity(run):
                yield
            return
        owners[scope_id] = task
        try:
            with run_identity(run):
                yield
        finally:
            del owners[scope_id]

    @asynccontextmanager
    async def tool_slot(self) -> AsyncIterator[None]:
        """Serialize intent, execution, and result persistence across workspace tools."""
        async with self._tools:
            yield

    def waiting_child(self, children: Sequence[Run]) -> Run | None:
        return next(
            (
                child for child in children
                if child.status is RunStatus.WAITING and child.open_dependencies()
            ),
            None,
        )


async def run_blocking(
    execute: Callable[[], _T], *, on_cancel: Callable[[], None] | None = None
) -> _T:
    """Run blocking I/O without abandoning a worker when its await is cancelled."""
    def invoke() -> _T | BaseException:
        try:
            return execute()
        except BaseException as exc:
            return exc

    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        outcome = await asyncio.shield(task)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        # Cancelling an await does not stop its thread or external side effects.
        # Let the worker finish before its caller releases the resources it owns.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        cancelled_outcome = task.result()
        if isinstance(cancelled_outcome, UserCancelledError):
            raise cancelled_outcome from None
        raise
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


class _ScheduledModel:
    def __init__(self, delegate: ModelAdapter, scheduler: CooperativeScheduler) -> None:
        self.delegate = delegate
        self.scheduler = scheduler

    async def aclose(self) -> None:
        await close_model(self.delegate)

    async def complete(
        self, context: ModelContext, tools: tuple[ToolDefinition, ...]
    ) -> ModelTurn:
        async with self.scheduler._requests:
            return await self.delegate.complete(context, tools)
