"""Durable child batches and structured delegation handoff."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field

from handshake.context import ActiveContextManagers
from handshake.context.errors import ContextPressureError
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter, close_model
from handshake.models.protocol import ModelMessage, ToolCall
from handshake.runtime.agents import AgentSpec, model_name_for_run
from handshake.runtime.errors import AgentLoopError, UserCancelledError
from handshake.runtime.loop import (
    EmitCallback,
    _checkpoint,
    _run_turns,
    _sync_context,
    context_manager_for_run,
    continue_run,
)
from handshake.runtime.persistence import RunStore
from handshake.runtime.progress import bind_reporter
from handshake.runtime.recovery import resume_run
from handshake.runtime.runs import TERMINAL_RUN_STATUSES, ChildRunResult, create_child_run
from handshake.runtime.scheduler import CooperativeScheduler
from handshake.runtime.state import Run, RunStatus, Session, ToolResult
from handshake.storage.errors import StorageError
from handshake.tools import DELEGATE_TOOL
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner


@dataclass(slots=True)
class DelegationRuntime:
    """Configuration and process-local services for a child batch."""

    agent_specs: Mapping[str, AgentSpec]
    model_for_spec: Callable[[AgentSpec], ModelAdapter]
    context_managers: ActiveContextManagers
    scheduler: CooperativeScheduler = field(default_factory=CooperativeScheduler)
    resolve_input: Callable[[Run], Awaitable[ModelMessage | None]] | None = None

    def prepare_batch(
        self, calls: tuple[ToolCall, ...]
    ) -> tuple[tuple[AgentSpec, str], ...] | ToolResult:
        prepared = tuple(_delegation_arguments(call, self) for call in calls)
        error = next((item for item in prepared if isinstance(item, ToolResult)), None)
        if error is not None:
            return error
        return tuple(item for item in prepared if not isinstance(item, ToolResult))

    async def run_batch(
        self,
        session: Session,
        parent: Run,
        calls: tuple[ToolCall, ...],
        prepared: tuple[tuple[AgentSpec, str], ...],
        registry: ToolRegistry,
        secret_command_runner: SecretCommandRunner | None,
        emit: EmitCallback,
        store: RunStore,
    ) -> tuple[Run, ...]:
        children = _ensure_children(session, parent, calls, prepared, store)
        return await self.scheduler.advance(
            children,
            lambda child: _run_child(
                session, child, registry, secret_command_runner, emit, store, self
            ),
            include_waiting=self.resolve_input is not None,
        )

    def append_results(
        self, pending_messages: list[ModelMessage], parent: Run, children: tuple[Run, ...]
    ) -> None:
        if not self.scheduler.ready_to_join(children):
            raise AgentLoopError("Cannot join a delegation batch with active children")
        pending_messages.append(
            ModelMessage(
                role="user",
                content=tuple(_child_result_block(child) for child in children),
                run_id=parent.run_id,
            )
        )


def _delegation_arguments(
    call: ToolCall, delegation: DelegationRuntime
) -> tuple[AgentSpec, str] | ToolResult:
    spec_id = call.arguments.get("agent_spec_id")
    goal = call.arguments.get("goal")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return ToolResult(ok=False, stderr="agent_spec_id must be a non-empty string")
    if not isinstance(goal, str) or not goal.strip():
        return ToolResult(ok=False, stderr="goal must be a non-empty string")
    spec = delegation.agent_specs.get(spec_id)
    if spec is None:
        return ToolResult(ok=False, stderr=f"Agent spec {spec_id} is unavailable")
    if DELEGATE_TOOL.name in spec.allowed_tools:
        return ToolResult(ok=False, stderr=f"Agent spec {spec_id} cannot be used as a child")
    return spec, goal.strip()


def _child_result_block(child: Run) -> dict[str, object]:
    result = ChildRunResult.from_run(child)
    if child.delegation_tool_call_id is None:
        raise AgentLoopError("Child has no delegation tool call")
    return {
        "type": "tool_result",
        "tool_use_id": child.delegation_tool_call_id,
        "content": json.dumps(asdict(result), ensure_ascii=False),
        "is_error": child.status is not RunStatus.COMPLETED,
    }


def _ensure_children(
    session: Session,
    parent: Run,
    calls: tuple[ToolCall, ...],
    prepared: tuple[tuple[AgentSpec, str], ...],
    store: RunStore,
) -> tuple[Run, ...]:
    if len(calls) != len(prepared) or not calls:
        raise AgentLoopError("Delegation batch must contain prepared tool calls")
    children: list[Run] = []
    for call, (spec, goal) in zip(calls, prepared, strict=True):
        runs = store.list_runs(session.session_id)
        existing = tuple(
            run for run in runs
            if run.parent_run_id == parent.run_id and run.delegation_tool_call_id == call.id
        )
        if len(existing) > 1:
            raise AgentLoopError("Delegation tool call has multiple child runs")
        if existing:
            child = existing[0]
            if child.agent_spec_id != spec.spec_id or child.goal != goal:
                raise AgentLoopError("Stored child does not match its delegation tool call")
        else:
            child = create_child_run(
                session, runs, parent, goal,
                agent_spec_id=spec.spec_id,
                model_name=spec.model_name,
                max_steps=spec.default_max_steps,
                tool_call_id=call.id,
            )
            child.transition(RunStatus.RUNNING)
            store.create_run(
                session,
                child,
                (
                    ModelMessage(
                        "user", ({"type": "text", "text": child.goal},), run_id=child.run_id
                    ),
                ),
            )
        children.append(child)
    return tuple(children)


async def _run_child(
    session: Session,
    child: Run,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: EmitCallback,
    store: RunStore,
    delegation: DelegationRuntime,
) -> Run:
    emit = bind_reporter(emit, child)
    spec = delegation.agent_specs.get(child.agent_spec_id)
    if spec is None:
        child.error = f"Agent spec {child.agent_spec_id} is unavailable"
        child.transition(RunStatus.FAILED)
        store.checkpoint_run(session, child)
        bind_reporter(emit, child)(f"Run failed: {child.error}")
        return child
    model: ModelAdapter | None = None
    try:
        model = delegation.scheduler.model(delegation.model_for_spec(spec))
        manager = context_manager_for_run(delegation.context_managers, child, spec, model)
        if child.status is RunStatus.RUNNING:
            child = await resume_run(
                session, child, model, registry,
                store=store, agent_spec=spec, emit=emit,
                secret_command_runner=secret_command_runner,
                context_manager=manager, scheduler=delegation.scheduler,
            )
        while (child.status is RunStatus.WAITING and child.open_dependencies()
               and delegation.resolve_input is not None):
            continuation = await delegation.resolve_input(child)
            if continuation is None:
                break
            child = await continue_run(
                session, child, model, registry, store=store, agent_spec=spec, emit=emit,
                secret_command_runner=secret_command_runner, context_manager=manager,
                scheduler=delegation.scheduler, pending_messages=(continuation,),
            )
        return child

    except (UserCancelledError, StorageError):
        raise
    except Exception as exc:
        persisted = store.load_run(child.run_id)
        if persisted.status in TERMINAL_RUN_STATUSES:
            bind_reporter(emit, persisted)(f"Run {persisted.status.value}: {persisted.error or ''}")
            return persisted
        persisted.error = (
            str(exc)
            if isinstance(exc, (HandshakeError, ContextPressureError))
            else f"Unexpected child failure ({type(exc).__name__})"
        )
        persisted.transition(RunStatus.FAILED)
        store.checkpoint_run(session, persisted)
        bind_reporter(emit, persisted)(f"Run failed: {persisted.error}")
        return persisted
    finally:
        if model is not None:
            await close_model(model)


def _pending_calls(parent: Run, store: RunStore) -> tuple[ToolCall, ...]:
    messages = tuple(
        item.message for item in store.read_messages(parent.context_scope_id)
        if item.message.run_id == parent.run_id
    )
    if not messages or messages[-1].role != "assistant":
        raise AgentLoopError("Delegating run has no pending assistant tool turn")
    calls: list[ToolCall] = []
    for block in messages[-1].content:
        if block.get("type") != "tool_use":
            continue
        call_id = block.get("id")
        arguments = block.get("input")
        if (
            block.get("name") != DELEGATE_TOOL.name
            or not isinstance(call_id, str)
            or not isinstance(arguments, dict)
        ):
            raise AgentLoopError("Stored delegation batch contains an invalid tool call")
        calls.append(ToolCall(call_id, DELEGATE_TOOL.name, arguments))
    if not calls:
        raise AgentLoopError("Delegating run has no pending delegation tool calls")
    return tuple(calls)


async def continue_delegation(
    session: Session,
    parent: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    store: RunStore,
    agent_spec: AgentSpec,
    delegation: DelegationRuntime,
    emit: EmitCallback | None = None,
    secret_command_runner: SecretCommandRunner | None = None,
) -> Run:
    """Recover a child batch, join it once, and resume the parent."""
    scheduler = delegation.scheduler
    with scheduler.run_scope(parent):
        if (
            parent.status not in {RunStatus.WAITING, RunStatus.RUNNING}
            or parent.open_dependencies()
        ):
            raise AgentLoopError("Run is not ready for a delegated child handoff")
        if parent.session_id != session.session_id:
            raise AgentLoopError("Delegating run does not belong to the session")
        parent.model_name = model_name_for_run(parent, agent_spec)
        output = emit or (lambda _message: None)
        model = delegation.scheduler.model(model)
        manager = context_manager_for_run(delegation.context_managers, parent, agent_spec, model)
        _sync_context(parent, store, manager)

        if parent.status is RunStatus.RUNNING:
            messages = tuple(
                item.message for item in store.read_messages(parent.context_scope_id)
                if item.message.run_id == parent.run_id
            )
            if len(messages) < 2 or messages[-1].role != "user" or messages[-2].role != "assistant":
                raise AgentLoopError("Run has no completed child handoff to resume")
            call_ids = tuple(
                block.get("id") for block in messages[-2].content
                if block.get("type") == "tool_use" and block.get("name") == DELEGATE_TOOL.name
            )
            result_ids = tuple(
                block.get("tool_use_id") for block in messages[-1].content
                if block.get("type") == "tool_result"
            )
            tool_uses = tuple(
                block for block in messages[-2].content if block.get("type") == "tool_use"
            )
            if not call_ids or len(call_ids) != len(tool_uses) or call_ids != result_ids:
                raise AgentLoopError("Run has no completed child handoff to resume")
            return await delegation.scheduler.advance_runnable(
                parent,
                lambda runnable: _run_turns(
                    session, runnable, model, registry, secret_command_runner,
                    output, store, manager, agent_spec, delegation, delegation.scheduler,
                ),
            )

        calls = _pending_calls(parent, store)
        prepared = delegation.prepare_batch(calls)
        if isinstance(prepared, ToolResult):
            raise AgentLoopError("Stored delegation batch is no longer available")
        children = await delegation.run_batch(
            session, parent, calls, prepared, registry,
            secret_command_runner, output, store,
        )
        if not delegation.scheduler.ready_to_join(children):
            return parent
        pending: list[ModelMessage] = []
        delegation.append_results(pending, parent, children)
        parent.transition(RunStatus.RUNNING)
        _checkpoint(session, parent, store, manager, pending)
        output(f"Joined {len(children)} child runs; resuming parent")
        return await delegation.scheduler.advance_runnable(
            parent,
            lambda runnable: _run_turns(
                session, runnable, model, registry, secret_command_runner,
                output, store, manager, agent_spec, delegation, delegation.scheduler,
            ),
        )
