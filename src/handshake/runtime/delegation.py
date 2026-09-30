"""Synchronous child-run delegation and durable result handoff."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass

from handshake.context import ActiveContextManagers
from handshake.context.errors import ContextPressureError
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter
from handshake.models.protocol import ModelMessage, ToolCall
from handshake.runtime.agents import AgentSpec, model_name_for_run
from handshake.runtime.errors import AgentLoopError, UserCancelledError
from handshake.runtime.loop import (
    EmitCallback,
    _checkpoint,
    _run_turns,
    _sync_context,
    context_manager_for_run,
    start_run,
)
from handshake.runtime.persistence import RunStore
from handshake.runtime.runs import ChildRunResult, create_child_run
from handshake.runtime.state import Run, RunStatus, Session, ToolResult
from handshake.storage.errors import StorageError
from handshake.tools import DELEGATE_TOOL
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner


@dataclass(slots=True)
class DelegationRuntime:
    """Configuration and process-local services for synchronous child runs."""

    agent_specs: Mapping[str, AgentSpec]
    model_for_spec: Callable[[AgentSpec], ModelAdapter]
    context_managers: ActiveContextManagers

    def prepare(self, call: ToolCall) -> tuple[AgentSpec, str] | ToolResult:
        return _delegation_arguments(call, self)

    def run_child(
        self,
        session: Session,
        parent: Run,
        call: ToolCall,
        spec: AgentSpec,
        goal: str,
        registry: ToolRegistry,
        secret_command_runner: SecretCommandRunner | None,
        emit: EmitCallback,
        store: RunStore,
    ) -> Run:
        return _run_child(
            session, parent, call, spec, goal, registry,
            secret_command_runner, emit, store, self,
        )

    def append_result(
        self, pending_messages: list[ModelMessage], parent: Run, child: Run
    ) -> None:
        _append_child_result(pending_messages, parent, child)


def _delegation_arguments(
    call: ToolCall, delegation: DelegationRuntime | None
) -> tuple[AgentSpec, str] | ToolResult:
    if delegation is None:
        return ToolResult(ok=False, stderr="Delegation is unavailable")
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


def _append_child_result(pending_messages: list[ModelMessage], parent: Run, child: Run) -> None:
    result = ChildRunResult.from_run(child)
    if child.delegation_tool_call_id is None:
        raise AgentLoopError("Child has no delegation tool call")
    pending_messages.append(
        ModelMessage(
            role="user",
            content=(
                {
                    "type": "tool_result",
                    "tool_use_id": child.delegation_tool_call_id,
                    "content": json.dumps(asdict(result), ensure_ascii=False),
                    "is_error": child.status is not RunStatus.COMPLETED,
                },
            ),
            run_id=parent.run_id,
        )
    )


def _run_child(
    session: Session,
    parent: Run,
    call: ToolCall,
    spec: AgentSpec,
    goal: str,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: EmitCallback,
    store: RunStore,
    delegation: DelegationRuntime,
) -> Run:
    child = create_child_run(
        session,
        store.list_runs(session.session_id),
        parent,
        goal,
        agent_spec_id=spec.spec_id,
        model_name=spec.model_name,
        max_steps=spec.default_max_steps,
        tool_call_id=call.id,
    )
    try:
        model = delegation.model_for_spec(spec)
        manager = context_manager_for_run(delegation.context_managers, child, spec, model)
    except (UserCancelledError, StorageError):
        raise
    except Exception as exc:
        error = (
            str(exc)
            if isinstance(exc, (HandshakeError, ContextPressureError))
            else f"Unexpected child setup failure ({type(exc).__name__})"
        )
        child.transition(RunStatus.RUNNING)
        store.create_run(
            session,
            child,
            (ModelMessage("user", ({"type": "text", "text": child.goal},), run_id=child.run_id),),
        )
        child.error = error
        child.transition(RunStatus.FAILED)
        store.checkpoint_run(session, child)
        return child
    try:
        return start_run(
            session,
            child,
            model,
            registry,
            emit=emit,
            store=store,
            secret_command_runner=secret_command_runner,
            context_manager=manager,
            agent_spec=spec,
            delegation=delegation,
        )
    except UserCancelledError:
        raise
    except (HandshakeError, ContextPressureError):
        persisted = store.load_run(child.run_id)
        if persisted.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return persisted
        raise

def continue_delegation(
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
    """Finish a child handoff, or create one after a post-checkpoint interruption."""
    if parent.status not in {RunStatus.WAITING, RunStatus.RUNNING} or parent.open_dependencies():
        raise AgentLoopError("Run is not ready for a delegated child handoff")
    if parent.session_id != session.session_id:
        raise AgentLoopError("Delegating run does not belong to the session")
    parent.model_name = model_name_for_run(parent, agent_spec)
    output = emit or (lambda _message: None)
    manager = context_manager_for_run(delegation.context_managers, parent, agent_spec, model)
    _sync_context(parent, store, manager)
    parent_messages = tuple(
        item.message
        for item in store.read_messages(parent.context_scope_id)
        if item.message.run_id == parent.run_id
    )
    if not parent_messages:
        raise AgentLoopError("Delegating run has no protocol history")
    if parent.status is RunStatus.RUNNING:
        last = parent_messages[-1]
        if last.role != "user" or len(last.content) != 1:
            raise AgentLoopError("Run has no completed child handoff to resume")
        block = last.content[0]
        if block.get("type") != "tool_result" or not isinstance(
            block.get("tool_use_id"), str
        ):
            raise AgentLoopError("Run has no completed child handoff to resume")
        call_id = block["tool_use_id"]
        children = tuple(
            child for child in store.list_runs(session.session_id)
            if child.parent_run_id == parent.run_id
            and child.delegation_tool_call_id == call_id
        )
        if len(children) != 1 or children[0].status not in {
            RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED
        }:
            raise AgentLoopError("Run has no completed child handoff to resume")
        return _run_turns(
            session, parent, model, registry, secret_command_runner, output,
            store, manager, agent_spec, delegation,
        )
    if parent_messages[-1].role != "assistant":
        raise AgentLoopError("Delegating run has no pending assistant tool call")
    blocks = tuple(
        block for block in parent_messages[-1].content if block.get("type") == "tool_use"
    )
    if len(blocks) != 1 or blocks[0].get("name") != DELEGATE_TOOL.name:
        raise AgentLoopError("Delegating run has no pending delegation tool call")
    block = blocks[0]
    tool_call_id = block.get("id")
    arguments = block.get("input")
    if not isinstance(tool_call_id, str) or not isinstance(arguments, dict):
        raise AgentLoopError("Stored delegation tool call is invalid")
    call = ToolCall(tool_call_id, DELEGATE_TOOL.name, arguments)
    children = tuple(
        run
        for run in store.list_runs(session.session_id)
        if run.parent_run_id == parent.run_id and run.delegation_tool_call_id == call.id
    )
    if len(children) > 1:
        raise AgentLoopError("Delegation tool call has multiple child runs")
    if children:
        child = children[0]
    else:
        prepared = _delegation_arguments(call, delegation)
        if isinstance(prepared, ToolResult):
            raise AgentLoopError(prepared.stderr)
        spec, goal = prepared
        child = _run_child(
            session,
            parent,
            call,
            spec,
            goal,
            registry,
            secret_command_runner,
            output,
            store,
            delegation,
        )
    if child.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
        return parent
    pending: list[ModelMessage] = []
    _append_child_result(pending, parent, child)
    parent.transition(RunStatus.RUNNING)
    _checkpoint(session, parent, store, manager, pending)
    output(f"Child run {child.run_id}: {child.status.value}; resuming parent")
    return _run_turns(
        session,
        parent,
        model,
        registry,
        secret_command_runner,
        output,
        store,
        manager,
        agent_spec,
        delegation,
    )
