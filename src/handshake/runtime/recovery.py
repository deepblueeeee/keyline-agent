"""Resume a run from its last durable model and tool boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from handshake.context import AgentContextManager, LLMHistoryCondenser
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter
from handshake.models.protocol import ModelMessage, ToolCall
from handshake.runtime.agents import AgentSpec, model_name_for_run
from handshake.runtime.errors import AgentLoopError, UserCancelledError
from handshake.runtime.loop import (
    EmitCallback,
    _append_tool_results,
    _checkpoint,
    _execute_tool_calls,
    _run_turns,
    _sync_context,
    _terminate_run,
)
from handshake.runtime.persistence import RunStore
from handshake.runtime.secret_dependencies import reissue_lost_secret_dependencies
from handshake.runtime.state import Run, RunStatus, Session
from handshake.tools import DELEGATE_TOOL, SECRET_WAIT_TOOL, TEXT_WAIT_TOOL
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner

if TYPE_CHECKING:
    from handshake.runtime.delegation import DelegationRuntime

def resume_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    store: RunStore,
    agent_spec: AgentSpec,
    emit: EmitCallback | None = None,
    secret_command_runner: SecretCommandRunner | None = None,
    context_manager: AgentContextManager | None = None,
    delegation: DelegationRuntime | None = None,
) -> Run:
    """Resume a running run from its last durable protocol and tool boundary."""
    if run.status is not RunStatus.RUNNING or run.session_id != session.session_id:
        raise AgentLoopError("Only a running run in this session can be resumed")
    run.model_name = model_name_for_run(run, agent_spec)
    output = emit or (lambda _message: None)
    manager = context_manager or AgentContextManager(
        policy=agent_spec.default_budget,
        condenser=LLMHistoryCondenser(model),
        scope_id=run.context_scope_id,
    )
    manager.bind_scope(run.context_scope_id)
    _sync_context(run, store, manager)
    count = store.message_count(run.context_scope_id)
    if count == 0:
        raise AgentLoopError("Running run has no durable model history")
    latest = tuple(store.read_messages(run.context_scope_id, count - 1, count))
    if len(latest) != 1 or latest[0].message.run_id != run.run_id:
        raise AgentLoopError("Running run does not own the context tail")
    message = latest[0].message
    if message.role == "assistant":
        calls: list[ToolCall] = []
        for block in message.content:
            if block.get("type") != "tool_use":
                continue
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                raise AgentLoopError("Stored tool call is invalid")
            calls.append(ToolCall(call_id, name, arguments))
        if not calls or any(
            call.name in {TEXT_WAIT_TOOL.name, SECRET_WAIT_TOOL.name, DELEGATE_TOOL.name}
            for call in calls
        ):
            raise AgentLoopError("Running run has no resumable workspace tool turn")
        pending: list[ModelMessage] = []
        try:
            results = _execute_tool_calls(
                session, run, tuple(calls), registry, secret_command_runner,
                output, store, pending,
            )
            _append_tool_results(pending, run, tuple(calls), results)
        except UserCancelledError as exc:
            _terminate_run(run, RunStatus.CANCELLED, str(exc))
            _checkpoint(session, run, store, manager, pending)
            raise
        except HandshakeError as exc:
            _terminate_run(run, RunStatus.FAILED, str(exc))
            _checkpoint(session, run, store, manager, pending)
            raise
        _checkpoint(session, run, store, manager, pending)
        if secret_command_runner is not None and reissue_lost_secret_dependencies(
            run, secret_command_runner
        ):
            store.checkpoint_run(session, run)
            return run
    elif message.role != "user":
        raise AgentLoopError("Running run has invalid model history")
    return _run_turns(
        session, run, model, registry, secret_command_runner, output,
        store, manager, agent_spec, delegation,
    )
