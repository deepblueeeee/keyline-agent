"""Session-first ReAct loop with durable dependencies and resumption."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from handshake.context import (
    AgentContextManager,
    ContextPressureError,
    LLMHistoryCondenser,
)
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter, ToolDefinition
from handshake.models.errors import ModelProtocolError
from handshake.models.protocol import ModelContext, ModelMessage, ToolCall
from handshake.runtime.errors import (
    AgentLoopError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.persistence import RunStore
from handshake.runtime.secret_dependencies import reissue_lost_secret_dependencies
from handshake.runtime.state import (
    Dependency,
    DependencyStatus,
    Run,
    RunStatus,
    Session,
    ToolResult,
    ToolRun,
    ToolRunStatus,
    WorkItem,
    WorkItemStatus,
)
from handshake.tools import (
    PLAN_TOOL,
    SECRET_COMMAND_TOOL,
    SECRET_WAIT_TOOL,
    TEXT_WAIT_TOOL,
    handle_plan,
    prepare_secret_dependency,
    prepare_text_dependency,
    runtime_tool_definitions,
)
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner

EmitCallback = Callable[[str], None]


def _result_content(result: ToolResult) -> str:
    return json.dumps(
        {
            "ok": result.ok,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        },
        ensure_ascii=False,
    )


def _running_work_item(run: Run) -> WorkItem:
    running = tuple(
        item for item in run.work_items if item.status is WorkItemStatus.RUNNING
    )
    if len(running) != 1:
        raise AgentLoopError(
            f"Run {run.run_id} must have exactly one running work item, found {len(running)}"
        )
    return running[0]


def _is_runnable(run: Run, work_item: WorkItem) -> bool:
    if work_item.status is not WorkItemStatus.READY:
        return False
    return not any(
        dependency.status is DependencyStatus.OPEN
        for dependency in run.dependencies_for(work_item)
    )


def _runnable_work_items(run: Run) -> tuple[WorkItem, ...]:
    return tuple(item for item in run.work_items if _is_runnable(run, item))


def _resume_one_work_item(run: Run) -> WorkItem:
    runnable = _runnable_work_items(run)
    if len(runnable) != 1:
        raise AgentLoopError(
            f"Run {run.run_id} must have exactly one runnable work item, found {len(runnable)}"
        )
    runnable[0].transition(WorkItemStatus.RUNNING)
    return runnable[0]


def _validate_turn_tool_calls(
    content: tuple[dict[str, Any], ...], calls: tuple[ToolCall, ...]
) -> None:
    content_ids = [
        block.get("id") for block in content if block.get("type") == "tool_use"
    ]
    call_ids = [call.id for call in calls]
    if content_ids != call_ids:
        raise ModelProtocolError("Model tool calls do not match assistant content blocks")
    if len(call_ids) != len(set(call_ids)):
        raise ModelProtocolError("Model returned duplicate tool call IDs")


def _execute_tool_call(
    session: Session,
    run: Run,
    call: ToolCall,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: EmitCallback,
) -> ToolResult:
    if call.name == PLAN_TOOL.name:
        return handle_plan(run, call)

    work_item = _running_work_item(run)
    tool_run = ToolRun(
        work_id=work_item.work_id,
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=dict(call.arguments),
    )
    run.add_tool_run(work_item, tool_run)
    tool_run.transition(ToolRunStatus.RUNNING)
    emit(f"Running tool: {call.name}")
    try:
        if call.name == SECRET_COMMAND_TOOL.name:
            if secret_command_runner is None:
                result = ToolResult(
                    ok=False,
                    stderr=(
                        "Secret command capability is unavailable; "
                        "no shell fallback was used"
                    ),
                )
            else:
                result = secret_command_runner.execute(
                    run, session.workspace, call.arguments
                )
        else:
            result = registry.execute(call.name, session.workspace, call.arguments)
    except Exception as exc:
        tool_run.result = ToolResult(
            ok=False,
            stderr=f"Tool execution interrupted ({type(exc).__name__})",
        )
        tool_run.transition(ToolRunStatus.FAILED)
        raise
    tool_run.result = result
    tool_run.transition(ToolRunStatus.SUCCEEDED if result.ok else ToolRunStatus.FAILED)
    return result


def _append_tool_results(
    pending_messages: list[ModelMessage],
    run: Run,
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> None:
    pending_messages.append(
        ModelMessage(
            role="user",
            content=tuple(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": _result_content(result),
                    "is_error": not result.ok,
                }
                for call, result in zip(calls, results, strict=True)
            ),
            run_id=run.run_id,
        )
    )


def _execute_tool_calls(
    session: Session,
    run: Run,
    calls: tuple[ToolCall, ...],
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: EmitCallback,
    pending_messages: list[ModelMessage],
) -> tuple[ToolResult, ...]:
    results: list[ToolResult] = []
    for index, call in enumerate(calls):
        try:
            results.append(
                _execute_tool_call(
                    session, run, call, registry, secret_command_runner, emit
                )
            )
        except Exception as exc:
            results.append(
                ToolResult(
                    ok=False,
                    stderr=f"Tool execution interrupted ({type(exc).__name__})",
                )
            )
            results.extend(
                ToolResult(
                    ok=False,
                    stderr="Tool was not executed because an earlier call was interrupted",
                )
                for _call in calls[index + 1 :]
            )
            _append_tool_results(pending_messages, run, calls, tuple(results))
            raise
    return tuple(results)


def _sync_context(
    run: Run,
    store: RunStore,
    context_manager: AgentContextManager,
) -> None:
    scope = store.load_context_scope(run.context_scope_id)
    if not context_manager.is_initialized(scope):
        end = store.message_count(run.context_scope_id)
        context_manager.restore(
            scope,
            store.load_context_checkpoint(run.context_scope_id),
            end,
            lambda start, stop: store.read_messages(
                run.context_scope_id, start, stop
            ),
        )
        return
    start = context_manager.message_count(scope)
    end = store.message_count(run.context_scope_id)
    if start > end:
        raise AgentLoopError("Context view is ahead of the durable transcript")
    context_manager.ingest_messages(scope, store.read_messages(run.context_scope_id, start, end))


def _prepare_model_context(
    session: Session,
    run: Run,
    tools: tuple[ToolDefinition, ...],
    store: RunStore,
    context_manager: AgentContextManager,
) -> ModelContext:
    """Prepare a view and durably save only newly reduced projection state."""
    try:
        return context_manager.prepare(
            session,
            run,
            tools,
            raw_message_reader=lambda scope_id, sequence: _read_raw_message(
                store, scope_id, sequence
            ),
        )
    finally:
        scope = store.load_context_scope(run.context_scope_id)
        checkpoint = context_manager.pending_checkpoint(scope)
        if checkpoint is not None:
            saved = store.save_context_checkpoint(run.context_scope_id, checkpoint)
            context_manager.checkpoint_saved(scope, saved)


def _read_raw_message(store: RunStore, scope_id: str, sequence: int) -> ModelMessage:
    messages = tuple(store.read_messages(scope_id, sequence, sequence + 1))
    if len(messages) != 1 or messages[0].sequence != sequence:
        raise AgentLoopError(f"Missing durable message sequence {sequence}")
    return messages[0].message


def _checkpoint(
    session: Session,
    run: Run,
    store: RunStore,
    context_manager: AgentContextManager,
    pending_messages: list[ModelMessage],
    *,
    create: bool = False,
) -> None:
    messages = tuple(pending_messages)
    if create:
        appended = store.create_run(session, run, messages)
        pending_messages.clear()
        _sync_context(run, store, context_manager)
        return
    else:
        appended = store.checkpoint_run(session, run, messages)
    pending_messages.clear()
    scope = store.load_context_scope(run.context_scope_id)
    context_manager.ingest_messages(scope, appended)
    checkpoint = context_manager.pending_checkpoint(scope)
    if checkpoint is not None:
        saved = store.save_context_checkpoint(run.context_scope_id, checkpoint)
        context_manager.checkpoint_saved(scope, saved)


def _create_text_dependency(
    run: Run,
    call: ToolCall,
    pending_messages: list[ModelMessage],
) -> Dependency | None:
    work_item = _running_work_item(run)
    prepared = prepare_text_dependency(call, work_item)
    if isinstance(prepared, ToolResult):
        _append_tool_results(pending_messages, run, (call,), (prepared,))
        return None

    dependency = prepared
    run.add_dependency(work_item, dependency)
    work_item.transition(WorkItemStatus.READY)
    result = ToolResult(
        ok=True,
        stdout=(
            f"Registered text dependency {dependency.dependency_id} "
            f"for work item {work_item.work_id}"
        ),
    )
    _append_tool_results(pending_messages, run, (call,), (result,))

    if not _runnable_work_items(run) and run.open_dependencies():
        run.transition(RunStatus.WAITING)
    return dependency


def _create_secret_dependency(
    run: Run,
    call: ToolCall,
    pending_messages: list[ModelMessage],
) -> Dependency | None:
    work_item = _running_work_item(run)
    prepared = prepare_secret_dependency(call, work_item)
    if isinstance(prepared, ToolResult):
        _append_tool_results(pending_messages, run, (call,), (prepared,))
        return None

    dependency, handle = prepared
    run.add_dependency(work_item, dependency)
    run.add_secret_handle(dependency, handle)
    work_item.transition(WorkItemStatus.READY)
    result = ToolResult(
        ok=True,
        stdout=(
            f"Registered Secret dependency {dependency.dependency_id}\n"
            f"Secret handle: {handle.secret_handle}\n"
            f"Consumer: {handle.consumer}\n"
            f"Environment variable: {handle.environment_variable}"
        ),
    )
    _append_tool_results(pending_messages, run, (call,), (result,))

    if not _runnable_work_items(run) and run.open_dependencies():
        run.transition(RunStatus.WAITING)
    return dependency


def _run_turns(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: EmitCallback,
    store: RunStore,
    context_manager: AgentContextManager,
) -> Run:
    definitions = (*runtime_tool_definitions(), *registry.definitions())
    while True:
        pending_messages: list[ModelMessage] = []
        messages_after_checkpoint: list[str] = []
        return_after_checkpoint = False
        try:
            if run.model_turns >= run.max_steps:
                raise StepLimitExceededError(
                    f"Agent exceeded the maximum of {run.max_steps} model turns"
                )
            emit(f"Model turn {run.model_turns + 1}/{run.max_steps}")
            context = _prepare_model_context(
                session, run, definitions, store, context_manager
            )
            turn = model.complete(context, definitions)
            run.model_turns += 1
            if not turn.content:
                raise ModelProtocolError("Model returned an empty turn")
            _validate_turn_tool_calls(turn.content, turn.tool_calls)
            context_manager.record_usage(
                store.load_context_scope(run.context_scope_id), context, definitions, turn
            )
            pending_messages.append(
                ModelMessage(role="assistant", content=turn.content, run_id=run.run_id)
            )

            if turn.tool_calls:
                if turn.text:
                    emit(f"Model: {turn.text}")
                dependency_calls = tuple(
                    call
                    for call in turn.tool_calls
                    if call.name in {TEXT_WAIT_TOOL.name, SECRET_WAIT_TOOL.name}
                )
                if dependency_calls and (
                    len(dependency_calls) != 1 or len(turn.tool_calls) != 1
                ):
                    dependency_error = ToolResult(
                        ok=False,
                        stderr=(
                            "wait_for_text or wait_for_secret must be the only "
                            "tool call in its turn"
                        ),
                    )
                    _append_tool_results(
                        pending_messages,
                        run,
                        turn.tool_calls,
                        tuple(dependency_error for _call in turn.tool_calls),
                    )
                    messages_after_checkpoint.append(
                        "Rejected tool turn: dependency request must be called alone"
                    )
                elif dependency_calls:
                    dependency_call = dependency_calls[0]
                    if dependency_call.name == TEXT_WAIT_TOOL.name:
                        dependency = _create_text_dependency(
                            run, dependency_call, pending_messages
                        )
                        wait_message = "Waiting for text"
                    else:
                        dependency = _create_secret_dependency(
                            run, dependency_call, pending_messages
                        )
                        wait_message = "Waiting for Secret input"
                    if dependency is None:
                        messages_after_checkpoint.append(
                            f"Tool {dependency_call.name}: failed"
                        )
                    elif run.status is RunStatus.WAITING:
                        messages_after_checkpoint.append(
                            f"{wait_message}: {dependency.prompt}"
                        )
                        return_after_checkpoint = True
                else:
                    results = _execute_tool_calls(
                        session,
                        run,
                        turn.tool_calls,
                        registry,
                        secret_command_runner,
                        emit,
                        pending_messages,
                    )
                    _append_tool_results(
                        pending_messages, run, turn.tool_calls, results
                    )
                    for call, result in zip(turn.tool_calls, results, strict=True):
                        if call.name == PLAN_TOOL.name:
                            if result.ok:
                                messages_after_checkpoint.append(
                                    "Plan updated: " + " | ".join(run.plan)
                                )
                            else:
                                messages_after_checkpoint.append(
                                    f"Plan update failed: {result.stderr}"
                                )
                        else:
                            outcome = "succeeded" if result.ok else "failed"
                            messages_after_checkpoint.append(
                                f"Tool {call.name}: {outcome}"
                            )
            elif turn.stop_reason == "end_turn" and turn.text:
                run.final_summary = turn.text
                _running_work_item(run).transition(WorkItemStatus.COMPLETED)
                run.transition(RunStatus.COMPLETED)
                messages_after_checkpoint.append("Run completed")
                return_after_checkpoint = True
            else:
                raise ModelProtocolError(
                    "Model stopped without tool calls or a non-empty final end_turn response"
                )
        except UserCancelledError as exc:
            _terminate_run(run, RunStatus.CANCELLED, str(exc))
            _checkpoint(session, run, store, context_manager, pending_messages)
            raise
        except (HandshakeError, ContextPressureError) as exc:
            _terminate_run(run, RunStatus.FAILED, str(exc))
            _checkpoint(session, run, store, context_manager, pending_messages)
            raise
        except Exception as exc:
            unexpected_error = f"Unexpected runtime failure ({type(exc).__name__})"
            _terminate_run(run, RunStatus.FAILED, unexpected_error)
            _checkpoint(session, run, store, context_manager, pending_messages)
            raise AgentLoopError(unexpected_error) from None

        _checkpoint(session, run, store, context_manager, pending_messages)
        for message in messages_after_checkpoint:
            emit(message)
        if return_after_checkpoint:
            return run


def _terminate_run(run: Run, status: RunStatus, error: str) -> None:
    run.error = error
    _fail_running_work_items(run)
    if run.status in {RunStatus.RUNNING, RunStatus.WAITING}:
        run.transition(status)


def _fail_running_work_items(run: Run) -> None:
    for work_item in run.work_items:
        if work_item.status is WorkItemStatus.RUNNING:
            work_item.transition(WorkItemStatus.FAILED)


def start_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    emit: EmitCallback | None = None,
    store: RunStore,
    secret_command_runner: SecretCommandRunner | None = None,
    context_manager: AgentContextManager | None = None,
) -> Run:
    """Start one user-request run and execute until terminal state or a wait."""
    if run.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if run.status is not RunStatus.CREATED:
        raise AgentLoopError("Only a newly created run can be started")
    if run.session_id != session.session_id:
        raise AgentLoopError(f"Run {run.run_id} does not belong to the session")
    if store.has_run_messages(run.run_id):
        raise AgentLoopError(f"Run {run.run_id} already has model messages")

    output = emit or (lambda _message: None)
    current_context_manager = context_manager or AgentContextManager(
        condenser=LLMHistoryCondenser(model)
    )
    session.workspace = session.workspace.resolve()
    run.transition(RunStatus.RUNNING)
    _resume_one_work_item(run)
    pending_messages = [
        ModelMessage(
            role="user",
            content=({"type": "text", "text": run.goal},),
            run_id=run.run_id,
        )
    ]
    _checkpoint(
        session,
        run,
        store,
        current_context_manager,
        pending_messages,
        create=True,
    )
    output(f"Run started: {run.run_id}")
    return _run_turns(
        session,
        run,
        model,
        registry,
        secret_command_runner,
        output,
        store,
        current_context_manager,
    )


def prepare_run_for_resume(
    session: Session,
    run: Run,
    secret_command_runner: SecretCommandRunner,
    *,
    store: RunStore,
) -> bool:
    """Recover process-local Secret state and persist it before CLI presentation."""
    if run.session_id != session.session_id:
        raise AgentLoopError(f"Run {run.run_id} does not belong to the session")
    changed = reissue_lost_secret_dependencies(run, secret_command_runner)
    if changed:
        store.checkpoint_run(session, run)
    return changed


def continue_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    emit: EmitCallback | None = None,
    store: RunStore,
    pending_messages: tuple[ModelMessage, ...],
    secret_command_runner: SecretCommandRunner | None = None,
    context_manager: AgentContextManager | None = None,
) -> Run:
    """Persist a resolved dependency, then continue its durable run."""
    if run.status is not RunStatus.RUNNING:
        raise AgentLoopError("Only a running run can continue")
    if run.session_id != session.session_id:
        raise AgentLoopError(f"Run {run.run_id} does not belong to the session")
    if not store.has_run_messages(run.run_id):
        raise AgentLoopError("A resumed run must have model protocol history")
    output = emit or (lambda _message: None)
    current_context_manager = context_manager or AgentContextManager(
        condenser=LLMHistoryCondenser(model)
    )
    _sync_context(run, store, current_context_manager)
    pending = list(pending_messages)
    _checkpoint(session, run, store, current_context_manager, pending)
    output("Input received; resuming run")
    _resume_one_work_item(run)
    return _run_turns(
        session,
        run,
        model,
        registry,
        secret_command_runner,
        output,
        store,
        current_context_manager,
    )
