"""Session-first ReAct loop with durable text waits and resumption."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from handshake.models.base import ModelAdapter
from handshake.models.protocol import ModelContext, ModelMessage, ToolCall
from handshake.runtime.errors import (
    AgentLoopError,
    HandshakeError,
    ModelProtocolError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.state import (
    Run,
    RunStatus,
    Session,
    ToolResult,
    ToolRun,
    ToolRunStatus,
)
from handshake.storage.base import RunStore
from handshake.tools import (
    PLAN_TOOL,
    TEXT_WAIT_TOOL,
    handle_plan,
    prepare_text_wait,
    runtime_tool_definitions,
)
from handshake.tools.registry import ToolRegistry

EmitCallback = Callable[[str], None]


def _system_prompt(session: Session, run: Run) -> str:
    plan = "\n".join(f"- {step}" for step in run.plan) or "(not set yet)"
    return f"""You are Handshake, a local coding agent operating on one workspace.

Current goal:
{run.goal}

Workspace:
{session.workspace}

Current plan:
{plan}

Rules:
- Investigate the repository before modifying files.
- Keep a short current plan using update_plan.
- All file paths must be relative to the workspace.
- Make targeted changes and run appropriate validation afterwards.
- Shell tools require user approval. If denied, adapt instead of pretending they ran.
- Ask for a non-sensitive user fact with wait_for_text only when it cannot be inferred from the
  repository. Call wait_for_text by itself in a turn.
- Never request passwords, API keys, tokens, or other secrets with wait_for_text.
- Use tool results as evidence and correct failures when possible.
- Do not claim a test passed unless its tool result says it passed.
- When the current request is complete, respond without tool calls with a concise summary of
  changes, validation performed, and any unresolved issue.
"""


def _context(session: Session, run: Run) -> ModelContext:
    return ModelContext(
        system_prompt=_system_prompt(session, run),
        goal=run.goal,
        workspace=str(session.workspace),
        plan=tuple(run.plan),
        messages=tuple(session.messages),
    )


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
    emit: EmitCallback,
) -> ToolResult:
    if call.name == PLAN_TOOL.name:
        result = handle_plan(run, call)
        if result.ok:
            emit("Plan updated: " + " | ".join(run.plan))
        else:
            emit(f"Plan update failed: {result.stderr}")
        return result

    tool_run = ToolRun(
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=dict(call.arguments),
    )
    run.tool_runs.append(tool_run)
    tool_run.transition(ToolRunStatus.RUNNING)
    emit(f"Running tool: {call.name}")
    try:
        result = registry.execute(call.name, session.workspace, call.arguments)
    except Exception as exc:
        tool_run.result = ToolResult(
            ok=False,
            stderr=f"Tool execution interrupted ({type(exc).__name__})",
        )
        tool_run.transition(ToolRunStatus.FAILED)
        emit(f"Tool {call.name}: interrupted")
        raise
    tool_run.result = result
    tool_run.transition(ToolRunStatus.SUCCEEDED if result.ok else ToolRunStatus.FAILED)
    emit(f"Tool {call.name}: {'succeeded' if result.ok else 'failed'}")
    return result


def _append_tool_results(
    session: Session,
    run: Run,
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> None:
    session.messages.append(
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
    emit: EmitCallback,
    store: RunStore | None,
) -> tuple[ToolResult, ...]:
    results: list[ToolResult] = []
    for index, call in enumerate(calls):
        try:
            results.append(_execute_tool_call(session, run, call, registry, emit))
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
            _append_tool_results(session, run, calls, tuple(results))
            _safe_checkpoint(session, run, store)
            raise
    return tuple(results)


def _checkpoint(session: Session, run: Run, store: RunStore | None) -> None:
    if store is not None:
        store.checkpoint_run(session, run)


def _create_text_wait(
    session: Session,
    run: Run,
    call: ToolCall,
    store: RunStore | None,
    emit: EmitCallback,
) -> bool:
    prepared = prepare_text_wait(call)
    if isinstance(prepared, ToolResult):
        _append_tool_results(session, run, (call,), (prepared,))
        _checkpoint(session, run, store)
        return False
    if store is None:
        _append_tool_results(
            session,
            run,
            (call,),
            (
                ToolResult(
                    ok=False,
                    stderr="wait_for_text requires durable session storage",
                ),
            ),
        )
        return False

    work_item, wait = prepared
    run.work_items.append(work_item)
    run.waits.append(wait)
    run.transition(RunStatus.WAITING)
    store.checkpoint_run(session, run)
    emit(f"Waiting for text: {wait.prompt}")
    return True


def _run_turns(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    emit: EmitCallback,
    store: RunStore | None,
) -> Run:
    definitions = (*runtime_tool_definitions(), *registry.definitions())
    while run.model_turns < run.max_steps:
        emit(f"Model turn {run.model_turns + 1}/{run.max_steps}")
        turn = model.complete(_context(session, run), definitions)
        run.model_turns += 1
        if not turn.content:
            raise ModelProtocolError("Model returned an empty turn")
        _validate_turn_tool_calls(turn.content, turn.tool_calls)
        session.messages.append(
            ModelMessage(role="assistant", content=turn.content, run_id=run.run_id)
        )

        if turn.tool_calls:
            if turn.text:
                emit(f"Model: {turn.text}")
            wait_calls = tuple(
                call for call in turn.tool_calls if call.name == TEXT_WAIT_TOOL.name
            )
            if wait_calls and (len(wait_calls) != 1 or len(turn.tool_calls) != 1):
                error = ToolResult(
                    ok=False,
                    stderr="wait_for_text must be the only tool call in its turn",
                )
                _append_tool_results(
                    session,
                    run,
                    turn.tool_calls,
                    tuple(error for _call in turn.tool_calls),
                )
                _checkpoint(session, run, store)
                emit("Rejected tool turn: wait_for_text must be called alone")
                continue
            if wait_calls:
                if _create_text_wait(session, run, wait_calls[0], store, emit):
                    return run
                continue

            results = _execute_tool_calls(
                session, run, turn.tool_calls, registry, emit, store
            )
            _append_tool_results(session, run, turn.tool_calls, results)
            _checkpoint(session, run, store)
            continue

        if turn.stop_reason == "end_turn" and turn.text:
            run.final_summary = turn.text
            run.transition(RunStatus.COMPLETED)
            _checkpoint(session, run, store)
            emit("Run completed")
            return run
        raise ModelProtocolError(
            "Model stopped without tool calls or a non-empty final end_turn response"
        )

    raise StepLimitExceededError(
        f"Agent exceeded the maximum of {run.max_steps} model turns"
    )


def _safe_checkpoint(
    session: Session, run: Run, store: RunStore | None
) -> None:
    if store is None:
        return
    with suppress(HandshakeError):
        store.checkpoint_run(session, run)


def _run_with_failure_handling(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    emit: EmitCallback,
    store: RunStore | None,
) -> Run:
    try:
        return _run_turns(session, run, model, registry, emit, store)
    except UserCancelledError as exc:
        run.error = str(exc)
        if run.status in {RunStatus.RUNNING, RunStatus.WAITING}:
            run.transition(RunStatus.CANCELLED)
        _safe_checkpoint(session, run, store)
        raise
    except HandshakeError as exc:
        run.error = str(exc)
        if run.status in {RunStatus.RUNNING, RunStatus.WAITING}:
            run.transition(RunStatus.FAILED)
        _safe_checkpoint(session, run, store)
        raise
    except Exception as exc:
        run.error = f"Unexpected runtime failure ({type(exc).__name__})"
        if run.status in {RunStatus.RUNNING, RunStatus.WAITING}:
            run.transition(RunStatus.FAILED)
        _safe_checkpoint(session, run, store)
        raise AgentLoopError(run.error) from None


def start_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    emit: EmitCallback | None = None,
    store: RunStore | None = None,
) -> Run:
    """Start one user-request run and execute until terminal state or a wait."""
    if run.max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if run.status is not RunStatus.CREATED:
        raise AgentLoopError("Only a newly created run can be started")
    if run.session_id != session.session_id:
        raise AgentLoopError(f"Run {run.run_id} does not belong to the session")
    if any(message.run_id == run.run_id for message in session.messages):
        raise AgentLoopError(f"Run {run.run_id} already has session messages")

    output = emit or (lambda _message: None)
    session.workspace = session.workspace.resolve()
    run.transition(RunStatus.RUNNING)
    session.messages.append(
        ModelMessage(
            role="user",
            content=({"type": "text", "text": run.goal},),
            run_id=run.run_id,
        )
    )
    if store is not None:
        store.create_run(session, run)
        output(f"Run started: {run.run_id}")
    return _run_with_failure_handling(session, run, model, registry, output, store)


def continue_run(
    session: Session,
    run: Run,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    emit: EmitCallback | None = None,
    store: RunStore | None = None,
) -> Run:
    """Continue a durable run after its text wait has been satisfied."""
    if run.status is not RunStatus.RUNNING:
        raise AgentLoopError("Only a running run can continue")
    if run.session_id != session.session_id:
        raise AgentLoopError(f"Run {run.run_id} does not belong to the session")
    if not any(message.run_id == run.run_id for message in session.messages):
        raise AgentLoopError("A resumed run must have model protocol history")
    output = emit or (lambda _message: None)
    return _run_with_failure_handling(session, run, model, registry, output, store)
