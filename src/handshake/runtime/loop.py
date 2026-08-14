"""The model/tool loop with durable text waits and automatic resumption."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from handshake.models.base import ModelAdapter
from handshake.runtime.errors import (
    AgentLoopError,
    HandshakeError,
    ModelProtocolError,
    StepLimitExceededError,
    UserCancelledError,
)
from handshake.runtime.state import (
    ModelContext,
    ModelMessage,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolRun,
    ToolRunStatus,
)
from handshake.storage.base import TaskStore
from handshake.tools import (
    PLAN_TOOL,
    TEXT_WAIT_TOOL,
    handle_plan,
    prepare_text_wait,
    runtime_tool_definitions,
)
from handshake.tools.registry import ToolRegistry

EmitCallback = Callable[[str], None]

def _system_prompt(task: Task) -> str:
    plan = "\n".join(f"- {step}" for step in task.plan) or "(not set yet)"
    return f"""You are Handshake, a local coding agent operating on one workspace.

Goal:
{task.goal}

Workspace:
{task.workspace}

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
- When the task is complete, respond without tool calls with a concise summary of changes,
  validation performed, and any unresolved issue.
"""


def _context(task: Task) -> ModelContext:
    return ModelContext(
        system_prompt=_system_prompt(task),
        goal=task.goal,
        workspace=str(task.workspace),
        plan=tuple(task.plan),
        messages=tuple(task.messages),
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
    task: Task,
    call: ToolCall,
    registry: ToolRegistry,
    emit: EmitCallback,
) -> ToolResult:
    if call.name == PLAN_TOOL.name:
        result = handle_plan(task, call)
        if result.ok:
            emit("Plan updated: " + " | ".join(task.plan))
        else:
            emit(f"Plan update failed: {result.stderr}")
        return result

    tool_run = ToolRun(
        tool_call_id=call.id,
        tool_name=call.name,
        arguments=dict(call.arguments),
    )
    task.tool_runs.append(tool_run)
    tool_run.transition(ToolRunStatus.RUNNING)
    emit(f"Running tool: {call.name}")
    result = registry.execute(call.name, task.workspace, call.arguments)
    tool_run.result = result
    tool_run.transition(ToolRunStatus.SUCCEEDED if result.ok else ToolRunStatus.FAILED)
    emit(f"Tool {call.name}: {'succeeded' if result.ok else 'failed'}")
    return result


def _append_tool_results(
    task: Task, calls: tuple[ToolCall, ...], results: tuple[ToolResult, ...]
) -> None:
    task.messages.append(
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
        )
    )


def _checkpoint(task: Task, store: TaskStore | None) -> None:
    if store is not None:
        store.checkpoint_task(task)


def _create_text_wait(
    task: Task,
    call: ToolCall,
    store: TaskStore | None,
    emit: EmitCallback,
) -> bool:
    prepared = prepare_text_wait(call)
    if isinstance(prepared, ToolResult):
        _append_tool_results(
            task,
            (call,),
            (prepared,),
        )
        _checkpoint(task, store)
        return False
    if store is None:
        _append_tool_results(
            task,
            (call,),
            (
                ToolResult(
                    ok=False,
                    stderr="wait_for_text requires durable task storage",
                ),
            ),
        )
        return False

    work_item, wait = prepared
    task.work_items.append(work_item)
    task.waits.append(wait)
    task.transition(TaskStatus.WAITING)
    store.checkpoint_task(task)
    emit(f"Waiting for text: {wait.prompt}")
    return True


def _run_turns(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    emit: EmitCallback,
    store: TaskStore | None,
) -> Task:
    definitions = (*runtime_tool_definitions(), *registry.definitions())
    while task.model_turns < task.max_steps:
        emit(f"Model turn {task.model_turns + 1}/{task.max_steps}")
        turn = model.complete(_context(task), definitions)
        task.model_turns += 1
        if not turn.content:
            raise ModelProtocolError("Model returned an empty turn")
        _validate_turn_tool_calls(turn.content, turn.tool_calls)
        task.messages.append(ModelMessage(role="assistant", content=turn.content))

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
                    task,
                    turn.tool_calls,
                    tuple(error for _call in turn.tool_calls),
                )
                _checkpoint(task, store)
                emit("Rejected tool turn: wait_for_text must be called alone")
                continue
            if wait_calls:
                if _create_text_wait(task, wait_calls[0], store, emit):
                    return task
                continue

            results = tuple(
                _execute_tool_call(task, call, registry, emit)
                for call in turn.tool_calls
            )
            _append_tool_results(task, turn.tool_calls, results)
            _checkpoint(task, store)
            continue

        if turn.stop_reason == "end_turn" and turn.text:
            task.final_summary = turn.text
            task.transition(TaskStatus.COMPLETED)
            _checkpoint(task, store)
            emit("Task completed")
            return task
        raise ModelProtocolError(
            "Model stopped without tool calls or a non-empty final end_turn response"
        )

    raise StepLimitExceededError(
        f"Agent exceeded the maximum of {task.max_steps} model turns"
    )


def _safe_checkpoint(task: Task, store: TaskStore | None) -> None:
    if store is None:
        return
    with suppress(HandshakeError):
        store.checkpoint_task(task)


def _run_with_failure_handling(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    emit: EmitCallback,
    store: TaskStore | None,
) -> Task:
    try:
        return _run_turns(task, model, registry, emit, store)
    except UserCancelledError as exc:
        task.error = str(exc)
        if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING}:
            task.transition(TaskStatus.CANCELLED)
        _safe_checkpoint(task, store)
        raise
    except HandshakeError as exc:
        task.error = str(exc)
        if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING}:
            task.transition(TaskStatus.FAILED)
        _safe_checkpoint(task, store)
        raise
    except Exception as exc:
        task.error = f"Unexpected runtime failure ({type(exc).__name__})"
        if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING}:
            task.transition(TaskStatus.FAILED)
        _safe_checkpoint(task, store)
        raise AgentLoopError(task.error) from None


def run_agent(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    max_steps: int = 50,
    emit: EmitCallback | None = None,
    store: TaskStore | None = None,
) -> Task:
    """Start a task and run until completion, failure, or a durable wait."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if task.status is not TaskStatus.CREATED:
        raise AgentLoopError("Only a newly created task can be run")
    output = emit or (lambda _message: None)
    task.workspace = task.workspace.resolve()
    task.max_steps = max_steps
    task.transition(TaskStatus.RUNNING)
    if not task.messages:
        task.messages.append(
            ModelMessage(
                role="user",
                content=({"type": "text", "text": task.goal},),
            )
        )
    if store is not None:
        store.create_task(task)
        output(f"Task created: {task.task_id}")
    return _run_with_failure_handling(task, model, registry, output, store)


def continue_agent(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    emit: EmitCallback | None = None,
    store: TaskStore | None = None,
) -> Task:
    """Continue a durable task after its text wait has been satisfied."""
    if task.status is not TaskStatus.RUNNING:
        raise AgentLoopError("Only a running task can continue")
    if not task.messages:
        raise AgentLoopError("A resumed task must contain model protocol history")
    output = emit or (lambda _message: None)
    return _run_with_failure_handling(task, model, registry, output, store)
