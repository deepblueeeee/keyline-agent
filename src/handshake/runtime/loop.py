"""The minimal real-model agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from handshake.models.base import ModelAdapter, ToolDefinition
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
from handshake.tools.registry import ToolRegistry

EmitCallback = Callable[[str], None]

_UPDATE_PLAN_TOOL = ToolDefinition(
    name="update_plan",
    description=(
        "Replace the current short task plan. Use this after initial investigation and when the "
        "approach materially changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
)


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


def _update_plan(task: Task, call: ToolCall) -> ToolResult:
    raw_steps = call.arguments.get("steps")
    if (
        not isinstance(raw_steps, list)
        or not raw_steps
        or any(not isinstance(step, str) or not step.strip() for step in raw_steps)
    ):
        return ToolResult(
            ok=False,
            stderr="steps must be a non-empty array of non-empty strings",
        )
    task.plan = [step.strip() for step in raw_steps]
    return ToolResult(ok=True, stdout="Plan updated")


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
    if call.name == "update_plan":
        result = _update_plan(task, call)
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


def _run_turns(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    max_steps: int,
    emit: EmitCallback,
) -> Task:
    definitions = (_UPDATE_PLAN_TOOL, *registry.definitions())
    for step_number in range(1, max_steps + 1):
        emit(f"Model turn {step_number}/{max_steps}")
        turn = model.complete(_context(task), definitions)
        if not turn.content:
            raise ModelProtocolError("Model returned an empty turn")
        _validate_turn_tool_calls(turn.content, turn.tool_calls)
        task.messages.append(ModelMessage(role="assistant", content=turn.content))

        if turn.tool_calls:
            if turn.text:
                emit(f"Model: {turn.text}")
            result_blocks: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                result = _execute_tool_call(task, call, registry, emit)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": _result_content(result),
                        "is_error": not result.ok,
                    }
                )
            task.messages.append(ModelMessage(role="user", content=tuple(result_blocks)))
            continue

        if turn.stop_reason == "end_turn" and turn.text:
            task.final_summary = turn.text
            task.transition(TaskStatus.COMPLETED)
            emit("Task completed")
            return task
        raise ModelProtocolError(
            "Model stopped without tool calls or a non-empty final end_turn response"
        )

    raise StepLimitExceededError(f"Agent exceeded the maximum of {max_steps} model turns")


def run_agent(
    task: Task,
    model: ModelAdapter,
    registry: ToolRegistry,
    *,
    max_steps: int = 50,
    emit: EmitCallback | None = None,
) -> Task:
    """Run a task until a final model response or an explicit failure."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if task.status is not TaskStatus.CREATED:
        raise AgentLoopError("Only a newly created task can be run")
    output = emit or (lambda _message: None)
    task.workspace = task.workspace.resolve()
    task.transition(TaskStatus.RUNNING)
    if not task.messages:
        task.messages.append(
            ModelMessage(
                role="user",
                content=({"type": "text", "text": task.goal},),
            )
        )
    try:
        return _run_turns(task, model, registry, max_steps, output)
    except UserCancelledError as exc:
        task.error = str(exc)
        task.transition(TaskStatus.CANCELLED)
        raise
    except HandshakeError as exc:
        task.error = str(exc)
        task.transition(TaskStatus.FAILED)
        raise
    except Exception as exc:
        task.error = f"Unexpected runtime failure ({type(exc).__name__})"
        task.transition(TaskStatus.FAILED)
        raise AgentLoopError(task.error) from None
