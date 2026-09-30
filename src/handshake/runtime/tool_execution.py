"""Durable execution and conservative replay of one workspace tool call."""

from __future__ import annotations

from collections.abc import Callable

from handshake.models.protocol import ToolCall
from handshake.runtime.errors import AgentLoopError, ToolOutcomeUnknownError
from handshake.runtime.persistence import RunStore
from handshake.runtime.state import (
    ReplayPolicy,
    Run,
    Session,
    ToolResult,
    ToolRun,
    ToolRunStatus,
)
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner


def execute_tool_call(
    session: Session,
    run: Run,
    call: ToolCall,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: Callable[[str], None],
    store: RunStore,
) -> ToolResult:
    """Commit intent before execution and outcome afterward, replaying only safe tools."""
    tool_run = next(
        (
            item for item in run.tool_runs
            if item.model_turn == run.model_turns and item.tool_call_id == call.id
        ),
        None,
    )
    if tool_run is None:
        policy = (
            ReplayPolicy.MANUAL
            if call.name == "run_command_with_secret"
            else registry.replay_policy(call.name)
        )
        tool_run = ToolRun(
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments),
            model_turn=run.model_turns,
            replay_policy=policy,
        )
        run.add_tool_run(tool_run)
        tool_run.transition(ToolRunStatus.RUNNING)
        store.checkpoint_run(session, run)
    else:
        if tool_run.tool_name != call.name or tool_run.arguments != call.arguments:
            raise AgentLoopError("Persisted tool call does not match model protocol history")
        if tool_run.status in {ToolRunStatus.SUCCEEDED, ToolRunStatus.FAILED}:
            if tool_run.result is None:
                raise AgentLoopError("Completed tool has no durable result")
            return tool_run.result
        if tool_run.status is ToolRunStatus.RUNNING:
            tool_run.transition(ToolRunStatus.UNKNOWN)
            store.checkpoint_run(session, run)
        if tool_run.status is ToolRunStatus.UNKNOWN:
            if tool_run.replay_policy is ReplayPolicy.MANUAL:
                raise ToolOutcomeUnknownError(
                    f"Tool {call.name} ({call.id}) has an unknown outcome; "
                    "it was not retried automatically"
                )
            tool_run.transition(ToolRunStatus.RUNNING)
            store.checkpoint_run(session, run)
        elif tool_run.status is not ToolRunStatus.RUNNING:
            raise AgentLoopError("Persisted tool has no executable state")

    emit(f"Running tool: {call.name}")
    try:
        if call.name == "run_command_with_secret":
            if secret_command_runner is None:
                result = ToolResult(
                    ok=False,
                    stderr="Secret command capability is unavailable; no shell fallback was used",
                )
            else:
                result = secret_command_runner.execute(
                    run, session.workspace, call.arguments
                )
        else:
            result = registry.execute(call.name, session.workspace, call.arguments)
    except Exception:
        tool_run.transition(ToolRunStatus.UNKNOWN)
        store.checkpoint_run(session, run)
        raise
    tool_run.result = result
    tool_run.transition(ToolRunStatus.SUCCEEDED if result.ok else ToolRunStatus.FAILED)
    store.checkpoint_run(session, run)
    return result
