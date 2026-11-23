"""Prepare tools without holding the workspace slot, then persist execution and outcome."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import replace
from threading import Event

from handshake.inputs.cancellation import cancellation_scope, check_cancelled
from handshake.models.protocol import ToolCall
from handshake.runtime.errors import AgentLoopError, ToolOutcomeUnknownError
from handshake.runtime.persistence import RunStore
from handshake.runtime.progress import emit_wait
from handshake.runtime.scheduler import CooperativeScheduler, run_blocking
from handshake.runtime.state import ReplayPolicy, Run, Session, ToolResult, ToolRun, ToolRunStatus
from handshake.runtime.workspaces import bound_worktree, execute_worktree_operation
from handshake.tools.registry import PreparedTool, ToolPreparation, ToolRegistry
from handshake.tools.secret_command import SecretCommandRunner
from handshake.tools.shell import worktree_process_scope
from handshake.tools.worktree import WORKTREE_TOOL_NAMES
from handshake.worktrees.manager import WorktreeManager


def _pending_tool(
    session: Session, run: Run, call: ToolCall, registry: ToolRegistry, store: RunStore
) -> ToolRun:
    tool = next((item for item in run.tool_runs
                 if item.model_turn == run.model_turns and item.tool_call_id == call.id), None)
    if tool is None:
        policy = registry.replay_policy(call.name)
        if call.name == "run_command_with_secret":
            policy = ReplayPolicy.MANUAL
        elif call.name in WORKTREE_TOOL_NAMES:
            # These operations reconcile their own Git intents before issuing any mutation.
            policy = ReplayPolicy.AUTO
        tool = ToolRun(
            tool_call_id=call.id, tool_name=call.name, arguments=dict(call.arguments),
            model_turn=run.model_turns,
            replay_policy=policy,
        )
        run.add_tool_run(tool)
        store.checkpoint_run(session, run)
    elif tool.tool_name != call.name or tool.arguments != call.arguments:
        raise AgentLoopError("Persisted tool call does not match model protocol history")
    if tool.status is ToolRunStatus.RUNNING:
        tool.transition(ToolRunStatus.UNKNOWN)
        store.checkpoint_run(session, run)
    if tool.status is ToolRunStatus.UNKNOWN and tool.replay_policy is ReplayPolicy.MANUAL:
        raise ToolOutcomeUnknownError(
            f"Tool {call.name} ({call.id}) has an unknown outcome; "
            "it was not retried automatically"
        )
    return tool


def _save_result(
    session: Session, run: Run, tool: ToolRun, result: ToolResult, store: RunStore
) -> ToolResult:
    tool.result = result
    tool.transition(ToolRunStatus.SUCCEEDED if result.ok else ToolRunStatus.FAILED)
    store.checkpoint_run(session, run)
    return result


async def execute_tool_call(
    session: Session,
    run: Run,
    call: ToolCall,
    registry: ToolRegistry,
    secret_command_runner: SecretCommandRunner | None,
    emit: Callable[[str], None],
    store: RunStore,
    scheduler: CooperativeScheduler,
    *,
    worktrees: WorktreeManager | None = None,
    allowed: bool = True,
) -> ToolResult:
    tool = _pending_tool(session, run, call, registry, store)
    if tool.status in {ToolRunStatus.SUCCEEDED, ToolRunStatus.FAILED}:
        if tool.result is None:
            raise AgentLoopError("Completed tool has no durable result")
        return tool.result
    if not allowed:
        result = ToolResult(False, stderr="Tool is no longer allowed for this agent")
        if tool.status is ToolRunStatus.PENDING:
            tool.transition(ToolRunStatus.RUNNING)
            return _save_result(session, run, tool, result, store)
        return result

    worktrees = worktrees or WorktreeManager()
    tree = bound_worktree(session, run, store)
    if tree is not None:
        await run_blocking(lambda: worktrees.validate(tree))
    workspace = run.workspace(session, tree)
    cancellation = Event()
    secret_run = (replace(run, secret_handles=[replace(h) for h in run.secret_handles])
                  if call.name == "run_command_with_secret" else None)
    original_workspace = session.workspace if tree is not None else None
    with cancellation_scope(cancellation), worktree_process_scope(original_workspace):
        emit(f"Preparing tool: {call.name}")
        preparation: ToolPreparation
        if call.name in WORKTREE_TOOL_NAMES:
            def operation() -> ToolResult:
                return execute_worktree_operation(session, run, call, store, worktrees)

            preparation = operation
        elif secret_run is not None:
            if secret_command_runner is None:
                preparation = ToolResult(ok=False, stderr=(
                    "Secret command capability is unavailable; no shell fallback was used"
                ))
            else:
                preparation = await run_blocking(
                    lambda: secret_command_runner.prepare(
                        secret_run, workspace, call.arguments
                    ),
                    on_cancel=cancellation.set,
                )
        else:
            preparation = await run_blocking(
                lambda: registry.prepare(call.name, workspace, call.arguments),
                on_cancel=cancellation.set,
            )
        prepared: PreparedTool = (
            await preparation if inspect.isawaitable(preparation) else preparation
        )
        if isinstance(prepared, ToolResult):
            tool.transition(ToolRunStatus.RUNNING)
            return _save_result(session, run, tool, prepared, store)
        emit_wait(emit, f"Waiting for tool slot: {call.name}", "tool_slot")
        async with scheduler.tool_slot():
            check_cancelled()
            if tree is not None:
                await run_blocking(lambda: worktrees.validate(tree))
            tool.transition(ToolRunStatus.RUNNING)
            store.checkpoint_run(session, run)
            emit(f"Running tool: {call.name}")

            async def execute() -> ToolResult | BaseException:
                try:
                    try:
                        result = await run_blocking(prepared)
                    except (Exception, asyncio.CancelledError):
                        tool.transition(ToolRunStatus.UNKNOWN)
                        raise
                    finally:
                        if secret_run is not None:
                            for handle in secret_run.secret_handles:
                                current = run.secret_handle_by_id(handle.secret_handle)
                                if current is not None:
                                    current.status = handle.status
                                    current.updated_at = handle.updated_at
                        if tool.status is ToolRunStatus.UNKNOWN:
                            store.checkpoint_run(session, run)
                    return _save_result(session, run, tool, result, store)
                except BaseException as exc:
                    return exc

            task = asyncio.create_task(execute(), name=f"tool:{call.id}")
            try:
                outcome = await asyncio.shield(task)
            except asyncio.CancelledError:
                # External work has started: retain its slot until its outcome is durable.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                raise
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
