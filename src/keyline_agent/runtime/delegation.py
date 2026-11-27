"""Durable child batches and structured delegation handoff."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field

from keyline_agent.context import ActiveContextManagers
from keyline_agent.context.errors import ContextPressureError
from keyline_agent.errors import KeylineError
from keyline_agent.models.base import ModelAdapter, close_model
from keyline_agent.models.protocol import ModelMessage, ToolCall
from keyline_agent.runtime.agents import AgentSpec, model_name_for_run
from keyline_agent.runtime.errors import AgentLoopError, UserCancelledError
from keyline_agent.runtime.loop import (
    EmitCallback,
    _checkpoint,
    _run_turns,
    _sync_context,
    context_manager_for_run,
    continue_run,
)
from keyline_agent.runtime.persistence import RunStore
from keyline_agent.runtime.progress import bind_reporter
from keyline_agent.runtime.recovery import resume_run
from keyline_agent.runtime.runs import TERMINAL_RUN_STATUSES, ChildRunResult, create_child_run
from keyline_agent.runtime.scheduler import CooperativeScheduler, run_blocking
from keyline_agent.runtime.state import (
    Run,
    RunStatus,
    Session,
    ToolResult,
    WorkspaceBinding,
    WorktreeStatus,
)
from keyline_agent.runtime.workspaces import (
    WORKTREE_LIFETIME_NOTICE,
    bound_worktree,
    owned_worktree,
    reclaim_worktree,
    test_outcomes,
)
from keyline_agent.storage.errors import StorageError, WorktreeUnavailableError
from keyline_agent.tools import DELEGATE_TOOL
from keyline_agent.tools.registry import ToolRegistry
from keyline_agent.tools.secret_command import SecretCommandRunner
from keyline_agent.worktrees.manager import WorktreeError, WorktreeManager


@dataclass(frozen=True, slots=True)
class DelegatedTask:
    spec: AgentSpec
    goal: str
    target_run_id: str | None = None


@dataclass(slots=True)
class DelegationRuntime:
    """Configuration and process-local services for a child batch."""

    agent_specs: Mapping[str, AgentSpec]
    model_for_spec: Callable[[AgentSpec], ModelAdapter]
    context_managers: ActiveContextManagers
    scheduler: CooperativeScheduler = field(default_factory=CooperativeScheduler)
    resolve_input: Callable[[Run], Awaitable[ModelMessage | None]] | None = None
    worktrees: WorktreeManager = field(default_factory=WorktreeManager)

    def prepare_batch(
        self, calls: tuple[ToolCall, ...]
    ) -> tuple[DelegatedTask, ...] | ToolResult:
        prepared = tuple(_delegation_arguments(call, self) for call in calls)
        error = next((item for item in prepared if isinstance(item, ToolResult)), None)
        if error is not None:
            return error
        tasks = tuple(item for item in prepared if not isinstance(item, ToolResult))
        if sum(task.spec.spec_id == "implementer" for task in tasks) > 3:
            return ToolResult(False, stderr="A delegation batch allows at most 3 implementers")
        return tasks

    async def run_batch(
        self,
        session: Session,
        parent: Run,
        calls: tuple[ToolCall, ...],
        prepared: tuple[DelegatedTask, ...],
        registry: ToolRegistry,
        secret_command_runner: SecretCommandRunner | None,
        emit: EmitCallback,
        store: RunStore,
    ) -> tuple[Run, ...]:
        children = await run_blocking(
            lambda: _ensure_children(session, parent, calls, prepared, store, self.worktrees)
        )
        for child in children:
            if child.status is RunStatus.COMPLETED:
                await _save_child_changes(child, store, self.worktrees, bind_reporter(emit, child))
        return await self.scheduler.advance(
            children,
            lambda child: _run_child(
                session, child, registry, secret_command_runner, emit, store, self
            ),
            include_waiting=self.resolve_input is not None,
        )

    async def append_results(
        self, pending_messages: list[ModelMessage], parent: Run, children: tuple[Run, ...],
        store: RunStore,
    ) -> None:
        if not self.scheduler.ready_to_join(children):
            raise AgentLoopError("Cannot join a delegation batch with active children")
        pending_messages.append(
            ModelMessage(
                role="user",
                content=await run_blocking(lambda: tuple(
                    _child_result_block(child, store, self.worktrees) for child in children
                )),
                run_id=parent.run_id,
            )
        )


def _delegation_arguments(
    call: ToolCall, delegation: DelegationRuntime
) -> DelegatedTask | ToolResult:
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
    target = call.arguments.get("target_run_id")
    if "target_run_id" in call.arguments:
        if spec_id not in {"explorer", "reviewer"}:
            return ToolResult(False, stderr="Only explorer and reviewer can set target_run_id")
        if not isinstance(target, str) or not target.strip():
            return ToolResult(False, stderr="target_run_id must be a non-empty Run ID")
        return DelegatedTask(spec, goal.strip(), target.strip())
    return DelegatedTask(spec, goal.strip())


def _child_result_block(
    child: Run, store: RunStore, manager: WorktreeManager,
) -> dict[str, object]:
    result = asdict(ChildRunResult.from_run(child))
    if child.workspace_binding.worktree_owner_id == child.run_id:
        tree = store.load_worktree(child.run_id)
        if tree is not None:
            result["worktree"] = {
                "path": str(tree.path), "branch": tree.branch,
                "status": tree.status.value, "commit": tree.saved_commit,
            }
            result["tests"] = test_outcomes(child)
            if tree.saved_commit is not None:
                try:
                    result["changes"] = manager.changes(tree)
                except WorktreeError as exc:
                    result["change_error"] = str(exc)
            elif child.status is RunStatus.COMPLETED:
                result["change_error"] = "Changes could not be saved; the Worktree is retained"
    if child.delegation_tool_call_id is None:
        raise AgentLoopError("Child has no delegation tool call")
    return {
        "type": "tool_result",
        "tool_use_id": child.delegation_tool_call_id,
        "content": json.dumps(result, ensure_ascii=False),
        "is_error": child.status is not RunStatus.COMPLETED,
    }


def _ensure_children(
    session: Session,
    parent: Run,
    calls: tuple[ToolCall, ...],
    prepared: tuple[DelegatedTask, ...],
    store: RunStore,
    manager: WorktreeManager,
) -> tuple[Run, ...]:
    if len(calls) != len(prepared) or not calls:
        raise AgentLoopError("Delegation batch must contain prepared tool calls")
    runs = list(store.list_runs(session.session_id))
    existing = {
        run.delegation_tool_call_id: run for run in runs if run.parent_run_id == parent.run_id
    }
    children: list[Run] = []
    base = None
    # Pin remaining children to the original batch base after partial creation.
    for call in calls:
        child = existing.get(call.id)
        if child is not None and child.workspace_binding.worktree_owner_id == child.run_id:
            record = store.load_worktree(child.run_id)
            if record is not None:
                base = (record.repository, record.base_commit, record.main_ref)
                break

    for call, task in zip(calls, prepared, strict=True):
        child = existing.get(call.id)
        if child is not None:
            expected_owner = (
                child.run_id if task.spec.spec_id == "implementer" else task.target_run_id
            )
            if (
                child.agent_spec_id != task.spec.spec_id or child.goal != task.goal
                or (child.status is not RunStatus.FAILED
                    and child.workspace_binding.worktree_owner_id != expected_owner)
            ):
                raise AgentLoopError("Stored child does not match its delegation tool call")
        else:
            child = create_child_run(
                session, runs, parent, task.goal, agent_spec_id=task.spec.spec_id,
                model_name=task.spec.model_name, max_steps=task.spec.default_max_steps,
                tool_call_id=call.id,
            )
            record = None
            child.transition(RunStatus.RUNNING)
            try:
                if task.spec.spec_id == "implementer":
                    base = base or manager.creation_base(session.workspace)
                    record = manager.plan(child.run_id, base)
                    child.workspace_binding = WorkspaceBinding(child.run_id)
                elif task.target_run_id is not None:
                    owner, target = owned_worktree(session, task.target_run_id, store)
                    if owner.status not in TERMINAL_RUN_STATUSES:
                        raise WorktreeError("Target implementer is still active")
                    manager.validate(target)
                    if not target.accepts_bindings:
                        raise WorktreeError("Target Worktree is closed to new bindings")
                    child.workspace_binding = WorkspaceBinding(task.target_run_id)
            except WorktreeError as exc:
                child.error = str(exc)
                child.transition(RunStatus.FAILED)

            messages = (
                ModelMessage("user", ({"type": "text", "text": child.goal},), run_id=child.run_id),
            )
            if record is None:
                try:
                    store.create_run(session, child, messages)
                except WorktreeUnavailableError as exc:
                    child.workspace_binding = WorkspaceBinding()
                    child.error = str(exc)
                    child.transition(RunStatus.FAILED)
                    store.create_run(session, child, messages)
            else:
                store.create_run(session, child, messages, worktree=record)
            runs.append(child)
        children.append(child)
    return tuple(children)


async def _save_child_changes(
    child: Run, store: RunStore, manager: WorktreeManager, emit: EmitCallback,
) -> None:
    if child.workspace_binding.worktree_owner_id != child.run_id:
        return
    tree = store.load_worktree(child.run_id)
    if tree is not None and tree.status is WorktreeStatus.ACTIVE:
        try:
            await run_blocking(lambda: manager.snapshot(tree, store))
        except WorktreeError as exc:
            emit(f"Could not save changes. Worktree retained: {exc}")
        else:
            emit("Saved changes in Worktree")


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
    model: ModelAdapter | None = None
    try:
        spec = delegation.agent_specs.get(child.agent_spec_id)
        if spec is None:
            raise AgentLoopError(f"Agent spec {child.agent_spec_id} is unavailable")
        tree = bound_worktree(session, child, store)
        if tree is not None:
            if tree.owner_run_id == child.run_id:
                await run_blocking(lambda: delegation.worktrees.provision(tree, store))
                if tree.status is WorktreeStatus.PLANNED:
                    emit(WORKTREE_LIFETIME_NOTICE)
            else:
                await run_blocking(lambda: delegation.worktrees.validate(tree))
        model = delegation.scheduler.model(delegation.model_for_spec(spec))
        manager = context_manager_for_run(delegation.context_managers, child, spec, model)
        if child.status is RunStatus.RUNNING:
            child = await resume_run(
                session, child, model, registry,
                store=store, agent_spec=spec, emit=emit,
                secret_command_runner=secret_command_runner,
                context_manager=manager, scheduler=delegation.scheduler, delegation=delegation,
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
                delegation=delegation,
            )
        if child.status is RunStatus.COMPLETED:
            await _save_child_changes(child, store, delegation.worktrees, emit)
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
            if isinstance(exc, (KeylineError, ContextPressureError))
            else f"Unexpected child failure ({type(exc).__name__})"
        )
        persisted.transition(RunStatus.FAILED)
        store.checkpoint_run(session, persisted)
        bind_reporter(emit, persisted)(f"Run failed: {persisted.error}")
        return persisted
    finally:
        if model is not None:
            await close_model(model)
        owner_id = child.workspace_binding.worktree_owner_id
        if (
            owner_id is not None and owner_id != child.run_id
            and store.load_run(child.run_id).status in TERMINAL_RUN_STATUSES
        ):
            target = store.load_worktree(owner_id)
            if target is not None:
                await run_blocking(lambda: reclaim_worktree(
                    target, store, delegation.worktrees, emit,
                ))


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
        await delegation.append_results(pending, parent, children, store)
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
