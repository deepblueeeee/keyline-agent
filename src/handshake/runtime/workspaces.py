"""Resolve Run workspace bindings and authorize main-agent workspace operations."""

from __future__ import annotations

import json
from collections.abc import Callable

from handshake.models.protocol import ToolCall
from handshake.runtime.persistence import RunStore
from handshake.runtime.runs import TERMINAL_RUN_STATUSES
from handshake.runtime.state import Run, RunStatus, Session, ToolResult, Worktree, WorktreeStatus
from handshake.storage.errors import RunNotFoundError, StorageError
from handshake.tools.worktree import CLEANUP_WORKTREE, INSPECT_CHANGES, INTEGRATE_CHANGES
from handshake.worktrees.manager import WorktreeError, WorktreeManager

WORKTREE_LIFETIME_NOTICE = (
    "Worktrees are temporary. Cleanup discards files outside the saved Git commit, "
    "including ignored files and local test data."
)


def owned_worktree(session: Session, owner_id: str, store: RunStore) -> tuple[Run, Worktree]:
    try:
        owner = store.load_run(owner_id)
    except RunNotFoundError:
        raise WorktreeError("Target Run does not exist in this Session") from None
    if owner.session_id != session.session_id:
        raise WorktreeError("Target Run does not belong to this Session")
    tree = store.load_worktree(owner_id)
    if owner.workspace_binding.worktree_owner_id != owner_id or tree is None:
        raise WorktreeError("Target Run does not own a Worktree")
    if tree.repository != session.workspace:
        raise WorktreeError("Worktree repository differs from the Session workspace")
    return owner, tree


def bound_worktree(session: Session, run: Run, store: RunStore) -> Worktree | None:
    owner_id = run.workspace_binding.worktree_owner_id
    return owned_worktree(session, owner_id, store)[1] if owner_id is not None else None


def test_outcomes(run: Run) -> list[dict[str, object]]:
    return [
        {
            "tool_call_id": tool.tool_call_id,
            "status": tool.status.value,
            "exit_code": tool.result.exit_code if tool.result else None,
        }
        for tool in run.tool_runs if tool.tool_name == "run_tests"
    ]


def reclaim_worktree(
    tree: Worktree, store: RunStore, manager: WorktreeManager, emit: Callable[[str], None],
) -> tuple[Worktree, str | None]:
    if not tree.ready_for_cleanup or tree.cleaned:
        return tree, None
    try:
        if not store.has_active_worktree_bindings(tree.owner_run_id):
            emit(
                f"Reclaiming temporary Worktree for {tree.owner_run_id}. {WORKTREE_LIFETIME_NOTICE}"
            )
        updated = manager.cleanup(tree, store, abandon=False)
    except (WorktreeError, StorageError) as exc:
        emit(f"Worktree for {tree.owner_run_id} is {tree.status.value}; cleanup failed: {exc}")
        return store.load_worktree(tree.owner_run_id) or tree, str(exc)
    if not updated.cleaned:
        emit(f"Worktree cleanup deferred; another Run is using {tree.owner_run_id}")
    return updated, None


def reclaim_session_worktrees(
    session: Session, store: RunStore, manager: WorktreeManager, emit: Callable[[str], None],
) -> None:
    for run in store.list_runs(session.session_id):
        if (
            run.status in TERMINAL_RUN_STATUSES
            and run.workspace_binding.worktree_owner_id == run.run_id
        ):
            try:
                tree = bound_worktree(session, run, store)
            except WorktreeError as exc:
                emit(f"Worktree cleanup failed for {run.run_id}: {exc}")
                continue
            if tree is not None:
                reclaim_worktree(tree, store, manager, emit)


def execute_worktree_operation(
    session: Session, run: Run, call: ToolCall, store: RunStore, manager: WorktreeManager,
    emit: Callable[[str], None] = lambda _message: None,
) -> ToolResult:
    if run.parent_run_id is not None or run.workspace_binding.worktree_owner_id is not None:
        return ToolResult(False, stderr="Only the main Run can manage child Worktrees")
    owner_id = call.arguments.get("child_run_id")
    if not isinstance(owner_id, str) or not owner_id:
        return ToolResult(False, stderr="child_run_id must be a non-empty Run ID")

    try:
        owner, tree = owned_worktree(session, owner_id, store)
        if owner.status not in TERMINAL_RUN_STATUSES:
            raise WorktreeError("Worktree owner is still running or waiting")
        if (
            call.name != CLEANUP_WORKTREE.name
            and owner.status is RunStatus.COMPLETED
            and tree.status is WorktreeStatus.ACTIVE
        ):
            tree = manager.snapshot(tree, store)
        if call.name == INSPECT_CHANGES.name:
            start = call.arguments.get("start_line", 1)
            if type(start) is not int or start < 1:
                raise WorktreeError("start_line must be a positive integer")
            data = {**manager.changes(tree), **manager.diff(tree, start_line=start)}
        elif call.name == INTEGRATE_CHANGES.name:
            if owner.status is not RunStatus.COMPLETED:
                raise WorktreeError("Only changes from a completed implementer may be integrated")
            tree = manager.integrate(tree, store)
            data = manager.changes(tree)
            tree, cleanup_error = reclaim_worktree(tree, store, manager, emit)
            data["cleaned"] = tree.cleaned
            if cleanup_error is not None:
                data["cleanup_error"] = cleanup_error
                data["cleanup_status"] = "failed"
            else:
                data["cleanup_status"] = "cleaned" if tree.cleaned else "deferred"
        elif call.name == CLEANUP_WORKTREE.name:
            abandon = call.arguments.get("abandon", False)
            if type(abandon) is not bool:
                raise WorktreeError("abandon must be a boolean")
            if not tree.cleaned:
                emit(WORKTREE_LIFETIME_NOTICE)
            tree = manager.cleanup(tree, store, abandon=abandon)
            data = manager.changes(tree)
            data["cleanup_status"] = "cleaned" if tree.cleaned else "deferred"
            if not tree.cleaned:
                emit(f"Worktree cleanup deferred; another Run is using {tree.owner_run_id}")
        else:
            raise WorktreeError(f"Unknown Worktree operation: {call.name}")
        data["tests"] = test_outcomes(owner)
        return ToolResult(True, stdout=json.dumps(data, ensure_ascii=False))
    except WorktreeError as exc:
        return ToolResult(False, stderr=str(exc))
