"""Resolve Run workspace bindings and authorize main-agent workspace operations."""

from __future__ import annotations

import json

from handshake.models.protocol import ToolCall
from handshake.runtime.persistence import RunStore
from handshake.runtime.runs import TERMINAL_RUN_STATUSES
from handshake.runtime.state import Run, RunStatus, Session, ToolResult, Worktree, WorktreeStatus
from handshake.storage.errors import RunNotFoundError
from handshake.tools.worktree import CLEANUP_WORKTREE, INSPECT_CHANGES, INTEGRATE_CHANGES
from handshake.worktrees.manager import WorktreeError, WorktreeManager


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


def execute_worktree_operation(
    session: Session, run: Run, call: ToolCall, store: RunStore, manager: WorktreeManager,
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
        elif call.name == CLEANUP_WORKTREE.name:
            abandon = call.arguments.get("abandon", False)
            if type(abandon) is not bool:
                raise WorktreeError("abandon must be a boolean")
            if any(
                other.run_id != owner_id
                and other.workspace_binding.worktree_owner_id == owner_id
                and other.status not in TERMINAL_RUN_STATUSES
                for other in store.list_runs(session.session_id)
            ):
                raise WorktreeError("Another Run still uses this Worktree")
            tree = manager.cleanup(tree, store, abandon=abandon)
            data = manager.changes(tree)
        else:
            raise WorktreeError(f"Unknown Worktree operation: {call.name}")
        data["tests"] = test_outcomes(owner)
        return ToolResult(True, stdout=json.dumps(data, ensure_ascii=False))
    except WorktreeError as exc:
        return ToolResult(False, stderr=str(exc))
