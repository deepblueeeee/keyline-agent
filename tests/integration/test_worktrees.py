"""Exercise the production delegation, tool and recovery paths with real Git repositories."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

from keyline_agent import cli
from keyline_agent.context import ActiveContextManagers
from keyline_agent.context.budget import ContextBudgetPolicy
from keyline_agent.inputs import EphemeralSecretStore
from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ModelContext, ModelTurn, ToolCall
from keyline_agent.models.scripted import ScriptedModel
from keyline_agent.runtime.agents import child_agent_specs, main_agent_spec
from keyline_agent.runtime.delegation import DelegationRuntime, continue_delegation
from keyline_agent.runtime.errors import UserCancelledError
from keyline_agent.runtime.loop import start_run
from keyline_agent.runtime.recovery import resume_run, update_paused_step_limit
from keyline_agent.runtime.runs import create_child_run
from keyline_agent.runtime.secret_dependencies import resolve_secret_dependency
from keyline_agent.runtime.state import (
    Run,
    RunStatus,
    Session,
    ToolRunStatus,
    WorkspaceBinding,
    WorktreeStatus,
)
from keyline_agent.runtime.workspaces import execute_worktree_operation
from keyline_agent.storage.errors import WorktreeUnavailableError
from keyline_agent.storage.sqlite import SQLiteStateStore
from keyline_agent.tools import create_default_registry, runtime_tool_definitions
from keyline_agent.tools.registry import ToolBinding
from keyline_agent.tools.secret_command import SecretCommandRunner
from keyline_agent.worktrees import manager as git_worktrees
from keyline_agent.worktrees.manager import WorktreeError, WorktreeManager
from tests.history_support import run_async


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


def turn(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(
        tuple({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
              for call in calls),
        calls, "tool_use",
    )


def done() -> ModelTurn:
    return ModelTurn(({"type": "text", "text": "Finished"},), stop_reason="end_turn")


def delegate(call_id: str, spec: str, goal: str, **options: str) -> ToolCall:
    return ToolCall(call_id, "delegate", {"agent_spec_id": spec, "goal": goal, **options})


class Scenario:
    def __init__(self, tmp_path: Path) -> None:
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "main")
        git(self.repo, "config", "user.name", "Test")
        git(self.repo, "config", "user.email", "test@example.test")
        (self.repo / "shared.txt").write_text("base\n")
        (self.repo / "remove.txt").write_text("remove\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        self.session = Session(self.repo)
        self.store = SQLiteStateStore(tmp_path / "state.db")
        self.store.create_session(self.session)
        self.parent = Run("Coordinate", self.session.session_id, "main")
        self.approvals: list[Path] = []

        def approve(cwd: Path, _argv: tuple[str, ...]) -> bool:
            self.approvals.append(cwd)
            return True

        self.registry = create_default_registry(approve)
        budget = ContextBudgetPolicy()
        self.main = main_agent_spec(
            "scripted", (tool.name for tool in (
                *self.registry.definitions(), *runtime_tool_definitions(),
            )), budget,
        )
        self.specs = {s.spec_id: s for s in child_agent_specs("scripted", budget)}
        self.manager = WorktreeManager(tmp_path / "worktrees")
        self.runtime = DelegationRuntime(
            self.specs, lambda _spec: ScriptedModel([]), ActiveContextManagers(),
            worktrees=self.manager,
        )

    def children(self) -> tuple[Run, ...]:
        return self.store.list_runs(self.session.session_id)[1:]

    def results(self) -> dict[str, dict]:
        return {
            block["tool_use_id"]: block
            for row in self.store.read_messages(self.parent.context_scope_id)
            for block in row.message.content if block["type"] == "tool_result"
        }

    async def start(
        self, model, children=None, *, secrets: SecretCommandRunner | None = None,
    ) -> None:
        if children is not None:
            self.runtime.model_for_spec = children
        await start_run(
            self.session, self.parent, model, self.registry, store=self.store,
            agent_spec=self.main, delegation=self.runtime, secret_command_runner=secrets,
        )

    async def restart_batch(self, model, children) -> Run:
        self.store = SQLiteStateStore(self.store.database_path)
        self.session = self.store.load_session(self.session.session_id)
        self.parent = self.store.load_run(self.parent.run_id)
        self.runtime = DelegationRuntime(
            self.specs, children, ActiveContextManagers(),
            worktrees=WorktreeManager(self.manager.root),
        )
        return await continue_delegation(
            self.session, self.parent, model, self.registry, store=self.store,
            agent_spec=self.main, delegation=self.runtime,
        )


@run_async
async def test_parallel_implementers_and_read_only_worktree_review(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    waiting = asyncio.Event()
    started: set[str] = set()
    seen: dict[str, Path] = {}

    class Implementer:
        async def complete(self, context: ModelContext, tools: tuple[ToolDefinition, ...]):
            path = Path(context.workspace)
            names = {tool.name for tool in tools}
            assert "write_file" in names and "delegate" not in names
            if context.goal not in started:
                started.add(context.goal)
                seen[context.goal] = path
                if len(started) == 3:
                    waiting.set()
                await asyncio.wait_for(waiting.wait(), 3)
                return turn(
                    ToolCall("write", "write_file", {
                        "path": "shared.txt", "content": context.goal + "\n",
                    }),
                    ToolCall("test", "run_tests", {
                        "argv": [sys.executable, "-c", (
                            "from pathlib import Path; "
                            f"assert Path('shared.txt').read_text().strip() == {context.goal!r}; "
                            "Path('new.txt').write_text('new'); Path('remove.txt').unlink()"
                        )],
                    }),
                )
            return done()

    class Reader:
        def __init__(self) -> None:
            self.read = False

        async def complete(self, context: ModelContext, tools: tuple[ToolDefinition, ...]):
            names = {tool.name for tool in tools}
            assert "read_file" in names and not names & {"write_file", "edit_file", "run_shell"}
            if not self.read:
                self.read = True
                return turn(ToolCall("read", "read_file", {"path": "shared.txt"}))
            use_main = context.goal == "default"
            expected = "base" if use_main else "impl-1"
            assert expected in context.messages[-1].content[0]["content"]
            assert Path(context.workspace) == (case.repo if use_main else seen["impl-1"])
            return done()

    class Main:
        steps = 0

        async def complete(self, _context, _tools):
            self.steps += 1
            if self.steps == 1:
                return turn(*(delegate(f"d{i}", "implementer", f"impl-{i}") for i in range(3)))
            owner_id = case.children()[1].run_id
            if self.steps == 2:
                return turn(
                    delegate("default", "explorer", "default"),
                    delegate("default-review", "reviewer", "default"),
                    delegate("explore", "explorer", "inspect unmerged", target_run_id=owner_id),
                    delegate("review", "reviewer", "review unmerged", target_run_id=owner_id),
                )
            if self.steps == 3:
                return turn(ToolCall("diff", "inspect_changes", {"child_run_id": owner_id}))
            return done()

    await case.start(
        Main(), lambda spec: Implementer() if spec.spec_id == "implementer" else Reader(),
    )
    assert case.parent.status is RunStatus.COMPLETED
    assert len(set(seen.values())) == 3
    assert (case.repo / "shared.txt").read_text() == "base\n"
    assert case.session.workspace == case.repo
    assert set(case.approvals) == set(seen.values())
    for index, child in enumerate(case.children()[:3]):
        saved = case.store.load_worktree(child.run_id)
        assert child.status is RunStatus.COMPLETED
        assert saved is not None and saved.status is WorktreeStatus.SAVED
        assert (saved.path / "shared.txt").read_text() == f"impl-{index}\n"
        assert (saved.path / "new.txt").read_text() == "new"
        handoff = json.loads(case.results()[f"d{index}"]["content"])
        assert handoff["worktree"]["commit"] == saved.saved_commit
        assert handoff["tests"][0]["exit_code"] == 0
        assert {item["status"] for item in handoff["changes"]["files"]} == {"A", "D", "M"}
    outcomes = case.results()
    assert all(
        not outcomes[name]["is_error"]
        for name in ("default", "default-review", "explore", "review", "diff")
    )
    assert "+impl-1" in json.loads(json.loads(outcomes["diff"]["content"])["stdout"])["diff"]


@pytest.mark.parametrize("dirty", [False, True])
@run_async
async def test_rejects_four_implementers_or_dirty_main(tmp_path: Path, dirty: bool) -> None:
    case = Scenario(tmp_path)
    if dirty:
        (case.repo / "shared.txt").write_text("user edit\n")
        git(case.repo, "add", "shared.txt")
        (case.repo / "untracked.txt").write_text("user file\n")
    before = git(case.repo, "status", "--porcelain")
    count = 1 if dirty else 4
    await case.start(ScriptedModel([
        turn(*(delegate(f"d{i}", "implementer", f"task {i}") for i in range(count))), done(),
    ]))
    results = case.results()
    assert all(result["is_error"] for result in results.values())
    assert ("uncommitted" if dirty else "at most 3") in results["d0"]["content"]
    assert git(case.repo, "status", "--porcelain") == before
    assert git(case.repo, "rev-parse", "HEAD") == case.base
    assert not case.manager.root.exists()


@pytest.mark.parametrize(("conflict", "interrupt_at"), [
    (False, "before_apply"), (False, "legacy_before_apply"), (False, "before_ref_update"),
    (False, "after_ref_update"), (True, "after_ref_update"), (False, "main_advanced"),
])
@run_async
async def test_integrate_and_reconcile_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: bool, interrupt_at: str,
) -> None:
    case = Scenario(tmp_path)
    original_head = case.base

    class Implementer:
        async def complete(self, context: ModelContext, _tools):
            if context.messages[-1].content[0]["type"] == "tool_result":
                return done()
            file = "shared.txt" if conflict else f"{context.goal}.txt"
            return turn(ToolCall("write", "write_file", {
                "path": file, "content": context.goal + "\n",
            }))

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            nonlocal original_head
            self.step += 1
            if self.step == 1:
                return turn(
                    delegate("one", "implementer", "one"), delegate("two", "implementer", "two"),
                )
            if self.step == 2:
                if interrupt_at in {"before_apply", "legacy_before_apply"}:
                    (case.repo / "divergent.txt").write_text("unrelated main change")
                    git(case.repo, "add", ".")
                    git(case.repo, "commit", "-m", "Advance main")
                    original_head = git(case.repo, "rev-parse", "HEAD")
                one, two = case.children()
                return turn(
                    ToolCall("accept-one", "integrate_changes", {"child_run_id": one.run_id}),
                    ToolCall("accept-two", "integrate_changes", {"child_run_id": two.run_id}),
                    ToolCall("repeat", "integrate_changes", {"child_run_id": one.run_id}),
                )
            return done()

    save = case.store.save_worktree
    invoke_git = git_worktrees._git

    def interrupt(expected, updated):
        if interrupt_at == "after_ref_update" and updated.status is WorktreeStatus.INTEGRATED:
            raise KeyboardInterrupt
        save(expected, updated)
        if (
            interrupt_at in {"before_apply", "legacy_before_apply", "main_advanced"}
            and updated.status is WorktreeStatus.INTEGRATING
        ):
            raise KeyboardInterrupt

    def interrupt_ref_update(path, *args, **kwargs):
        if interrupt_at == "before_ref_update" and args[:2] == ("update-ref", "--no-deref"):
            raise KeyboardInterrupt
        return invoke_git(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(case.store, "save_worktree", interrupt)
        patch.setattr(git_worktrees, "_git", interrupt_ref_update)
        with pytest.raises(KeyboardInterrupt):
            await case.start(Main(), lambda _: Implementer())
    first, second = case.children()
    first_tree = case.store.load_worktree(first.run_id)
    assert first_tree is not None and first_tree.status is WorktreeStatus.INTEGRATING
    assert git(case.repo, "rev-parse", "HEAD") == (
        first_tree.integration_commit if interrupt_at == "after_ref_update" else original_head
    )
    if interrupt_at in {"before_apply", "legacy_before_apply"}:
        assert first_tree.integration_commit != first_tree.saved_commit
        if interrupt_at == "legacy_before_apply":
            pinned = git(case.repo, "for-each-ref", "--format=%(refname)",
                         "refs/keyline/integrations/")
            git(case.repo, "update-ref", "-d", pinned)
        git(case.repo, "gc", "--prune=now")
        if interrupt_at == "before_apply":
            assert git(case.repo, "cat-file", "-t", first_tree.integration_commit) == "commit"
        else:
            with pytest.raises(subprocess.CalledProcessError):
                git(case.repo, "cat-file", "-t", first_tree.integration_commit)
    if interrupt_at == "before_ref_update":
        assert git(case.repo, "diff", "--cached", "--name-only") == "one.txt"
    if interrupt_at == "main_advanced":
        (case.repo / "user.txt").write_text("committed after interruption")
        git(case.repo, "add", "user.txt")
        git(case.repo, "commit", "-m", "User advances main before resume")

    store = SQLiteStateStore(case.store.database_path)
    resumed = store.load_run(case.parent.run_id)
    assert resumed.tool_runs[0].status is ToolRunStatus.RUNNING
    await resume_run(
        store.load_session(case.session.session_id), resumed, ScriptedModel([done()]),
        case.registry, store=store, agent_spec=case.main, delegation=case.runtime,
    )
    assert resumed.status is RunStatus.COMPLETED
    assert case.results()["accept-two"]["is_error"] is conflict
    assert not case.results()["accept-one"]["is_error"]
    assert not case.results()["repeat"]["is_error"]
    assert git(case.repo, "status", "--porcelain") == ""
    assert git(case.repo, "for-each-ref", "refs/keyline/integrations/") == ""
    assert not (case.repo / ".git" / "MERGE_HEAD").exists()
    second_tree = store.load_worktree(second.run_id)
    assert second_tree is not None
    if conflict:
        assert "merge conflict" in case.results()["accept-two"]["content"]
        assert git(case.repo, "rev-parse", "HEAD") == first_tree.integration_commit
        assert (case.repo / "shared.txt").read_text() == "one\n"
        assert (second_tree.path / "shared.txt").read_text() == "two\n"
        assert second_tree.status is WorktreeStatus.SAVED
    else:
        assert (case.repo / "one.txt").read_text() == "one\n"
        assert (case.repo / "two.txt").read_text() == "two\n"
        assert second_tree.status is WorktreeStatus.INTEGRATED
        assert second_tree.cleaned and not second_tree.path.exists()
    if interrupt_at == "main_advanced":
        assert (case.repo / "user.txt").read_text() == "committed after interruption"
        restored = store.load_worktree(first.run_id)
        assert restored is not None and restored.integration_commit != first_tree.integration_commit


@pytest.mark.parametrize("creation_boundary", ["before_git", "after_git"])
@run_async
async def test_provision_resumes_original_child_and_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, creation_boundary: str,
) -> None:
    case = Scenario(tmp_path)
    provision = case.manager.provision
    save = case.store.save_worktree

    def interrupt_provision(tree, store):
        if creation_boundary == "before_git":
            raise KeyboardInterrupt
        return provision(tree, store)

    def interrupt_save(expected, updated):
        if updated.status is WorktreeStatus.ACTIVE:
            raise KeyboardInterrupt
        save(expected, updated)

    with monkeypatch.context() as patch:
        patch.setattr(case.manager, "provision", interrupt_provision)
        patch.setattr(case.store, "save_worktree", interrupt_save)
        with pytest.raises(KeyboardInterrupt):
            await case.start(ScriptedModel([turn(delegate("impl", "implementer", "task"))]))

    (child,) = case.children()
    planned = case.store.load_worktree(child.run_id)
    assert planned is not None and planned.status is WorktreeStatus.PLANNED
    assert planned.path.exists() is (creation_boundary == "after_git")
    initial_registration = git(case.repo, "worktree", "list", "--porcelain")
    result = await case.restart_batch(ScriptedModel([done()]), lambda _: ScriptedModel([done()]))
    assert result.status is RunStatus.COMPLETED
    assert [run.run_id for run in case.children()] == [child.run_id]
    saved = case.store.load_worktree(child.run_id)
    assert saved is not None and saved.status is WorktreeStatus.SAVED
    assert (saved.path, saved.branch, saved.base_commit) == (
        planned.path, planned.branch, planned.base_commit,
    )
    registered = git(case.repo, "worktree", "list", "--porcelain")
    assert registered.count(f"worktree {saved.path}") == 1
    if creation_boundary == "after_git":
        assert registered == initial_registration


@run_async
async def test_resume_with_one_saved_child_and_another_with_partial_edits(tmp_path: Path) -> None:
    case = Scenario(tmp_path)

    class Child:
        async def complete(self, context: ModelContext, _tools):
            if context.messages[-1].content[0]["type"] != "tool_result":
                return turn(ToolCall("write", "write_file", {
                    "path": "shared.txt", "content": context.goal,
                }))
            if context.goal == "first":
                return done()
            for _ in range(100):
                first = case.store.load_worktree(case.children()[0].run_id)
                if first is not None and first.status is WorktreeStatus.SAVED:
                    raise KeyboardInterrupt
                await asyncio.sleep(0.02)
            raise AssertionError("First child did not finish")

    with pytest.raises(KeyboardInterrupt):
        await case.start(ScriptedModel([turn(
            delegate("first", "implementer", "first"), delegate("second", "implementer", "second"),
        )]), lambda _: Child())
    first, second = case.children()
    first_tree = case.store.load_worktree(first.run_id)
    second_tree = case.store.load_worktree(second.run_id)
    assert first.status is RunStatus.COMPLETED
    assert first_tree is not None and first_tree.status is WorktreeStatus.SAVED
    assert second.status is RunStatus.RUNNING
    assert second_tree is not None and second_tree.status is WorktreeStatus.ACTIVE
    assert (second_tree.path / "shared.txt").read_text() == "second"

    model = ScriptedModel([done()])
    result = await case.restart_batch(ScriptedModel([done()]), lambda _: model)
    assert result.status is RunStatus.COMPLETED
    assert [run.run_id for run in case.children()] == [first.run_id, second.run_id]
    assert len(model.contexts) == 1
    assert Path(model.contexts[0].workspace) == second_tree.path
    resumed = case.store.load_worktree(second.run_id)
    assert resumed is not None and resumed.status is WorktreeStatus.SAVED
    assert resumed.path == second_tree.path and resumed.branch == second_tree.branch
    assert (first_tree.path / "shared.txt").read_text() == "first"
    assert (resumed.path / "shared.txt").read_text() == "second"
    assert case.store.load_worktree(first.run_id) == first_tree


@run_async
async def test_target_validation_and_cleanup(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    (tmp_path / "other").mkdir()
    foreign = Scenario(tmp_path / "other")
    foreign.store = case.store
    foreign.store.create_session(foreign.session)
    await foreign.start(
        ScriptedModel([turn(delegate("impl", "implementer", "foreign")), done()]),
        lambda _: ScriptedModel([done()]),
    )
    (other_child,) = foreign.children()

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(
                    delegate("own", "implementer", "write"), delegate("plain", "explorer", "read"),
                )
            owner, plain, *_ = case.children()
            if self.step == 2:
                return turn(
                    delegate("foreign", "reviewer", "foreign", target_run_id=other_child.run_id),
                    delegate("unknown", "reviewer", "unknown", target_run_id="missing-run"),
                    delegate("abs-path", "explorer", "arbitrary", target_run_id=str(case.repo)),
                    delegate("plain-target", "explorer", "root", target_run_id=plain.run_id),
                )
            if self.step == 3:
                return turn(delegate("reuse", "implementer", "reuse", target_run_id=owner.run_id))
            if self.step == 4:
                return turn(
                    ToolCall("unsafe-clean", "cleanup_worktree", {"child_run_id": owner.run_id}),
                    ToolCall("abandon", "cleanup_worktree", {
                        "child_run_id": owner.run_id, "abandon": True,
                    }),
                    ToolCall("repeat-clean", "cleanup_worktree", {"child_run_id": owner.run_id}),
                )
            if self.step == 5:
                return turn(delegate("cleaned", "reviewer", "cleaned", target_run_id=owner.run_id))
            return done()

    await case.start(Main(), lambda spec: ScriptedModel(
        [turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "saved"})), done()]
        if spec.spec_id == "implementer" else [done()]
    ))
    results = case.results()
    errors = {
        "foreign": "does not belong to this Session",
        "unknown": "does not exist",
        "abs-path": "does not exist",
        "plain-target": "does not own a Worktree",
        "reuse": "Only explorer and reviewer",
        "unsafe-clean": "explicitly abandon",
        "cleaned": "was cleaned",
    }
    for call, error in errors.items():
        assert results[call]["is_error"] and error in results[call]["content"]
    assert not results["abandon"]["is_error"] and not results["repeat-clean"]["is_error"]
    worktree = case.store.load_worktree(case.children()[0].run_id)
    assert worktree is not None and worktree.cleaned and not worktree.path.exists()
    assert worktree.status is WorktreeStatus.ABANDONED
    assert git(case.repo, "show", f"{worktree.branch}:new.txt") == "saved"


@run_async
async def test_reviewer_resumes_with_original_worktree_binding(tmp_path: Path) -> None:
    case = Scenario(tmp_path)

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("own", "implementer", "write"))
            (owner,) = case.children()
            return turn(delegate("review", "reviewer", "read", target_run_id=owner.run_id))

    class Reviewer:
        async def complete(self, _context, _tools):
            raise KeyboardInterrupt

    awaitable = case.start(Main(), lambda spec: (
        Reviewer() if spec.spec_id == "reviewer" else ScriptedModel([
            turn(ToolCall("write", "write_file", {"path": "shared.txt", "content": "unmerged"})),
            done(),
        ])
    ))
    with pytest.raises(KeyboardInterrupt):
        await awaitable
    owner, reviewer = case.children()
    tree = case.store.load_worktree(owner.run_id)
    assert tree is not None and reviewer.workspace_binding.worktree_owner_id == owner.run_id
    model = ScriptedModel([turn(ToolCall("read", "read_file", {"path": "shared.txt"})), done()])
    await case.restart_batch(ScriptedModel([done()]), lambda _: model)
    assert case.children()[1].run_id == reviewer.run_id
    assert Path(model.contexts[0].workspace) == tree.path
    assert "unmerged" in model.contexts[-1].messages[-1].content[0]["content"]
    assert not case.results()["review"]["is_error"]


@pytest.mark.parametrize("cancel", [False, True])
@run_async
async def test_failed_or_cancelled_child_keeps_changes(tmp_path: Path, cancel: bool) -> None:
    case = Scenario(tmp_path)

    class Child:
        async def complete(self, context, _tools):
            if context.messages[-1].content[0]["type"] != "tool_result":
                return turn(ToolCall("write", "write_file", {
                    "path": "shared.txt", "content": "partial changes",
                }))
            if cancel:
                raise UserCancelledError("Cancelled during implementation")
            return await ScriptedModel([]).complete(context, ())

    main = ScriptedModel([turn(delegate("own", "implementer", "write")), done()])
    if cancel:
        with pytest.raises(UserCancelledError):
            await case.start(main, lambda _: Child())
    else:
        await case.start(main, lambda _: Child())
    (child,) = case.children()
    assert child.status is (RunStatus.CANCELLED if cancel else RunStatus.FAILED)
    tree = case.store.load_worktree(child.run_id)
    assert tree is not None and tree.status is WorktreeStatus.ACTIVE
    assert (tree.path / "shared.txt").read_text() == "partial changes"
    assert (case.repo / "shared.txt").read_text() == "base\n"
    assert git(case.repo, "rev-parse", "HEAD") == case.base


@run_async
async def test_preparation_and_secret_process_use_bound_workspace(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    secret_store = EphemeralSecretStore()
    secret = "worktree-test-canary"
    approved: list[Path] = []

    async def approve(cwd: Path, _argv: tuple[str, ...]) -> bool:
        approved.append(cwd)
        return True

    async def answer(child: Run):
        dependency, = child.open_dependencies()
        return resolve_secret_dependency(
            case.session, child, dependency.dependency_id, secret, secret_store.writer(),
        )

    case.runtime.resolve_input = answer

    class Child:
        step = 0

        async def complete(self, context: ModelContext, _tools):
            self.step += 1
            if self.step == 1:
                return turn(ToolCall("wait", "wait_for_secret", {
                    "prompt": "Enter a test Secret", "purpose": "Verify bound workspace",
                    "environment_variable": "TEST_API_KEY",
                }))
            if self.step == 2:
                child, = case.children()
                return turn(ToolCall("secure", "run_command_with_secret", {
                    "secret_handle": child.secret_handles[0].secret_handle,
                    "argv": [sys.executable, "-c", (
                        "import os; from pathlib import Path; "
                        "assert os.environ['TEST_API_KEY']; print(Path.cwd())"
                    )],
                }))
            assert context.workspace in context.messages[-1].content[0]["content"]
            return done()

    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "secure test")), done()]),
        lambda _: Child(), secrets=SecretCommandRunner(
            secret_store.consumer(), lambda *_: False, approve,
        ),
    )
    child, = case.children()
    tree = case.store.load_worktree(child.run_id)
    assert tree is not None
    assert child.status is RunStatus.COMPLETED
    assert approved == [tree.path]
    result = child.tool_runs[-1].result
    assert result is not None and result.ok and result.stdout.strip() == str(tree.path)
    assert secret.encode() not in case.store.database_path.read_bytes()


@run_async
async def test_missing_checkout_is_reported_and_preserved(tmp_path: Path) -> None:
    case = Scenario(tmp_path)

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "write"))
            if self.step == 2:
                child, = case.children()
                tree = case.store.load_worktree(child.run_id)
                assert tree is not None
                tree.path.rename(tmp_path / "moved")
                return turn(delegate("read", "reviewer", "inspect", target_run_id=child.run_id))
            return done()

    await case.start(Main(), lambda _: ScriptedModel([
        turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "kept"})), done(),
    ]))
    result = case.results()["read"]
    assert result["is_error"] and "missing or moved" in result["content"]
    assert (tmp_path / "moved" / "new.txt").read_text() == "kept"


@pytest.mark.parametrize("dirty", [True, False], ids=["tracked", "ignored"])
@run_async
async def test_integration_preserves_user_files_added_after_delegation(
    tmp_path: Path, dirty: bool,
) -> None:
    case = Scenario(tmp_path)
    file = "shared.txt" if dirty else "secret-local.txt"

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "write"))
            if self.step == 2:
                if not dirty:
                    (case.repo / ".git" / "info" / "exclude").write_text(file + "\n")
                (case.repo / file).write_text("local user changes")
                child, = case.children()
                return turn(ToolCall("merge", "integrate_changes", {"child_run_id": child.run_id}))
            return done()

    await case.start(Main(), lambda _: ScriptedModel([
        turn(ToolCall("write", "write_file", {"path": file, "content": "child changes"})), done(),
    ]))
    assert case.results()["merge"]["is_error"]
    assert (case.repo / file).read_text() == "local user changes"
    assert git(case.repo, "rev-parse", "HEAD") == case.base
    assert not (case.repo / ".git" / "MERGE_HEAD").exists()
    child, = case.children()
    tree = case.store.load_worktree(child.run_id)
    assert tree is not None and (tree.path / file).read_text() == "child changes"
    assert tree.status is WorktreeStatus.SAVED


@run_async
async def test_schema_migrates_existing_root_run_and_transcript(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    await case.start(ScriptedModel([done()]))
    with sqlite3.connect(case.store.database_path) as connection:
        connection.execute("DROP TABLE worktrees")
        connection.execute("ALTER TABLE runs DROP COLUMN workspace_owner_id")
        connection.execute("PRAGMA user_version = 15")
    store = SQLiteStateStore(case.store.database_path)
    existing = store.load_run(case.parent.run_id)
    assert existing.workspace_binding.worktree_owner_id is None
    assert existing.workspace(store.load_session(existing.session_id)) == case.repo
    assert existing.status is RunStatus.COMPLETED
    assert store.message_count(existing.context_scope_id) == 2
    assert store.load_worktree(existing.run_id) is None


@run_async
async def test_interrupted_side_effect_is_not_replayed_in_worktree(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    invoked: list[Path] = []

    def side_effect(workspace: Path, _arguments: object):
        invoked.append(workspace)
        with (workspace / "effects.txt").open("a") as file:
            file.write("wrote once\n")
        raise KeyboardInterrupt

    case.registry.register(ToolBinding(
        ToolDefinition("side_effect", "Write and then interrupt", {"type": "object"}),
        side_effect,
    ))
    spec = case.specs["implementer"]
    case.specs["implementer"] = replace(spec, allowed_tools=spec.allowed_tools | {"side_effect"})

    with pytest.raises(KeyboardInterrupt):
        await case.start(
            ScriptedModel([turn(delegate("impl", "implementer", "write"))]),
            lambda _: ScriptedModel([turn(ToolCall("effect", "side_effect", {}))]),
        )
    child, = case.children()
    assert child.tool_runs[0].status is ToolRunStatus.RUNNING

    result = await case.restart_batch(ScriptedModel([done()]), lambda _: ScriptedModel([]))
    assert result.status is RunStatus.COMPLETED
    restored, = case.children()
    assert restored.status is RunStatus.FAILED
    assert restored.tool_runs[0].status is ToolRunStatus.UNKNOWN
    assert "not retried" in (restored.error or "")
    tree = case.store.load_worktree(restored.run_id)
    assert tree is not None and invoked == [tree.path]
    assert (tree.path / "effects.txt").read_text() == "wrote once\n"


@pytest.mark.parametrize("other_state", ["same_head", "different_head", "additional_staged_edit"])
@run_async
async def test_switch_during_integration_only_reverses_confirmed_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_state: str,
) -> None:
    case = Scenario(tmp_path)
    git(case.repo, "branch", "other")
    if other_state != "same_head":
        git(case.repo, "switch", "other")
        (case.repo / "other.txt").write_text("other branch")
        git(case.repo, "add", "other.txt")
        git(case.repo, "commit", "-m", "Commit unique to other branch")
        git(case.repo, "switch", "main")
    other_head = git(case.repo, "rev-parse", "other")
    invoke_git = git_worktrees._git

    def switch_before_apply(path, *args, **kwargs):
        if args[:2] == ("apply", "--index"):
            git(case.repo, "switch", "other")
            result = invoke_git(path, *args, **kwargs)
            if other_state == "additional_staged_edit":
                (case.repo / "user.txt").write_text("user's staged content")
                git(case.repo, "add", "user.txt")
            return result
        return invoke_git(path, *args, **kwargs)

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "write"))
            if self.step == 2:
                child, = case.children()
                return turn(ToolCall("merge", "integrate_changes", {"child_run_id": child.run_id}))
            return done()

    monkeypatch.setattr(git_worktrees, "_git", switch_before_apply)
    await case.start(Main(), lambda _: ScriptedModel([
        turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "child"})), done(),
    ]))

    result = case.results()["merge"]
    assert result["is_error"]
    assert git(case.repo, "branch", "--show-current") == "other"
    assert git(case.repo, "rev-parse", "main") == case.base
    assert git(case.repo, "rev-parse", "other") == other_head
    (child,) = case.children()
    tree = case.store.load_worktree(child.run_id)
    assert tree is not None
    assert (tree.path / "new.txt").read_text() == "child"
    if other_state == "additional_staged_edit":
        assert "Inspect the staging area" in result["content"]
        assert tree.status is WorktreeStatus.INTEGRATING
        assert git(case.repo, "diff", "--cached", "--name-only").splitlines() == [
            "new.txt", "user.txt",
        ]
        assert (case.repo / "user.txt").read_text() == "user's staged content"
    else:
        assert tree.status is WorktreeStatus.SAVED
        assert git(case.repo, "status", "--porcelain") == ""


@run_async
async def test_embedded_git_checkout_is_not_saved_as_a_gitlink(tmp_path: Path) -> None:
    case = Scenario(tmp_path)

    class Child:
        async def complete(self, context: ModelContext, _tools):
            nested = Path(context.workspace) / "nested"
            nested.mkdir()
            git(nested, "init", "-b", "main")
            (nested / "source.txt").write_text("independent repository content")
            git(nested, "add", ".")
            git(nested, "-c", "user.name=Test", "-c", "user.email=test@example.test",
                "commit", "-m", "Nested changes")
            return done()

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "create files"))
            if self.step == 2:
                child, = case.children()
                return turn(ToolCall("inspect", "inspect_changes", {"child_run_id": child.run_id}))
            return done()

    await case.start(Main(), lambda _: Child())
    child, = case.children()
    tree = case.store.load_worktree(child.run_id)
    assert child.status is RunStatus.COMPLETED
    assert tree is not None and tree.status is WorktreeStatus.ACTIVE
    assert tree.saved_commit is None and git(case.repo, "rev-parse", tree.branch) == case.base
    assert (tree.path / "nested" / "source.txt").read_text() == "independent repository content"
    inspected = case.results()["inspect"]
    assert inspected["is_error"] and "embedded Git repositories as files: 'nested'" in (
        inspected["content"]
    )


@run_async
async def test_worktree_tests_use_its_own_editable_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = Scenario(tmp_path)
    module = "worktree_isolation_example"
    source = case.repo / "src" / f"{module}.py"
    source.parent.mkdir()
    source.write_text("VALUE = 'main'\n")
    (case.repo / ".gitignore").write_text(".venv/\n")
    git(case.repo, "add", ".")
    git(case.repo, "commit", "-m", "Add src-layout module")
    original_venv = case.repo / ".venv"

    def make_editable_environment(path: Path) -> Path:
        venv = path / ".venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
        site_packages, = venv.glob("lib/python*/site-packages")
        (site_packages / "example.pth").write_text(f"{path / 'src'}\n")
        return venv / "bin" / "python"

    original_python = make_editable_environment(case.repo)
    monkeypatch.setenv("VIRTUAL_ENV", str(original_venv))
    monkeypatch.setenv("PYTHONPATH", str(source.parent))
    monkeypatch.setenv("PATH", os.pathsep.join((str(original_python.parent), os.defpath)))

    class Child:
        step = 0

        async def complete(self, context: ModelContext, _tools):
            self.step += 1
            workspace = Path(context.workspace)
            if self.step == 1:
                return turn(ToolCall("write", "write_file", {
                    "path": f"src/{module}.py", "content": "VALUE = 'child'\n",
                }))
            if self.step == 2:
                local_python = make_editable_environment(workspace)
                return turn(
                    ToolCall("main-python", "run_tests", {
                        "argv": [str(original_python), "-c", f"import {module}"],
                    }),
                    ToolCall("child-python", "run_tests", {
                        "argv": [str(local_python), "-c", (
                            "import os; from pathlib import Path; "
                            f"import {module} as tested; "
                            "assert tested.VALUE == 'child'; "
                            "assert Path(tested.__file__).is_relative_to(Path.cwd()); "
                            "assert 'VIRTUAL_ENV' not in os.environ; "
                            f"assert {str(original_python.parent)!r} not in os.environ['PATH']; "
                            "print(tested.VALUE)"
                        )],
                    }),
                )
            return done()

    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "validate child source")), done()]),
        lambda _: Child(),
    )

    child, = case.children()
    assert child.status is RunStatus.COMPLETED
    assert [tool.status for tool in child.tool_runs] == [
        ToolRunStatus.SUCCEEDED, ToolRunStatus.FAILED, ToolRunStatus.SUCCEEDED,
    ]
    original_test, child_test = child.tool_runs[1:]
    assert original_test.result is not None
    assert "Executable belongs to the original workspace" in original_test.result.stderr
    assert child_test.result is not None and child_test.result.stdout.strip() == "child"
    assert source.read_text() == "VALUE = 'main'\n"


@run_async
async def test_initialized_submodule_update_is_rejected_before_staging(tmp_path: Path) -> None:
    case = Scenario(tmp_path)
    module = tmp_path / "dependency"
    module.mkdir()
    git(module, "init", "-b", "main")
    git(module, "config", "user.name", "Test")
    git(module, "config", "user.email", "test@example.test")
    (module / "lib.txt").write_text("before")
    git(module, "add", "lib.txt")
    git(module, "commit", "-m", "First version")
    old_module_head = git(module, "rev-parse", "HEAD")
    git(case.repo, "-c", "protocol.file.allow=always", "submodule", "add", str(module), "dep")
    git(case.repo, "commit", "-m", "Add initialized submodule")
    original_head = git(case.repo, "rev-parse", "HEAD")
    (module / "lib.txt").write_text("after")
    git(module, "commit", "-am", "Second version")
    new_module_head = git(module, "rev-parse", "HEAD")

    class Child:
        async def complete(self, context: ModelContext, _tools):
            git(Path(context.workspace), "update-index", "--cacheinfo",
                f"160000,{new_module_head},dep")
            return done()

    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "bump dependency")), done()]),
        lambda _: Child(),
    )
    (child,) = case.children()
    saved = case.store.load_worktree(child.run_id)
    assert saved is not None and saved.status is WorktreeStatus.SAVED
    with pytest.raises(WorktreeError, match="initialized submodule: dep"):
        case.manager.integrate(saved, case.store)
    retained = case.store.load_worktree(child.run_id)
    assert retained == saved
    assert git(case.repo, "status", "--porcelain") == ""
    assert git(case.repo, "rev-parse", "HEAD") == original_head
    assert git(case.repo / "dep", "rev-parse", "HEAD") == old_module_head


@pytest.mark.parametrize("already_applied", [False, True])
@run_async
async def test_abandon_after_interruption_and_main_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_applied: bool,
) -> None:
    case = Scenario(tmp_path)
    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "create file")), done()]),
        lambda _: ScriptedModel([
            turn(ToolCall("write", "write_file", {"path": "child.txt", "content": "kept"})), done(),
        ]),
    )
    (child,) = case.children()
    saved = case.store.load_worktree(child.run_id)
    assert saved is not None and saved.status is WorktreeStatus.SAVED
    with monkeypatch.context() as patched:
        boundary = "_finish_integration" if already_applied else "_apply_integration"

        def interrupt(*_args):
            raise KeyboardInterrupt

        patched.setattr(case.manager, boundary, interrupt)
        with pytest.raises(KeyboardInterrupt):
            case.manager.integrate(saved, case.store)
    store = SQLiteStateStore(case.store.database_path)
    pending = store.load_worktree(child.run_id)
    assert pending is not None and pending.status is WorktreeStatus.INTEGRATING
    (case.repo / "user.txt").write_text("done by user")
    git(case.repo, "add", "user.txt")
    git(case.repo, "commit", "-m", "Advance main after interruption")
    user_head = git(case.repo, "rev-parse", "HEAD")

    # An unresolved change in the main checkout prevents intent cleanup.
    (case.repo / "user.txt").write_text("new local edit")
    with pytest.raises(WorktreeError, match="uncommitted"):
        case.manager.cleanup(pending, store, abandon=True)
    assert store.load_worktree(child.run_id) == pending
    assert (pending.path / "child.txt").read_text() == "kept"
    assert (case.repo / "user.txt").read_text() == "new local edit"
    (case.repo / "user.txt").write_text("done by user")

    cleaned = case.manager.cleanup(pending, store, abandon=True)
    assert cleaned.status is (WorktreeStatus.INTEGRATED if already_applied
                              else WorktreeStatus.ABANDONED)
    assert cleaned.cleaned and not cleaned.path.exists()
    assert git(case.repo, "show", f"{cleaned.branch}:child.txt") == "kept"
    assert (case.repo / "child.txt").exists() is already_applied
    assert git(case.repo, "rev-parse", "HEAD") == user_head
    assert git(case.repo, "status", "--porcelain") == ""
    assert git(case.repo, "for-each-ref", "refs/keyline/integrations/") == ""


@pytest.mark.parametrize("abandon", [True, False], ids=["abandon", "integrate"])
@run_async
async def test_finalized_worktree_reclaims_arbitrary_ignored_data(
    tmp_path: Path, abandon: bool,
) -> None:
    case = Scenario(tmp_path)
    (case.repo / ".git" / "info" / "exclude").write_text("local-output/\n.env\n")
    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "create and test code")), done()]),
        lambda _: ScriptedModel([
            turn(
                ToolCall("write", "write_file", {"path": "source.txt", "content": "saved code"}),
                ToolCall("test", "run_tests", {"argv": [sys.executable, "-c", (
                    "from pathlib import Path; "
                    "Path('local-output').mkdir(); Path('local-output/result').touch(); "
                    "Path('.env').write_text('discarded test data')"
                )]}),
            ),
            done(),
        ]),
    )
    (child,) = case.children()
    tree = case.store.load_worktree(child.run_id)
    assert tree is not None and tree.status is WorktreeStatus.SAVED and tree.path.is_dir()
    assert (tree.path / ".env").exists()
    link = tree.path / "local-output" / "link-to-original"
    link.symlink_to(case.repo, target_is_directory=True)
    arguments = {"child_run_id": child.run_id}
    operation = "integrate_changes"
    if abandon:
        arguments["abandon"] = True
        operation = "cleanup_worktree"

    notices: list[str] = []
    result = execute_worktree_operation(
        case.session, case.parent, ToolCall("finalize", operation, arguments),
        case.store, case.manager, notices.append,
    )
    assert result.ok, result.stderr
    assert json.loads(result.stdout)["cleanup_status"] == "cleaned"
    assert any("including ignored files" in notice for notice in notices)
    cleaned = case.store.load_worktree(child.run_id)
    assert cleaned is not None
    assert cleaned.status is (WorktreeStatus.ABANDONED if abandon else WorktreeStatus.INTEGRATED)
    assert cleaned.cleaned and not cleaned.path.exists()
    assert git(case.repo, "show", f"{cleaned.branch}:source.txt") == "saved code"
    inspected = execute_worktree_operation(
        case.session, case.parent,
        ToolCall("inspect", "inspect_changes", {"child_run_id": child.run_id}),
        case.store, case.manager,
    )
    assert inspected.ok and "+saved code" in json.loads(inspected.stdout)["diff"]
    assert (case.repo / "source.txt").exists() is not abandon
    assert (case.repo / "shared.txt").read_text() == "base\n"


@pytest.mark.parametrize("abandon", [False, True], ids=["integrate", "abandon"])
@pytest.mark.parametrize("paused", [False, True], ids=["interrupted", "budget-paused"])
@run_async
async def test_cleanup_waits_for_bound_reviewer_then_runs_automatically(
    tmp_path: Path, abandon: bool, paused: bool,
) -> None:
    case = Scenario(tmp_path)

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "write"))
            (owner,) = case.children()
            return turn(delegate("review", "reviewer", "read", target_run_id=owner.run_id))

    class InterruptedReviewer:
        async def complete(self, _context, _tools):
            raise KeyboardInterrupt

    read = turn(ToolCall("read", "read_file", {"path": "new.txt"}))

    def children(spec):
        if spec.spec_id == "reviewer":
            return ScriptedModel([read]) if paused else InterruptedReviewer()
        return ScriptedModel([
            turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "kept"})),
            done(),
        ])

    if paused:
        case.specs["reviewer"] = replace(case.specs["reviewer"], default_max_steps=1)
        await case.start(Main(), children)
    else:
        with pytest.raises(KeyboardInterrupt):
            await case.start(Main(), children)
    owner, reviewer = case.children()
    assert reviewer.status is (RunStatus.PAUSED if paused else RunStatus.RUNNING)
    assert "review" not in case.results()
    assert reviewer.workspace_binding.worktree_owner_id == owner.run_id

    call = (ToolCall("discard", "cleanup_worktree", {
        "child_run_id": owner.run_id, "abandon": True,
    }) if abandon else ToolCall("merge", "integrate_changes", {"child_run_id": owner.run_id}))
    result = execute_worktree_operation(case.session, case.parent, call, case.store, case.manager)
    assert result.ok, result.stderr
    assert json.loads(result.stdout)["cleanup_status"] == "deferred"
    deferred = case.store.load_worktree(owner.run_id)
    expected_status = WorktreeStatus.ABANDONED if abandon else WorktreeStatus.INTEGRATED
    assert deferred is not None and deferred.status is expected_status
    assert deferred.path.is_dir() and not deferred.cleanup_pending

    parent = case.store.load_run(case.parent.run_id)
    late = create_child_run(
        case.session, case.store.list_runs(case.session.session_id), parent, "late reviewer",
        agent_spec_id="reviewer", model_name="scripted", max_steps=5, tool_call_id="late",
    )
    late.workspace_binding = WorkspaceBinding(owner.run_id)
    with pytest.raises(WorktreeUnavailableError, match="closed to new bindings"):
        case.store.create_run(case.session, late)

    if paused:
        blocked = await case.restart_batch(ScriptedModel([]), lambda _: pytest.fail("Paused child"))
        assert blocked.status is RunStatus.WAITING and deferred.path.is_dir()
        update_paused_step_limit(case.session, reviewer, 2, store=case.store)
    reader = ScriptedModel([done()] if paused else [read, done()])
    parent_model = ScriptedModel([done()])
    resumed = await case.restart_batch(parent_model, lambda _: reader)
    assert resumed.status is RunStatus.COMPLETED
    assert Path(reader.contexts[0].workspace) == deferred.path
    assert "kept" in reader.contexts[-1].messages[-1].content[0]["content"]
    assert [b["tool_use_id"] for b in parent_model.contexts[0].messages[-1].content] == ["review"]
    handoff = json.loads(case.results()["review"]["content"])
    assert handoff["run_id"] == reviewer.run_id
    saved_reviewer = case.store.load_run(reviewer.run_id)
    assert saved_reviewer.status is RunStatus.COMPLETED and saved_reviewer.model_turns == 2
    assert [tool.tool_call_id for tool in saved_reviewer.tool_runs] == ["read"]
    cleaned = case.store.load_worktree(owner.run_id)
    assert cleaned is not None and cleaned.cleaned and not cleaned.path.exists()
    assert cleaned.status is expected_status


@pytest.mark.parametrize("interrupt_at", ["before_cleanup", "after_remove", "cleanup_failure"])
@run_async
async def test_session_resume_recovers_cleanup_without_reapplying_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_at: str,
) -> None:
    case = Scenario(tmp_path)

    class Main:
        step = 0

        async def complete(self, _context, _tools):
            self.step += 1
            if self.step == 1:
                return turn(delegate("impl", "implementer", "write"))
            if self.step == 2:
                child, = case.children()
                return turn(ToolCall("merge", "integrate_changes", {"child_run_id": child.run_id}))
            return done()

    save = case.store.save_worktree

    def stop_before_cleanup(*_args, **_kwargs):
        if interrupt_at == "cleanup_failure":
            raise WorktreeError("simulated remove failure")
        raise KeyboardInterrupt

    def stop_after_remove(expected, updated):
        if updated.cleaned:
            raise KeyboardInterrupt
        save(expected, updated)

    with monkeypatch.context() as patch:
        if interrupt_at == "after_remove":
            patch.setattr(case.store, "save_worktree", stop_after_remove)
        else:
            patch.setattr(case.manager, "cleanup", stop_before_cleanup)
        operation = case.start(Main(), lambda _: ScriptedModel([
            turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "kept"})), done(),
        ]))
        if interrupt_at == "cleanup_failure":
            await operation
            merge = case.results()["merge"]
            assert not merge["is_error"]
            reported = json.loads(json.loads(merge["content"])["stdout"])
            assert reported["worktree_status"] == "integrated"
            assert reported["cleanup_status"] == "failed"
            assert reported["cleanup_error"] == "simulated remove failure"
        else:
            with pytest.raises(KeyboardInterrupt):
                await operation

    child, = case.children()
    pending = case.store.load_worktree(child.run_id)
    assert pending is not None and pending.status is WorktreeStatus.INTEGRATED
    assert pending.cleanup_pending is (interrupt_at == "after_remove")
    assert pending.path.exists() is (interrupt_at != "after_remove")
    integrated_head = git(case.repo, "rev-parse", "HEAD")
    reopened = SQLiteStateStore(case.store.database_path)
    monkeypatch.setattr(cli, "_state_store", lambda: reopened)
    monkeypatch.setattr(cli, "_create_real_model", lambda _model: ScriptedModel([done()]))
    args = Namespace(session_id=case.session.session_id)
    assert await cli._resume_session(args) == 0
    assert await cli._resume_session(args) == 0
    recovered = reopened.load_worktree(child.run_id)
    assert recovered is not None and recovered.cleaned and not recovered.path.exists()
    assert not recovered.cleanup_pending
    assert git(case.repo, "rev-parse", "HEAD") == integrated_head
    assert git(case.repo, "show", f"{recovered.branch}:new.txt") == "kept"


@pytest.mark.parametrize(
    "alteration", ["dirty", "assume_unchanged", "replacement", "symlink", "legacy_record"],
)
@run_async
async def test_cleanup_refuses_unsaved_or_unverified_checkouts(
    tmp_path: Path, alteration: str,
) -> None:
    case = Scenario(tmp_path)
    await case.start(
        ScriptedModel([turn(delegate("impl", "implementer", "write")), done()]),
        lambda _: ScriptedModel([
            turn(ToolCall("write", "write_file", {"path": "new.txt", "content": "saved"})), done(),
        ]),
    )
    child, = case.children()
    saved = case.store.load_worktree(child.run_id)
    assert saved is not None
    integrated = case.manager.integrate(saved, case.store)
    if alteration in {"dirty", "assume_unchanged"}:
        if alteration == "assume_unchanged":
            git(integrated.path, "update-index", "--assume-unchanged", "new.txt")
        (integrated.path / "new.txt").write_text("unsaved")
        expected = "uncommitted" if alteration == "dirty" else "index hides tracked changes"
    elif alteration == "legacy_record":
        with sqlite3.connect(case.store.database_path) as connection:
            connection.execute(
                "UPDATE worktrees "
                "SET record_json = json_remove(record_json, '$.directory_identity') "
                "WHERE owner_run_id = ?",
                (child.run_id,),
            )
        integrated = replace(integrated, directory_identity=None)
        expected = "no recorded directory identity"
    else:
        moved = tmp_path / "moved-worktree"
        integrated.path.rename(moved)
        if alteration == "replacement":
            # Even a byte-for-byte copy with the original Git link is a new directory.
            shutil.copytree(moved, integrated.path, symlinks=True)
            expected = "directory was replaced"
        else:
            integrated.path.symlink_to(moved, target_is_directory=True)
            expected = "missing or moved"

    with pytest.raises(WorktreeError, match=expected):
        case.manager.cleanup(integrated, case.store, abandon=False)
    assert integrated.path.exists()
    assert (integrated.path / "new.txt").read_text() == (
        "unsaved" if alteration in {"dirty", "assume_unchanged"} else "saved"
    )
    assert (case.repo / "new.txt").read_text() == "saved"
    assert case.store.load_worktree(child.run_id) == integrated
