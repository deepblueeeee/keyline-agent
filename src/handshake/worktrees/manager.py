"""Create, verify, snapshot and integrate Handshake-owned Git worktrees."""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from handshake.errors import HandshakeError
from handshake.runtime.persistence import RunStore
from handshake.runtime.state import Worktree, WorktreeStatus
from handshake.tools.shell import safe_process_environment


class WorktreeError(HandshakeError):
    """The Git state does not permit the requested workspace operation."""


def _git(
    path: Path, *args: str, allowed: tuple[int, ...] = (0,), data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = safe_process_environment()
    env.update(GIT_TERMINAL_PROMPT="0", GIT_NO_REPLACE_OBJECTS="1")
    try:
        result = subprocess.run(
            [
                "git", "-C", str(path), "-c", "core.hooksPath=/dev/null",
                "-c", "core.fsmonitor=false", "-c", "commit.gpgSign=false",
                "-c", "submodule.recurse=false",
                "-c", "user.name=Handshake", "-c", "user.email=handshake@localhost",
                *args,
            ],
            env=env, input=data, capture_output=True, check=False, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(f"Git {args[0]} could not finish ({type(exc).__name__})") from None
    if result.returncode not in allowed:
        detail = _text(result.stderr or result.stdout).strip()[:1000]
        raise WorktreeError(f"Git {args[0]} failed: {detail or result.returncode}")
    return result


def _text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _head(path: Path, ref: str = "HEAD") -> str:
    return _text(_git(path, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout).strip()


def _ref(path: Path) -> str:
    return _text(_git(path, "symbolic-ref", "-q", "HEAD", allowed=(0, 1)).stdout).strip()


def _ancestor(path: Path, first: str, second: str) -> bool:
    result = _git(path, "merge-base", "--is-ancestor", first, second, allowed=(0, 1))
    return result.returncode == 0


def _check_idle(path: Path) -> None:
    git_dir = Path(os.fsdecode(_git(path, "rev-parse", "--absolute-git-dir").stdout.rstrip(b"\n")))
    for name in (
        "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD",
        "rebase-apply", "rebase-merge", "sequencer", "BISECT_START", "index.lock",
    ):
        if (git_dir / name).exists():
            raise WorktreeError(f"Git operation already in progress in {path} ({name})")


def _check_clean(path: Path, *, protect_ignored: bool = False) -> None:
    args = ("--ignored=matching",) if protect_ignored else ()
    status = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                  "--ignore-submodules=none", *args)
    for entry in status.stdout.split(b"\0")[:-1]:
        if entry.startswith(b"!! "):
            name = entry[3:]
            if name.startswith((b".venv/", b".pytest_cache/")):
                continue
            raise WorktreeError(f"Protected ignored path in Worktree: {os.fsdecode(name)!r}")
        raise WorktreeError(f"Workspace has uncommitted or untracked changes: {path}")


def _matches_checkout(path: Path, commit: str) -> bool:
    return (
        _git(path, "diff", "--cached", "--quiet", "--ignore-submodules=none", commit, "--",
             allowed=(0, 1)).returncode == 0
        and _git(path, "diff", "--quiet", "--ignore-submodules=none", "--",
                 allowed=(0, 1)).returncode == 0
        and not _git(path, "ls-files", "--others", "--exclude-standard", "-z").stdout
    )


def _patch(repo: Path, base: str, commit: str | None = None) -> bytes:
    revisions = ("--cached", base) if commit is None else (base, commit)
    return _git(
        repo, "diff", "--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv",
        "--no-renames", "--ignore-submodules=none", "--src-prefix=a/", "--dst-prefix=b/",
        *revisions, "--",
    ).stdout


def _check_initialized_submodules(repo: Path, head: str, candidate: str) -> None:
    raw = _git(
        repo, "diff", "--raw", "-z", "--no-renames", "--ignore-submodules=none",
        head, candidate, "--",
    ).stdout.split(b"\0")[:-1]
    for header, name in zip(raw[::2], raw[1::2], strict=True):
        if b"160000" in header[1:14].split():
            path = os.fsdecode(name)
            if os.path.lexists(repo / path / ".git"):
                raise WorktreeError(f"Cannot integrate changes to initialized submodule: {path}")


def _check_new_gitlinks(tree: Worktree) -> None:
    raw = _git(
        tree.path, "diff", "--cached", "--raw", "-z", "--no-renames", "--ignore-submodules=none",
        tree.base_commit, "--",
    ).stdout
    records = raw.split(b"\0")[:-1]
    links = {
        path for header, path in zip(records[::2], records[1::2], strict=True)
        if header[8:14] == b"160000" and header[1:7] != b"160000"
    }
    if not links:
        return
    modules = _git(
        tree.path, "config", "--null", "--blob", ":.gitmodules", "--get-regexp",
        r"^submodule\..*\.path$", allowed=(0, 1, 128),
    ).stdout
    registered = {entry.partition(b"\n")[2] for entry in modules.split(b"\0")}
    if unregistered := links - registered:
        paths = ", ".join(repr(os.fsdecode(path)) for path in sorted(unregistered))
        raise WorktreeError(f"Cannot save embedded Git repositories as files: {paths}")


def _registrations(repository: Path) -> dict[Path, dict[str, str]]:
    records: dict[Path, dict[str, str]] = {}
    current: dict[str, str] = {}
    output = _git(repository, "worktree", "list", "--porcelain", "-z").stdout
    for entry in os.fsdecode(output).split("\0"):
        if not entry:
            if "worktree" in current:
                records[Path(current["worktree"])] = current
            current = {}
        else:
            key, _, value = entry.partition(" ")
            current[key] = value
    return records


def _common_directory(path: Path) -> Path:
    result = _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(os.fsdecode(result.stdout.rstrip(b"\n"))).resolve()


class WorktreeManager:
    """Git operations and the durable boundaries around them."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.home() / ".handshake-worktrees").resolve()

    @staticmethod
    def _check_root(root: Path, repository: Path) -> None:
        private = (Path.home() / ".handshake").resolve()
        if root.resolve() != root or any(
            root.is_relative_to(path) or path.is_relative_to(root) for path in (repository, private)
        ):
            raise WorktreeError("Worktree storage must be outside the repository and credentials")

    def creation_base(self, workspace: Path) -> tuple[Path, str, str]:
        repository = workspace.resolve()
        if Path(_text(_git(repository, "rev-parse", "--show-toplevel").stdout).rstrip("\n")
                ).resolve() != repository:
            raise WorktreeError("Implementers require the Session workspace to be a Git root")
        self._check_root(self.root, repository)
        _check_idle(repository)
        _check_clean(repository)
        ref = _ref(repository)
        if not ref.startswith("refs/heads/"):
            raise WorktreeError("Implementers require the main workspace to be on a branch")
        return repository, _head(repository), ref

    def plan(self, owner_run_id: str, base: tuple[Path, str, str]) -> Worktree:
        repository, commit, main_ref = base
        key = sha256(owner_run_id.encode()).hexdigest()
        return Worktree(owner_run_id, repository, self.root / key, commit,
                        f"handshake/{key}", main_ref)

    @contextmanager
    def _locked(self, tree: Worktree) -> Iterator[None]:
        root = tree.path.parent
        self._check_root(root, tree.repository)
        name = f".lock-{sha256(os.fsencode(_common_directory(tree.repository))).hexdigest()}"
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = (root / name).open("a")
        except OSError as exc:
            raise WorktreeError(f"Cannot access the Worktree lock: {exc.strerror}") from None
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            except OSError as exc:
                raise WorktreeError(f"Cannot acquire the Worktree lock: {exc.strerror}") from None
            yield

    def validate(self, tree: Worktree) -> Path:
        if tree.cleaned or tree.cleanup_pending:
            raise WorktreeError(f"Worktree for {tree.owner_run_id} was cleaned or is being cleaned")
        if tree.path.resolve() != tree.path or not tree.path.is_dir():
            raise WorktreeError(f"Worktree is missing or moved: {tree.path}")
        registration = _registrations(tree.repository).get(tree.path)
        if (registration is None or registration.get("branch") != f"refs/heads/{tree.branch}"
                or "prunable" in registration):
            raise WorktreeError(f"Worktree registration or branch changed: {tree.path}")
        top = _text(_git(tree.path, "rev-parse", "--show-toplevel").stdout).rstrip("\n")
        if (
            Path(top).resolve() != tree.path
            or _ref(tree.path) != f"refs/heads/{tree.branch}"
            or _common_directory(tree.path) != _common_directory(tree.repository)
        ):
            raise WorktreeError(f"Worktree checkout does not match its binding: {tree.path}")
        if not _ancestor(tree.path, tree.base_commit, _head(tree.path)):
            raise WorktreeError(f"Worktree history no longer includes its base: {tree.path}")
        return tree.path

    def provision(self, tree: Worktree, store: RunStore) -> Worktree:
        with self._locked(tree):
            tree = self._reload(tree, store)
            if tree.status is not WorktreeStatus.PLANNED:
                self.validate(tree)
                return tree
            registered = _registrations(tree.repository).get(tree.path)
            if registered is None and not os.path.lexists(tree.path):
                _check_idle(tree.repository)
                _check_clean(tree.repository)
                ref = f"refs/heads/{tree.branch}"
                branch = _git(
                    tree.repository, "show-ref", "--verify", "--hash", ref, allowed=(0, 128),
                )
                if branch.returncode:
                    _git(tree.repository, "branch", tree.branch, tree.base_commit)
                elif _text(branch.stdout).strip() != tree.base_commit:
                    raise WorktreeError("Planned worktree branch points to different changes")
                _git(tree.repository, "worktree", "add", "--", str(tree.path), tree.branch)
            self.validate(tree)
            if _head(tree.path) != tree.base_commit:
                raise WorktreeError("Unfinished worktree creation has unexpected commits")
            _check_clean(tree.path)
            return self._update(store, tree, status=WorktreeStatus.ACTIVE)

    def snapshot(self, tree: Worktree, store: RunStore) -> Worktree:
        with self._locked(tree):
            return self._snapshot(self._reload(tree, store), store)

    def _snapshot(self, tree: Worktree, store: RunStore) -> Worktree:
        self.validate(tree)
        if tree.status is WorktreeStatus.SAVED:
            self._check_snapshot(tree)
            return tree
        if tree.status is not WorktreeStatus.ACTIVE:
            raise WorktreeError(f"Cannot save a worktree in state {tree.status.value}")
        _check_idle(tree.path)
        _git(tree.path, "add", "-A", "--", ".")
        _check_new_gitlinks(tree)
        if _git(tree.path, "diff", "--cached", "--quiet", "--", allowed=(0, 1)).returncode:
            _git(
                tree.path, "commit", "--no-gpg-sign", "-m", f"Save changes for {tree.owner_run_id}",
            )
        _check_clean(tree.path)
        return self._update(store, tree, status=WorktreeStatus.SAVED, saved_commit=_head(tree.path))

    def _check_snapshot(self, tree: Worktree) -> str:
        path = self.validate(tree)
        _check_idle(path)
        _check_clean(path)
        if tree.saved_commit is None or _head(path) != tree.saved_commit:
            raise WorktreeError("Worktree no longer matches its saved commit")
        return tree.saved_commit

    def changes(self, tree: Worktree) -> dict[str, object]:
        commit = tree.saved_commit
        if commit is None:
            raise WorktreeError("This worktree has no saved change snapshot")
        items = _text(_git(tree.repository, "diff", "--name-status", "-z", "--no-renames",
                           tree.base_commit, commit, "--").stdout).split("\0")
        files = [{"status": status, "path": path}
                 for status, path in zip(items[0:-1:2], items[1:-1:2], strict=True)]
        return {
            "child_run_id": tree.owner_run_id, "worktree_status": tree.status.value,
            "base_commit": tree.base_commit, "commit": commit, "branch": tree.branch,
            "integration_commit": tree.integration_commit, "cleaned": tree.cleaned,
            "files": files,
        }

    def diff(self, tree: Worktree, *, start_line: int = 1, limit: int = 150) -> dict[str, object]:
        if tree.saved_commit is None:
            raise WorktreeError("This worktree has no saved change snapshot")
        lines = _text(_git(tree.repository, "diff", "--no-ext-diff", "--no-textconv",
                           tree.base_commit, tree.saved_commit, "--").stdout).splitlines()
        page: list[str] = []
        used = 0
        for line in lines[start_line - 1 : start_line - 1 + limit]:
            if used + len(line) > 12000 and page:
                break
            page.append(line[:12000])
            used += len(page[-1]) + 1
        next_line = start_line + len(page)
        return {
            "diff": "\n".join(page), "start_line": start_line,
            "next_line": next_line if next_line <= len(lines) else None,
            "line_truncated": any(
                len(line) > 12000 for line in lines[start_line - 1 : next_line - 1]
            ),
        }

    def integrate(self, tree: Worktree, store: RunStore) -> Worktree:
        with self._locked(tree):
            tree = self._reload(tree, store)
            if tree.status is WorktreeStatus.INTEGRATED:
                self._release_candidate(tree)
                return tree
            repo = tree.repository
            _check_idle(repo)
            if _ref(repo) != tree.main_ref:
                raise WorktreeError("Main workspace is no longer on the original branch")
            head = _head(repo)
            if tree.status is WorktreeStatus.INTEGRATING:
                old_head = tree.integration_head
                if tree.integration_commit is None or old_head is None:
                    raise WorktreeError("Integration intent is incomplete")
                candidate = self._existing_candidate(tree)
                if candidate is not None and _ancestor(repo, candidate, head):
                    _check_clean(repo)
                    return self._finish_integration(tree, store)
                if old_head != head:
                    _check_clean(repo)
                    if not _ancestor(repo, old_head, head):
                        raise WorktreeError("Main history changed during integration; changes kept")
                    tree = self._clear_intent(tree, store)
                elif candidate is None:
                    tree = self._restore_candidate(tree, head, store)
            if tree.status is WorktreeStatus.SAVED:
                _check_clean(repo)
                saved = self._check_snapshot(tree)
                if _ancestor(repo, saved, head):
                    return self._update(
                        store, tree, status=WorktreeStatus.INTEGRATED, integration_commit=head,
                    )
                candidate = self._merge_candidate(repo, head, saved, tree.owner_run_id)
                _git(repo, "update-ref", self._candidate_ref(tree), candidate)
                tree = self._update(
                    store, tree, status=WorktreeStatus.INTEGRATING,
                    integration_head=head, integration_commit=candidate,
                )
            elif tree.status is not WorktreeStatus.INTEGRATING:
                raise WorktreeError(f"Cannot integrate a worktree in state {tree.status.value}")
            assert tree.integration_commit is not None
            try:
                self._apply_integration(tree, head, tree.integration_commit)
            except WorktreeError:
                self._record_unapplied_merge(tree, store)
                raise
            return self._finish_integration(tree, store)

    def _restore_candidate(self, tree: Worktree, head: str, store: RunStore) -> Worktree:
        # Older integration records may refer to a commit that Git could prune.
        _check_clean(tree.repository)
        saved = self._check_snapshot(tree)
        candidate = self._merge_candidate(tree.repository, head, saved, tree.owner_run_id)
        _git(tree.repository, "update-ref", self._candidate_ref(tree), candidate)
        return self._update(store, tree, integration_commit=candidate)

    @staticmethod
    def _apply_integration(tree: Worktree, head: str, candidate: str) -> None:
        repo = tree.repository
        patch = _patch(repo, head, candidate)
        if _ref(repo) != tree.main_ref or _head(repo) != head:
            raise WorktreeError("Main workspace changed while preparing integration")
        applied = False
        try:
            if not _matches_checkout(repo, candidate):
                _check_clean(repo)
                _check_initialized_submodules(repo, head, candidate)
                if patch:
                    _git(repo, "apply", "--index", "--whitespace=nowarn", data=patch)
                    applied = True
                if not _matches_checkout(repo, candidate):
                    raise WorktreeError("Main checkout differs from the integration candidate")
            if _ref(repo) != tree.main_ref or _head(repo) != head:
                raise WorktreeError("Main workspace changed during integration")
            # Compare and update the recorded branch without dereferencing HEAD.
            _git(repo, "update-ref", "--no-deref", "-m", "Handshake integration",
                 tree.main_ref, candidate, head)
            if _ref(repo) != tree.main_ref or _head(repo) != candidate:
                raise WorktreeError(
                    "Main checkout changed during integration; inspect before resuming"
                )
        except WorktreeError as exc:
            if applied and not WorktreeManager._undo_patch(repo, patch):
                raise WorktreeError(f"{exc}. Inspect the staging area before retrying.") from exc
            raise
        _check_clean(repo)

    @staticmethod
    def _undo_patch(repo: Path, patch: bytes) -> bool:
        # A branch switch may have changed HEAD. Compare against the current commit
        # and reverse only when the entire staged diff still equals our patch.
        ref, head = _ref(repo), _head(repo)
        unstaged = _git(repo, "diff", "--quiet", "--ignore-submodules=none", "--", allowed=(0, 1))
        if (
            unstaged.returncode == 0 and _patch(repo, head) == patch
            and _ref(repo) == ref and _head(repo) == head
        ):
            _git(repo, "apply", "--reverse", "--index", "--whitespace=nowarn", data=patch)
            return True
        return False

    @staticmethod
    def _existing_candidate(tree: Worktree) -> str | None:
        candidate = tree.integration_commit
        if candidate is not None and _git(
            tree.repository, "cat-file", "-e", f"{candidate}^{{commit}}", allowed=(0, 1, 128),
        ).returncode == 0:
            return candidate
        return None

    @staticmethod
    def _candidate_ref(tree: Worktree) -> str:
        return f"refs/handshake/integrations/{sha256(tree.owner_run_id.encode()).hexdigest()}"

    def _release_candidate(self, tree: Worktree) -> None:
        if tree.integration_head is not None and tree.integration_commit is not None:
            # An interrupted deletion may leave this ref absent. The expected OID
            # prevents deleting a ref that has been replaced independently.
            _git(tree.repository, "update-ref", "-d", self._candidate_ref(tree),
                 tree.integration_commit, allowed=(0, 1, 128))

    def _finish_integration(self, tree: Worktree, store: RunStore) -> Worktree:
        integrated = self._update(store, tree, status=WorktreeStatus.INTEGRATED)
        self._release_candidate(integrated)
        return integrated

    def _clear_intent(self, tree: Worktree, store: RunStore) -> Worktree:
        saved = self._update(
            store, tree, status=WorktreeStatus.SAVED,
            integration_head=None, integration_commit=None,
        )
        self._release_candidate(tree)
        return saved

    def _record_unapplied_merge(self, tree: Worktree, store: RunStore) -> None:
        repo = tree.repository
        try:
            _check_idle(repo)
            _check_clean(repo)
            unchanged = _head(repo, tree.main_ref) == tree.integration_head
        except WorktreeError:
            # Keep the integration intent when Git's result is uncertain.
            return
        if unchanged:
            self._clear_intent(tree, store)

    @staticmethod
    def _merge_candidate(repo: Path, head: str, saved: str, owner_id: str) -> str:
        if _ancestor(repo, head, saved):
            return saved
        merge = _git(repo, "merge-tree", "--write-tree", head, saved, allowed=(0, 1))
        lines = _text(merge.stdout).splitlines()
        if merge.returncode:
            details = "\n".join(lines[1:])[:3000]
            raise WorktreeError(f"Git merge conflict. Worktree retained.\n{details}")
        return _text(_git(repo, "commit-tree", lines[0], "-p", head, "-p", saved,
                          "-m", f"Integrate changes for {owner_id}").stdout).strip()

    def cleanup(self, tree: Worktree, store: RunStore, *, abandon: bool) -> Worktree:
        with self._locked(tree):
            tree = self._reload(tree, store)
            if tree.cleaned:
                return tree
            if not tree.cleanup_pending:
                if abandon and tree.status is WorktreeStatus.INTEGRATING:
                    _check_idle(tree.repository)
                    _check_clean(tree.repository)
                    candidate = self._existing_candidate(tree)
                    if candidate is not None and _ancestor(
                        tree.repository, candidate, _head(tree.repository, tree.main_ref),
                    ):
                        tree = self._finish_integration(tree, store)
                    else:
                        tree = self._clear_intent(tree, store)
                if tree.status is WorktreeStatus.INTEGRATED:
                    self._check_snapshot(tree)
                elif abandon and tree.status in {WorktreeStatus.ACTIVE, WorktreeStatus.SAVED}:
                    tree = self._snapshot(tree, store)
                elif tree.status is not WorktreeStatus.ABANDONED:
                    raise WorktreeError("Integrate the worktree or explicitly abandon its changes")
                _check_clean(tree.path, protect_ignored=True)
                tree = self._update(
                    store, tree, cleanup_pending=True,
                    status=(WorktreeStatus.INTEGRATED if tree.status is WorktreeStatus.INTEGRATED
                            else WorktreeStatus.ABANDONED),
                )
            registered = _registrations(tree.repository).get(tree.path)
            if registered is not None or os.path.lexists(tree.path):
                active = replace(tree, cleanup_pending=False)
                self._check_snapshot(active)
                _check_clean(tree.path, protect_ignored=True)
                _git(tree.repository, "worktree", "remove", "--", str(tree.path))
            # Keep the branch as an explicit backup, including for abandoned changes.
            return self._update(store, tree, cleanup_pending=False, cleaned=True)

    @staticmethod
    def _reload(tree: Worktree, store: RunStore) -> Worktree:
        current = store.load_worktree(tree.owner_run_id)
        if current is None:
            raise WorktreeError("Worktree record no longer exists")
        return current

    @staticmethod
    def _update(store: RunStore, tree: Worktree, **values: object) -> Worktree:
        updated = replace(tree, **values)  # type: ignore[arg-type]
        store.save_worktree(tree, updated)
        return updated
