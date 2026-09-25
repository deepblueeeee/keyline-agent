"""Small, conservative approval policy for explicit-argv commands."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class ApprovalMode(StrEnum):
    ALWAYS_ASK = "always_ask"
    DANGEROUS_ONLY = "dangerous_only"
    ALLOW_ALL = "allow_all"


def requires_approval(
    mode: ApprovalMode,
    cwd: Path,
    argv: tuple[str, ...],
    *,
    with_secret: bool = False,
) -> bool:
    """Unknown commands need approval; this heuristic is not a sandbox."""
    if mode is ApprovalMode.ALWAYS_ASK:
        return True
    if mode is ApprovalMode.ALLOW_ALL:
        return False
    if with_secret:
        return True
    return not _low_risk_command(cwd, argv)


def _workspace_path(cwd: Path, value: str) -> bool:
    path = Path(value.split("::", 1)[0])
    if value.startswith("-") or path.is_absolute() or ".." in path.parts:
        return False
    resolved = (cwd / path).resolve(strict=False)
    return resolved.is_relative_to(cwd.resolve()) and not resolved.is_relative_to(
        Path.home() / ".handshake"
    )


def _low_risk_command(cwd: Path, argv: tuple[str, ...]) -> bool:
    if not argv:
        return False
    if (Path.home() / ".handshake").resolve(strict=False).is_relative_to(cwd.resolve()):
        return False
    if "/" in argv[0] or "\\" in argv[0]:
        return False
    executable = Path(argv[0]).name
    args = argv[1:]
    if executable == "pwd":
        return not args
    if executable == "ls":
        return all(arg in {"-a", "-l", "-la", "-al"} or _workspace_path(cwd, arg) for arg in args)
    if executable == "git":
        return args in {
            ("status",),
            ("status", "--short"),
            ("status", "--porcelain"),
            ("diff", "--no-ext-diff"),
            ("diff", "--no-ext-diff", "--stat"),
        }
    if executable == "rg":
        if args == ("--files",):
            return True
        if args and args[0] == "-n":
            args = args[1:]
        return bool(args) and not args[0].startswith("-") and len(args) <= 2 and (
            len(args) == 1 or _workspace_path(cwd, args[1])
        )
    if executable == "ruff" and args and args[0] == "check":
        return all(
            arg in {"-q", "--no-fix"}
            or (not arg.startswith("-") and _workspace_path(cwd, arg))
            for arg in args[1:]
        )
    return False
