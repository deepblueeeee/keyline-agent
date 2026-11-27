"""Approved subprocess tools using explicit argv and no shell expansion."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from keyline_agent.inputs.cancellation import check_cancelled
from keyline_agent.models.base import ToolDefinition
from keyline_agent.runtime.errors import ToolExecutionError
from keyline_agent.runtime.state import ToolResult
from keyline_agent.tools.registry import PreparedTool, ToolBinding, ToolHandler, ToolPreparation

ApprovalCallback = Callable[[Path, tuple[str, ...]], bool]
AsyncApprovalCallback = Callable[[Path, tuple[str, ...]], Awaitable[bool]]

_ALLOWED_ENVIRONMENT_VARIABLES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "VIRTUAL_ENV",
}
_ORIGINAL_WORKSPACE: ContextVar[Path | None] = ContextVar("original_workspace", default=None)


def safe_process_environment() -> dict[str, str]:
    """Return useful non-secret process settings without inheriting credentials."""
    return {
        name: value
        for name, value in os.environ.items()
        if name in _ALLOWED_ENVIRONMENT_VARIABLES
    }


@contextmanager
def worktree_process_scope(original_workspace: Path | None) -> Iterator[None]:
    """Carry the original checkout through a tool's preparation and execution."""
    token = _ORIGINAL_WORKSPACE.set(original_workspace)
    try:
        yield
    finally:
        _ORIGINAL_WORKSPACE.reset(token)


def _inside_checkout(cwd: Path, path: str, original: Path) -> bool:
    candidate = Path(os.path.abspath(cwd / path))
    return candidate.is_relative_to(original) or candidate.resolve().is_relative_to(original)


def prepare_process_environment(cwd: Path, argv: tuple[str, ...]) -> dict[str, str]:
    env = safe_process_environment()
    original = _ORIGINAL_WORKSPACE.get()
    if original is None:
        return env
    for name in ("PATH", "PYTHONPATH"):
        if name in env:
            env[name] = os.pathsep.join(
                path for path in env[name].split(os.pathsep)
                if not _inside_checkout(cwd, path, original)
            )
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if name in env and _inside_checkout(cwd, env[name], original):
            del env[name]
            if name == "CONDA_PREFIX":
                env.pop("CONDA_DEFAULT_ENV", None)
    executable = argv[0]
    if not os.path.dirname(executable):
        search_path = env.get("PATH", os.defpath).split(os.pathsep)
        executable = shutil.which(executable, path=os.pathsep.join(
            str(cwd / entry) for entry in search_path
        )) or executable
    if _inside_checkout(cwd, executable, original):
        raise ToolExecutionError(
            "Executable belongs to the original workspace. Use an environment in this worktree."
        )
    return env


def parse_process_arguments(
    arguments: Mapping[str, Any], *, timeout_limit: int = 600,
) -> tuple[tuple[str, ...], float]:
    raw_argv = arguments.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or any(not isinstance(item, str) or not item for item in raw_argv)
    ):
        raise ToolExecutionError("argv must be a non-empty array of non-empty strings")
    raw_timeout = arguments.get("timeout_seconds", min(120, timeout_limit))
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not 0 < raw_timeout <= timeout_limit
    ):
        raise ToolExecutionError(
            f"timeout_seconds must be a number greater than 0 and at most {timeout_limit}"
        )
    return tuple(raw_argv), float(raw_timeout)


def timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def prepare_process_tool(
    approval: ApprovalCallback, workspace: Path, arguments: Mapping[str, Any]
) -> PreparedTool:
    argv, timeout = parse_process_arguments(arguments)
    cwd = workspace.resolve()
    env = prepare_process_environment(cwd, argv)
    if not approval(cwd, argv):
        return ToolResult(ok=False, stderr="Execution denied by user")
    check_cancelled()

    def execute() -> ToolResult:
        check_cancelled()
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                ok=False,
                stdout=timeout_stream(exc.stdout),
                stderr=timeout_stream(exc.stderr) + f"\nCommand timed out after {timeout:g}s",
            )
        except OSError as exc:
            return ToolResult(ok=False, stderr=f"Could not start process: {exc}")
        return ToolResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )

    return execute


def process_tool_handler(approval: ApprovalCallback) -> ToolHandler:
    def execute(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
        prepared = prepare_process_tool(approval, workspace, arguments)
        return prepared if isinstance(prepared, ToolResult) else prepared()

    return execute


def process_tool_bindings(
    approval: ApprovalCallback, async_approval: AsyncApprovalCallback | None = None
) -> tuple[ToolBinding, ...]:
    """Return process tool definitions already paired with an approved handler."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 600,
            },
        },
        "required": ["argv"],
        "additionalProperties": False,
    }

    def prepare(workspace: Path, arguments: Mapping[str, Any]) -> ToolPreparation:
        if async_approval is None:
            return prepare_process_tool(approval, workspace, arguments)
        argv, _timeout = parse_process_arguments(arguments)
        cwd = workspace.resolve()

        async def request() -> PreparedTool:
            approved = await async_approval(cwd, argv)
            return prepare_process_tool(lambda _cwd, _argv: approved, workspace, arguments)

        return request()

    handler = process_tool_handler(approval)
    return (
        ToolBinding(
            definition=ToolDefinition(
                name="run_shell",
                description=(
                    "Run an approved command in the workspace using explicit argv "
                    "(no shell syntax)."
                ),
                input_schema=schema,
            ),
            handler=handler,
            prepare=prepare,
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="run_tests",
                description="Run an approved test command in the workspace using explicit argv.",
                input_schema=schema,
            ),
            handler=handler,
            prepare=prepare,
        ),
    )
