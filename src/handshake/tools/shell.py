"""Approved subprocess tools using explicit argv and no shell expansion."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from handshake.models.base import ToolDefinition
from handshake.runtime.errors import ToolExecutionError
from handshake.runtime.state import ToolResult
from handshake.tools.registry import ToolBinding

ApprovalCallback = Callable[[Path, tuple[str, ...]], bool]

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


def _safe_process_environment() -> dict[str, str]:
    """Return useful non-secret process settings without inheriting credentials."""
    return {
        name: value
        for name, value in os.environ.items()
        if name in _ALLOWED_ENVIRONMENT_VARIABLES
    }


def _parse_process_arguments(arguments: Mapping[str, Any]) -> tuple[tuple[str, ...], float]:
    raw_argv = arguments.get("argv")
    if (
        not isinstance(raw_argv, list)
        or not raw_argv
        or any(not isinstance(item, str) or not item for item in raw_argv)
    ):
        raise ToolExecutionError("argv must be a non-empty array of non-empty strings")
    raw_timeout = arguments.get("timeout_seconds", 120)
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or not 0 < raw_timeout <= 600
    ):
        raise ToolExecutionError("timeout_seconds must be a number between 0 and 600")
    return tuple(raw_argv), float(raw_timeout)


def _timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def process_tool_handler(
    approval: ApprovalCallback,
) -> Callable[[Path, Mapping[str, Any]], ToolResult]:
    def execute(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
        argv, timeout = _parse_process_arguments(arguments)
        cwd = workspace.resolve()
        if not approval(cwd, argv):
            return ToolResult(ok=False, stderr="Execution denied by user")
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                shell=False,
                env=_safe_process_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                ok=False,
                stdout=_timeout_stream(exc.stdout),
                stderr=_timeout_stream(exc.stderr) + f"\nCommand timed out after {timeout:g}s",
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


def process_tool_bindings(approval: ApprovalCallback) -> tuple[ToolBinding, ...]:
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
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="run_tests",
                description="Run an approved test command in the workspace using explicit argv.",
                input_schema=schema,
            ),
            handler=handler,
        ),
    )
