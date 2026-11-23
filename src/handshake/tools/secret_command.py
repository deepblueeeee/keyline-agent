"""Dedicated command runner holding the one-time Secret consume capability."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from handshake.inputs import SecretConsumer
from handshake.inputs.cancellation import check_cancelled
from handshake.models.base import ToolDefinition
from handshake.runtime.state import Run, SecretHandleStatus, ToolResult
from handshake.tools.registry import PreparedTool, ToolPreparation
from handshake.tools.secret_wait import (
    SECRET_COMMAND_CONSUMER,
    SECRET_ENVIRONMENT_VARIABLES,
)
from handshake.tools.shell import (
    ApprovalCallback,
    AsyncApprovalCallback,
    parse_process_arguments,
    safe_process_environment,
    timeout_stream,
)

SECRET_COMMAND_TOOL = ToolDefinition(
    name=SECRET_COMMAND_CONSUMER,
    description=(
        "Run an approved command with a previously requested opaque Secret handle. "
        "The associated environment variable is fixed when the handle is requested."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "secret_handle": {"type": "string", "minLength": 1},
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
        "required": ["secret_handle", "argv"],
        "additionalProperties": False,
    },
)


class SecretCommandRunner:
    """Execute one approved command with one consumed Secret value."""

    def __init__(
        self, consumer: SecretConsumer, approval: ApprovalCallback,
        async_approval: AsyncApprovalCallback | None = None,
    ) -> None:
        self._consumer = consumer
        self._approval = approval
        self._async_approval = async_approval

    def contains_secret(self, secret_handle: str) -> bool:
        """Report process-local availability without exposing the raw value."""
        return self._consumer.contains(secret_handle)

    def execute(
        self, run: Run, workspace: Path, arguments: Mapping[str, Any]
    ) -> ToolResult:
        prepared = cast(PreparedTool, self._prepare(run, workspace, arguments, self._approval))
        return prepared if isinstance(prepared, ToolResult) else prepared()

    def prepare(
        self, run: Run, workspace: Path, arguments: Mapping[str, Any]
    ) -> ToolPreparation:
        return self._prepare(run, workspace, arguments, self._async_approval or self._approval)

    def _prepare(
        self, run: Run, workspace: Path, arguments: Mapping[str, Any],
        approval: ApprovalCallback | AsyncApprovalCallback,
    ) -> ToolPreparation:
        raw_handle = arguments.get("secret_handle")
        if not isinstance(raw_handle, str) or not raw_handle:
            return ToolResult(ok=False, stderr="secret_handle must be a non-empty string")
        handle = run.secret_handle_by_id(raw_handle)
        if handle is None:
            return ToolResult(ok=False, stderr="Secret handle does not belong to this run")
        if handle.status is not SecretHandleStatus.AVAILABLE:
            return ToolResult(
                ok=False,
                stderr=f"Secret handle is not available ({handle.status.value})",
            )
        if handle.consumer != SECRET_COMMAND_CONSUMER:
            return ToolResult(ok=False, stderr="Secret handle has the wrong consumer")
        if handle.environment_variable not in SECRET_ENVIRONMENT_VARIABLES:
            return ToolResult(
                ok=False,
                stderr="Secret handle has an unsupported environment variable",
            )
        try:
            argv, timeout = parse_process_arguments(arguments)
        except Exception as exc:
            return ToolResult(ok=False, stderr=str(exc))
        if not self._consumer.contains(handle.secret_handle):
            return ToolResult(ok=False, stderr="Secret value is unavailable; request it again")
        cwd = workspace.resolve()

        def execute() -> ToolResult:
            check_cancelled()
            raw_value = self._consumer.consume(handle.secret_handle)
            try:
                environment = safe_process_environment()
                environment[handle.environment_variable] = raw_value
                try:
                    completed = subprocess.run(
                        argv,
                        cwd=cwd,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                        shell=False,
                        env=environment,
                    )
                except subprocess.TimeoutExpired as exc:
                    result = ToolResult(
                        ok=False,
                        stdout=_redact(timeout_stream(exc.stdout), raw_value),
                        stderr=_redact(timeout_stream(exc.stderr), raw_value)
                        + f"\nCommand timed out after {timeout:g}s",
                    )
                except OSError as exc:
                    result = ToolResult(
                        ok=False,
                        stderr=_redact(
                            f"Could not start secure test process: {exc}", raw_value
                        ),
                    )
                else:
                    result = ToolResult(
                        ok=completed.returncode == 0,
                        stdout=_redact(completed.stdout, raw_value),
                        stderr=_redact(completed.stderr, raw_value),
                        exit_code=completed.returncode,
                    )
            finally:
                handle.transition(SecretHandleStatus.CONSUMED)
            del raw_value
            return result

        def after_approval(approved: bool) -> PreparedTool:
            if not approved:
                return ToolResult(ok=False, stderr="Execution denied by user")
            if not self._consumer.contains(handle.secret_handle):
                return ToolResult(ok=False, stderr="Secret value is unavailable; request it again")
            check_cancelled()
            return execute

        decision = approval(cwd, argv)
        if isinstance(decision, bool):
            return after_approval(decision)

        async def request() -> PreparedTool:
            return after_approval(await decision)

        return request()

    def __repr__(self) -> str:
        return "SecretCommandRunner(<secret consume capability>)"


def _redact(value: str, raw_secret: str) -> str:
    return value.replace(raw_secret, "[REDACTED]")
