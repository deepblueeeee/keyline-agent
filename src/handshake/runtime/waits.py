"""Runtime state transitions for non-sensitive text waits."""

from __future__ import annotations

import json

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import WaitResolutionError
from handshake.runtime.state import (
    Run,
    RunStatus,
    Session,
    ToolResult,
    WaitKind,
    WaitStatus,
    WorkItemStatus,
    utc_now,
)


def resolve_text_wait(
    session: Session, run: Run, wait_id: str, response: str
) -> None:
    """Satisfy one open text wait and make its run ready to continue."""
    normalized = response.strip()
    if not normalized:
        raise WaitResolutionError("Text response must not be empty")
    wait = next((item for item in run.waits if item.wait_id == wait_id), None)
    if wait is None:
        raise WaitResolutionError(
            f"Wait {wait_id} does not belong to run {run.run_id}"
        )
    if wait.kind is not WaitKind.TEXT:
        raise WaitResolutionError(f"Wait {wait_id} does not accept text input")
    if wait.status is not WaitStatus.OPEN:
        raise WaitResolutionError(f"Wait {wait_id} is already satisfied")
    if run.status is not RunStatus.WAITING:
        raise WaitResolutionError(f"Run {run.run_id} is not waiting for text input")
    work_item = next(
        (item for item in run.work_items if item.work_id == wait.work_id),
        None,
    )
    if work_item is None or work_item.status is not WorkItemStatus.WAITING:
        raise WaitResolutionError(f"Wait {wait_id} has no waiting work item")

    wait.transition(WaitStatus.SATISFIED)
    wait.response_text = normalized
    wait.resolved_at = utc_now()
    work_item.transition(WorkItemStatus.READY)
    run.transition(RunStatus.RUNNING)
    session.messages.append(
        ModelMessage(
            role="user",
            content=(
                {
                    "type": "tool_result",
                    "tool_use_id": wait.tool_call_id,
                    "content": _text_result_content(normalized),
                    "is_error": False,
                },
            ),
            run_id=run.run_id,
        )
    )
    work_item.transition(WorkItemStatus.COMPLETED)


def _text_result_content(response: str) -> str:
    result = ToolResult(ok=True, stdout=response)
    return json.dumps(
        {
            "ok": result.ok,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        },
        ensure_ascii=False,
    )
