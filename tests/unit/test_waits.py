from pathlib import Path

import pytest

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import WaitResolutionError
from handshake.runtime.state import (
    Run,
    RunStatus,
    Session,
    WaitCondition,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
)
from handshake.runtime.waits import resolve_text_wait


def _waiting_run(tmp_path: Path) -> tuple[Session, Run]:
    session = Session(workspace=tmp_path, session_id="test-session")
    run = Run(goal="选择发布颜色", session_id=session.session_id)
    run.transition(RunStatus.RUNNING)
    session.messages.append(
        ModelMessage(
            role="assistant",
            content=(
                {
                    "type": "tool_use",
                    "id": "ask-1",
                    "name": "wait_for_text",
                    "input": {"prompt": "发布按钮使用什么颜色？"},
                },
            ),
            run_id=run.run_id,
        )
    )
    work = WorkItem(description="选择颜色", tool_call_id="ask-1")
    run.work_items.append(work)
    run.waits.append(
        WaitCondition(
            work_id=work.work_id,
            tool_call_id="ask-1",
            prompt="发布按钮使用什么颜色？",
        )
    )
    run.transition(RunStatus.WAITING)
    return session, run


def test_resolving_wait_updates_runtime_state_and_links_tool_result(
    tmp_path: Path,
) -> None:
    session, run = _waiting_run(tmp_path)
    wait_id = run.waits[0].wait_id

    resolve_text_wait(session, run, wait_id, "  深蓝色  ")

    assert run.status is RunStatus.RUNNING
    assert run.waits[0].status is WaitStatus.SATISFIED
    assert run.waits[0].response_text == "深蓝色"
    assert run.work_items[0].status is WorkItemStatus.COMPLETED
    result = session.messages[-1].content[0]
    assert session.messages[-1].run_id == run.run_id
    assert result["tool_use_id"] == "ask-1"
    assert "深蓝色" in result["content"]
    with pytest.raises(WaitResolutionError, match="already satisfied"):
        resolve_text_wait(session, run, wait_id, "绿色")


def test_invalid_wait_and_empty_response_are_rejected(tmp_path: Path) -> None:
    session, run = _waiting_run(tmp_path)

    with pytest.raises(WaitResolutionError, match="must not be empty"):
        resolve_text_wait(session, run, run.waits[0].wait_id, "  ")
    with pytest.raises(WaitResolutionError, match="does not belong"):
        resolve_text_wait(session, run, "missing", "blue")
