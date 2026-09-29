from pathlib import Path

import pytest

from handshake.models.protocol import ModelMessage
from handshake.runtime.dependencies import resolve_text_dependency
from handshake.runtime.errors import WaitResolutionError
from handshake.runtime.state import (
    Dependency,
    DependencyStatus,
    Run,
    RunStatus,
    Session,
)


def _waiting_run(tmp_path: Path) -> tuple[Session, Run, list[ModelMessage]]:
    session = Session(workspace=tmp_path, session_id="test-session")
    run = Run(goal="选择发布颜色", session_id=session.session_id, agent_spec_id="test")
    run.transition(RunStatus.RUNNING)
    messages = [
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
            ),
            ModelMessage(
                role="user",
                content=(
                    {
                        "type": "tool_result",
                        "tool_use_id": "ask-1",
                        "content": "dependency registered",
                        "is_error": False,
                    },
                ),
                run_id=run.run_id,
            ),
        ]
    run.add_dependency(
        Dependency(
            tool_call_id="ask-1",
            prompt="发布按钮使用什么颜色？",
        ),
    )
    run.transition(RunStatus.WAITING)
    return session, run, messages


def test_resolving_dependency_appends_a_separate_continuation_message(
    tmp_path: Path,
) -> None:
    session, run, messages = _waiting_run(tmp_path)
    dependency_id = run.dependencies[0].dependency_id

    messages.append(resolve_text_dependency(session, run, dependency_id, "  深蓝色  "))

    assert run.status is RunStatus.RUNNING
    assert run.dependencies[0].status is DependencyStatus.SATISFIED
    continuation = messages[-1].content[0]
    assert messages[-1].run_id == run.run_id
    assert continuation["type"] == "text"
    assert dependency_id in continuation["text"]
    assert "深蓝色" in continuation["text"]
    assert sum(
        block.get("tool_use_id") == "ask-1"
        for message in messages
        for block in message.content
    ) == 1
    with pytest.raises(WaitResolutionError, match="already satisfied"):
        resolve_text_dependency(session, run, dependency_id, "绿色")


def test_invalid_dependency_and_empty_response_are_rejected(tmp_path: Path) -> None:
    session, run, _messages = _waiting_run(tmp_path)

    with pytest.raises(WaitResolutionError, match="must not be empty"):
        resolve_text_dependency(
            session, run, run.dependencies[0].dependency_id, "  "
        )
    with pytest.raises(WaitResolutionError, match="does not belong"):
        resolve_text_dependency(session, run, "missing", "blue")
