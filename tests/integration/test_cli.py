from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import handshake.cli as cli
from handshake.runtime.state import ModelContext, ModelTurn, TaskStatus, ToolCall
from handshake.storage.sqlite import SQLiteStateStore


class _TTY:
    def __init__(self, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


class _WaitThenFinishModel:
    def complete(self, context: ModelContext, tools: tuple[Any, ...]) -> ModelTurn:
        assert any(tool.name == "wait_for_text" for tool in tools)
        has_answer = any(
            block.get("type") == "tool_result"
            for message in context.messages
            for block in message.content
        )
        if has_answer:
            return ModelTurn(
                content=({"type": "text", "text": "Applied the user's choice."},),
                stop_reason="end_turn",
            )
        call = ToolCall(
            "ask-color",
            "wait_for_text",
            {"prompt": "Which button color should be used?"},
        )
        return ModelTurn(
            content=(
                {"type": "thinking", "thinking": "This is a product decision."},
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                },
            ),
            tool_calls=(call,),
            stop_reason="tool_use",
        )


class _RecordingFinishModel:
    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.contexts: list[ModelContext] = []

    def complete(self, context: ModelContext, _tools: tuple[Any, ...]) -> ModelTurn:
        self.contexts.append(context)
        return ModelTurn(
            content=({"type": "text", "text": self.summary},),
            stop_reason="end_turn",
        )


def _store(home: Path) -> SQLiteStateStore:
    return SQLiteStateStore(home / "handshake.db")


def test_demo_runs_without_api_key(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = cli.main(["demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task completed" in captured.out
    assert "verified the fixture tests pass" in captured.out


def test_real_run_requires_api_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = cli.main(["run", str(tmp_path), "inspect the repository"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "DEEPSEEK_API_KEY is required" in captured.err


def test_run_answers_text_wait_and_finishes_in_same_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _WaitThenFinishModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "navy" if prompt == "Response: " else "/exit",
    )

    exit_code = cli.main(["run", str(tmp_path), "style the button"])

    captured = capsys.readouterr()
    tasks = _store(home).list_tasks()
    assert exit_code == 0
    assert "Input received; resuming task" in captured.out
    assert "Applied the user's choice." in captured.out
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.COMPLETED
    assert tasks[0].waits[0].response_text == "navy"


def test_wait_survives_noninteractive_exit_interrupt_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _WaitThenFinishModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))

    first_exit = cli.main(["run", str(tmp_path), "style the button"])
    task = _store(home).list_tasks()[0]
    session_id = task.session_id

    assert first_exit == 2
    assert task.status is TaskStatus.WAITING
    assert cli.main(["list"]) == 0
    assert cli.main(["inspect", session_id]) == 0
    assert cli.main(["inspect-task", task.task_id]) == 0
    listing = capsys.readouterr().out
    assert task.task_id in listing
    assert "waiting" in listing
    assert "Which button color" in listing

    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert cli.main(["resume", session_id]) == 130
    assert _store(home).load_task(task.task_id).status is TaskStatus.WAITING

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "navy" if prompt == "Response: " else "/exit",
    )
    assert cli.main(["resume", session_id]) == 0
    resumed = _store(home).load_task(task.task_id)
    assert resumed.status is TaskStatus.COMPLETED
    assert resumed.messages[-2].content[0]["tool_use_id"] == "ask-color"


def test_interactive_session_creates_followup_task_with_sanitized_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "state"
    first_model = _RecordingFinishModel("first result")
    second_model = _RecordingFinishModel("second result")
    models = iter([first_model, second_model])
    prompts = iter(["second request", "/exit"])
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: next(models))
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))

    exit_code = cli.main(
        ["run", str(tmp_path), "first request", "--max-steps", "9"]
    )

    session = _store(home).list_sessions()[0]
    tasks = _store(home).list_tasks(session.session_id)
    assert exit_code == 0
    assert [task.goal for task in tasks] == ["first request", "second request"]
    assert tasks[1].previous_task_id == tasks[0].task_id
    assert tasks[1].max_steps == 9
    assert tasks[1].model_turns == 1
    assert second_model.contexts[0].session_history[0].outcome == "first result"
    assert "first result" in second_model.contexts[0].system_prompt
