from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import handshake.cli as cli
from handshake.models.protocol import ModelContext, ModelTurn, ToolCall
from handshake.runtime.state import DependencyStatus, RunStatus
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
            block.get("type") == "text" and "User response:" in block.get("text", "")
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
    assert "Run completed" in captured.out
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
    runs = _store(home).list_runs()
    assert exit_code == 0
    assert "Input received; resuming run" in captured.out
    assert "Applied the user's choice." in captured.out
    assert len(runs) == 1
    assert runs[0].status is RunStatus.COMPLETED
    assert runs[0].dependencies[0].status is DependencyStatus.SATISFIED
    session = _store(home).load_session(runs[0].session_id)
    assert "User response: navy" in session.messages[-2].content[0]["text"]


def test_wait_survives_noninteractive_exit_interrupt_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _WaitThenFinishModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))

    first_exit = cli.main(["run", str(tmp_path), "style the button"])
    run = _store(home).list_runs()[0]
    session_id = run.session_id

    assert first_exit == 2
    assert run.status is RunStatus.WAITING
    assert cli.main(["list"]) == 0
    assert cli.main(["inspect", session_id]) == 0
    assert cli.main(["inspect-run", run.run_id]) == 0
    listing = capsys.readouterr().out
    assert run.run_id in listing
    assert "waiting" in listing
    assert "Which button color" in listing

    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert cli.main(["resume", session_id]) == 130
    assert _store(home).load_run(run.run_id).status is RunStatus.WAITING

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "navy" if prompt == "Response: " else "/exit",
    )
    assert cli.main(["resume", session_id]) == 0
    resumed = _store(home).load_run(run.run_id)
    resumed_session = _store(home).load_session(session_id)
    assert resumed.status is RunStatus.COMPLETED
    assert resumed_session.messages[-3].content[0]["tool_use_id"] == "ask-color"
    assert "User response: navy" in resumed_session.messages[-2].content[0]["text"]


def test_interactive_session_creates_followup_run_with_full_history(
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
    runs = _store(home).list_runs(session.session_id)
    assert exit_code == 0
    assert [run.goal for run in runs] == ["first request", "second request"]
    assert runs[1].previous_run_id == runs[0].run_id
    assert runs[1].max_steps == 9
    assert runs[1].model_turns == 1
    messages = second_model.contexts[0].messages
    assert [message.run_id for message in messages] == [
        runs[0].run_id,
        runs[0].run_id,
        runs[1].run_id,
    ]
    assert messages[0].content[0]["text"] == "first request"
    assert messages[1].content[0]["text"] == "first result"
    assert messages[2].content[0]["text"] == "second request"
    assert "Earlier tasks" not in second_model.contexts[0].system_prompt
