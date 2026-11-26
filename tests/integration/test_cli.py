from __future__ import annotations

import re
import subprocess
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import handshake.cli as cli
import handshake.runtime.loop as runtime_loop
from handshake.config import credentials as provider_credentials
from handshake.config.approval import ApprovalMode
from handshake.context import AgentContextManager
from handshake.models.protocol import ModelContext, ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.state import (
    DependencyStatus,
    RunStatus,
    SecretHandleStatus,
)
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry
from tests.history_support import history


class _TTY:
    def __init__(self, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_home = tmp_path.parent / f"{tmp_path.name}-user-home"
    user_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: user_home)


class _WaitThenFinishModel:
    async def complete(self, context: ModelContext, tools: tuple[Any, ...]) -> ModelTurn:
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


class _MultipleWaitThenFinishModel:
    async def complete(self, context: ModelContext, _tools: tuple[Any, ...]) -> ModelTurn:
        answers = [
            str(block["text"])
            for message in context.messages
            for block in message.content
            if block.get("type") == "text" and "User response:" in str(block.get("text", ""))
        ]
        if answers:
            assert len(answers) == 2
            return ModelTurn(
                content=({"type": "text", "text": "Applied both choices."},),
                stop_reason="end_turn",
            )
        calls = (
            ToolCall("ask-color", "wait_for_text", {"prompt": "Which color?"}),
            ToolCall("ask-size", "wait_for_text", {"prompt": "Which size?"}),
        )
        return ModelTurn(
            content=tuple(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in calls
            ),
            tool_calls=calls,
            stop_reason="tool_use",
        )


class _DelegateWaitThenFinishModel:
    async def complete(self, context: ModelContext, _tools: tuple[Any, ...]) -> ModelTurn:
        child = "Explore the workspace" in context.system_prompt
        if child:
            answered = any(
                "User response:" in str(block.get("text", ""))
                for message in context.messages
                for block in message.content
            )
            if answered:
                return ModelTurn(
                    content=({"type": "text", "text": "Found module A"},),
                    stop_reason="end_turn",
                )
            call = ToolCall("child-wait", "wait_for_text", {"prompt": "Which module?"})
        else:
            returned = any(
                block.get("type") == "tool_result"
                for message in context.messages
                for block in message.content
            )
            if returned:
                return ModelTurn(
                    content=({"type": "text", "text": "Used child result"},),
                    stop_reason="end_turn",
                )
            call = ToolCall(
                "delegate-1",
                "delegate",
                {"agent_spec_id": "explorer", "goal": "Find the requested module"},
            )
        return ModelTurn(
            content=(
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

    async def complete(self, context: ModelContext, _tools: tuple[Any, ...]) -> ModelTurn:
        self.contexts.append(context)
        return ModelTurn(
            content=({"type": "text", "text": self.summary},),
            stop_reason="end_turn",
        )


class _SecretFlagshipModel:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.contexts: list[ModelContext] = []

    async def complete(self, context: ModelContext, tools: tuple[Any, ...]) -> ModelTurn:
        self.contexts.append(context)
        assert any(tool.name == "wait_for_secret" for tool in tools)
        assert any(tool.name == "run_command_with_secret" for tool in tools)
        if any(
            block.get("tool_use_id") == "secure-test"
            for message in context.messages
            for block in message.content
        ):
            return ModelTurn(
                content=({"type": "text", "text": "Authenticated test passed."},),
                stop_reason="end_turn",
            )
        continuation = next(
            (
                str(block.get("text"))
                for message in reversed(context.messages)
                for block in message.content
                if block.get("type") == "text" and "Secret handle:" in str(block.get("text"))
            ),
            None,
        )
        if continuation is None:
            call = ToolCall(
                "ask-secret",
                "wait_for_secret",
                {
                    "prompt": "Enter the local integration API key",
                    "purpose": "authenticate the local Mock API test",
                    "environment_variable": "TEST_API_KEY",
                },
            )
        else:
            match = re.search(r"Secret handle: (secret_[^\s]+)", continuation)
            assert match is not None
            call = ToolCall(
                "secure-test",
                "run_command_with_secret",
                {
                    "secret_handle": match.group(1),
                    "argv": [sys.executable, "integration_check.py", self.api_url],
                },
            )
        return ModelTurn(
            content=(
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


def _store(home: Path) -> SQLiteStateStore:
    return SQLiteStateStore(home / "handshake.db")


def test_demo_runs_without_api_key(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = cli.main(["demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Run completed" in captured.out
    assert "verified the fixture tests pass" in captured.out


def test_real_run_requires_api_key(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        provider_credentials,
        "_credential_path",
        lambda: tmp_path / ".handshake" / "credentials" / "deepseek_api_key",
    )

    exit_code = cli.main(["run", str(tmp_path), "inspect the repository"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "handshake config set-api-key" in captured.err


def test_config_show_displays_effective_nonsecret_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "state"
    config_home = Path.home() / ".handshake"
    config_home.mkdir(mode=0o700)
    (config_home / "config.toml").write_text(
        "[model]\nmax_tokens = 4096\n[context]\ncontext_window_tokens = 131072\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-print-this-key")

    assert cli.main(["config", "show"]) == 0

    captured = capsys.readouterr()
    assert f"Configuration: {config_home / 'config.toml'}" in captured.out
    assert "max_tokens = 4096" in captured.out
    assert "context_window_tokens = 131072" in captured.out
    assert 'mode = "always_ask"' in captured.out
    assert "pressure_trigger_ratio" not in captured.out
    assert "reserved_output_tokens" not in captured.out
    assert "never-print-this-key" not in captured.out + captured.err


def test_command_approval_modes_apply_with_noninteractive_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    cautious = create_default_registry(cli._approval_callback(ApprovalMode.DANGEROUS_ONLY))
    unrestricted = create_default_registry(cli._approval_callback(ApprovalMode.ALLOW_ALL))

    assert cautious.execute("run_shell", tmp_path, {"argv": ["pwd"]}).ok
    denied = cautious.execute(
        "run_shell", tmp_path, {"argv": [sys.executable, "-c", "print('ran')"]}
    )
    assert not denied.ok
    assert "denied" in denied.stderr.lower()
    approved = unrestricted.execute(
        "run_shell", tmp_path, {"argv": [sys.executable, "-c", "print('ran')"]}
    )
    assert approved.ok
    assert approved.stdout.strip() == "ran"


def test_run_uses_configured_provider_and_context_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "state"
    config_home = Path.home() / ".handshake"
    config_home.mkdir(mode=0o700)
    (config_home / "config.toml").write_text(
        "[model]\nmax_tokens = 4096\n[context]\ncontext_window_tokens = 32768\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-provider-key")
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    supplied: list[dict[str, Any]] = []
    policies: list[Any] = []
    context_manager = AgentContextManager

    def model_factory(**kwargs: Any) -> _RecordingFinishModel:
        supplied.append(kwargs)
        return _RecordingFinishModel("configured run finished")

    def manager_factory(*args: Any, **kwargs: Any) -> Any:
        policies.append(kwargs.get("policy"))
        return context_manager(*args, **kwargs)

    monkeypatch.setattr(cli, "DeepSeekAnthropicModel", model_factory)
    monkeypatch.setattr(runtime_loop, "AgentContextManager", manager_factory)

    assert cli.main(["run", str(tmp_path), "inspect the repository"]) == 0

    assert supplied[0]["max_tokens"] == 4096
    assert policies[0].context_window_tokens == 32768
    assert policies[0].reserved_output_tokens == 4096
    assert policies[0].condenser_input_tokens == 28672


def test_resume_reloads_context_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "state"
    config_home = Path.home() / ".handshake"
    config_home.mkdir(mode=0o700)
    config_path = config_home / "config.toml"
    config_path.write_text("[context]\ncontext_window_tokens = 131072\n", encoding="utf-8")
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setenv("DEEPSEEK_MODEL", "old-model")
    model_names: list[str] = []

    def create_model(name: str) -> _WaitThenFinishModel:
        model_names.append(name)
        return _WaitThenFinishModel()

    monkeypatch.setattr(cli, "_create_real_model", create_model)
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    policies: list[Any] = []
    context_manager = AgentContextManager

    def manager_factory(*args: Any, **kwargs: Any) -> Any:
        policies.append(kwargs.get("policy"))
        return context_manager(*args, **kwargs)

    monkeypatch.setattr(runtime_loop, "AgentContextManager", manager_factory)

    assert cli.main(["run", str(tmp_path), "style the button"]) == 2
    stored_run = _store(home).list_runs()[0]
    assert stored_run.agent_spec_id == "main"
    session_id = stored_run.session_id
    config_path.write_text("[context]\ncontext_window_tokens = 65536\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_MODEL", "new-model")
    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "navy" if prompt == "Response: " else "/exit",
    )

    assert cli.main(["resume", session_id]) == 0
    assert [policy.context_window_tokens for policy in policies] == [131072, 65536]
    assert model_names == ["old-model", "new-model"]
    assert _store(home).load_run(stored_run.run_id).model_name == "new-model"


def test_config_set_api_key_uses_hidden_input_without_echo_or_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "local-provider-canary-key"
    credential_path = tmp_path / ".handshake" / "credentials" / "deepseek_api_key"
    monkeypatch.setattr(provider_credentials, "_credential_path", lambda: credential_path)
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: canary)

    exit_code = cli.main(["config", "set-api-key"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "saved" in captured.out
    assert canary not in captured.out + captured.err
    assert credential_path.read_text(encoding="utf-8") == canary
    assert not (tmp_path / ".handshake" / "handshake.db").exists()


def test_config_set_api_key_rejects_noninteractive_stdin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_path = tmp_path / ".handshake" / "credentials" / "deepseek_api_key"
    monkeypatch.setattr(provider_credentials, "_credential_path", lambda: credential_path)
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))

    assert cli.main(["config", "set-api-key"]) == 1
    assert "interactive terminal" in capsys.readouterr().err
    assert not credential_path.exists()


def test_config_set_api_key_rejects_visible_getpass_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_path = tmp_path / ".handshake" / "credentials" / "deepseek_api_key"
    monkeypatch.setattr(provider_credentials, "_credential_path", lambda: credential_path)
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))

    def visible_fallback(_prompt: str) -> str:
        warnings.warn("terminal echo would be enabled", cli.getpass.GetPassWarning, stacklevel=2)
        return "must-not-be-saved"

    monkeypatch.setattr(cli.getpass, "getpass", visible_fallback)

    assert cli.main(["config", "set-api-key"]) == 1
    captured = capsys.readouterr()
    assert "Hidden API key input is unavailable" in captured.err
    assert "must-not-be-saved" not in captured.out + captured.err
    assert not credential_path.exists()


def test_real_run_reads_saved_key_without_persisting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "saved-provider-canary-key"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "state"
    credential_path = tmp_path / ".handshake" / "credentials" / "deepseek_api_key"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(provider_credentials, "_credential_path", lambda: credential_path)
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))
    provider_credentials.save_deepseek_api_key(canary)
    supplied_keys: list[str] = []

    def model_factory(**kwargs: Any) -> _RecordingFinishModel:
        supplied_keys.append(kwargs["api_key"])
        return _RecordingFinishModel("Completed with saved credential.")

    monkeypatch.setattr(cli, "DeepSeekAnthropicModel", model_factory)

    assert cli.main(["run", str(workspace), "inspect the repository"]) == 0
    captured = capsys.readouterr()
    assert supplied_keys == [canary]
    assert "Completed with saved credential." in captured.out
    assert canary not in captured.out + captured.err
    assert canary.encode() not in (home / "handshake.db").read_bytes()


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
    store = _store(home)
    session = store.load_session(runs[0].session_id)
    assert "User response: navy" in history(store, session)[-2].content[0]["text"]


def test_cli_reopens_the_same_run_after_disabling_its_turn_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))
    call = ToolCall("write", "write_file", {"path": "result.txt", "content": "first"})
    turn = ModelTurn(({
        "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments,
    },), (call,), "tool_use")
    monkeypatch.setattr(cli, "_create_real_model", lambda _: ScriptedModel([turn]))
    assert cli.main(["run", str(tmp_path), "write", "--max-steps", "1"]) == 2
    store = _store(home)
    (paused,) = store.list_runs()
    result = tmp_path / "result.txt"
    assert paused.status is RunStatus.PAUSED and result.read_text() == "first"
    result.write_text("kept across resume")

    monkeypatch.setattr(cli, "_create_real_model", lambda _: pytest.fail("Run is still paused"))
    resume = ["resume", paused.session_id]
    assert cli.main(resume) == 2
    assert cli.main([*resume, "--max-steps", "1"]) == 2
    assert store.load_run(paused.run_id).status is RunStatus.PAUSED

    model = ScriptedModel([_finish("done")])
    monkeypatch.setattr(cli, "_create_real_model", lambda _: model)
    assert cli.main([*resume, "--max-steps", "0", "--run-id", paused.run_id]) == 0
    (completed,) = store.list_runs()
    assert completed.run_id == paused.run_id and completed.status is RunStatus.COMPLETED
    assert completed.model_turns == 2 and completed.max_steps == 0
    assert result.read_text() == "kept across resume"
    assert model.contexts[0].messages[-1].content[0]["tool_use_id"] == call.id


def test_multiple_waits_resume_after_partial_input_across_cli_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _MultipleWaitThenFinishModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    answers = iter(("navy", None))

    def first_inputs(_prompt: str) -> str:
        answer = next(answers)
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", first_inputs)
    assert cli.main(["run", str(tmp_path), "configure button"]) == 2
    stored_run = _store(home).list_runs()[0]
    assert stored_run.status is RunStatus.WAITING
    assert [dependency.status for dependency in stored_run.dependencies] == [
        DependencyStatus.SATISFIED,
        DependencyStatus.OPEN,
    ]
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "large" if prompt == "Response: " else "/exit",
    )

    assert cli.main(["resume", stored_run.session_id]) == 0
    completed = _store(home).load_run(stored_run.run_id)
    assert completed.status is RunStatus.COMPLETED
    assert completed.final_summary == "Applied both choices."
    assert all(
        dependency.status is DependencyStatus.SATISFIED for dependency in completed.dependencies
    )


def test_delegated_child_wait_resumes_parent_across_cli_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _DelegateWaitThenFinishModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))

    assert cli.main(["run", str(tmp_path), "Locate a module"]) == 2
    store = _store(home)
    parent, child = store.list_runs()
    assert parent.status is RunStatus.WAITING
    assert child.status is RunStatus.WAITING
    assert child.parent_run_id == parent.run_id
    assert child.context_scope_id != parent.context_scope_id

    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "A" if prompt == "Response: " else "/exit",
    )
    assert cli.main(["resume", parent.session_id]) == 0
    store = _store(home)
    parent, child = store.list_runs(parent.session_id)
    assert parent.status is RunStatus.COMPLETED
    assert parent.final_summary == "Used child result"
    assert child.status is RunStatus.COMPLETED
    assert child.final_summary == "Found module A"
    assert len(tuple(store.read_messages(parent.context_scope_id))) == 4
    assert len(tuple(store.read_messages(child.context_scope_id))) == 5


def test_batch_wait_does_not_block_sibling_and_joins_after_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BatchModel:
        async def complete(self, context: ModelContext, _tools: tuple[Any, ...]) -> ModelTurn:
            if context.goal == "Explore":
                if any(
                    "User response:" in str(block.get("text", ""))
                    for message in context.messages for block in message.content
                ):
                    return ModelTurn(
                        content=({"type": "text", "text": "Explorer done"},),
                        stop_reason="end_turn",
                    )
                calls = (
                    ToolCall("ask", "wait_for_text", {"prompt": "Which file?"}),
                )
            elif context.goal == "Review":
                return ModelTurn(
                    content=({"type": "text", "text": "Reviewer done"},),
                    stop_reason="end_turn",
                )
            elif any(
                block.get("type") == "tool_result"
                for message in context.messages for block in message.content
            ):
                return ModelTurn(
                    content=({"type": "text", "text": "Used both child results"},),
                    stop_reason="end_turn",
                )
            else:
                calls = (
                    ToolCall(
                        "explore", "delegate",
                        {"agent_spec_id": "explorer", "goal": "Explore"},
                    ),
                    ToolCall(
                        "review", "delegate",
                        {"agent_spec_id": "reviewer", "goal": "Review"},
                    ),
                )
            return ModelTurn(
                content=tuple(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    for call in calls
                ),
                tool_calls=calls,
                stop_reason="tool_use",
            )

    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: BatchModel())
    monkeypatch.setattr(cli.sys, "stdin", _TTY(False))
    assert cli.main(["run", str(tmp_path), "Investigate"]) == 2
    store = _store(home)
    runs = store.list_runs()
    parent = next(run for run in runs if run.parent_run_id is None)
    parent, explorer, reviewer = store.list_runs(parent.session_id)
    assert [run.status for run in (parent, explorer, reviewer)] == [
        RunStatus.WAITING, RunStatus.WAITING, RunStatus.COMPLETED,
    ]
    assert len(tuple(store.read_messages(parent.context_scope_id))) == 2

    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "A" if prompt == "Response: " else "/exit",
    )
    assert cli.main(["resume", parent.session_id]) == 0
    store = _store(home)
    parent, explorer, reviewer = store.list_runs(parent.session_id)
    assert parent.status is RunStatus.COMPLETED
    assert parent.final_summary == "Used both child results"
    assert explorer.status is RunStatus.COMPLETED
    assert reviewer.status is RunStatus.COMPLETED
    messages = tuple(store.read_messages(parent.context_scope_id))
    assert [block["tool_use_id"] for block in messages[2].message.content] == [
        "explore", "review",
    ]


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
    store = _store(home)
    resumed_session = store.load_session(session_id)
    resumed_messages = history(store, resumed_session)
    assert resumed.status is RunStatus.COMPLETED
    assert resumed_messages[-3].content[0]["tool_use_id"] == "ask-color"
    assert "User response: navy" in resumed_messages[-2].content[0]["text"]


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

    exit_code = cli.main(["run", str(tmp_path), "first request", "--max-steps", "9"])

    session = _store(home).list_sessions()[0]
    runs = _store(home).list_runs(session.session_id)
    assert exit_code == 0
    assert [run.goal for run in runs] == ["first request", "second request"]
    assert runs[1].context_scope_id == runs[0].context_scope_id
    assert runs[0].max_steps == 9 and runs[1].max_steps == 0
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


@pytest.mark.parametrize(
    "approval_mode",
    (ApprovalMode.ALWAYS_ASK, ApprovalMode.DANGEROUS_ONLY, ApprovalMode.ALLOW_ALL),
)
def test_secret_flagship_uses_getpass_secret_command_and_leaks_no_canary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    approval_mode: ApprovalMode,
) -> None:
    canary = "flagship-canary-secret-value"

    class MockApiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorized = self.headers.get("Authorization") == f"Bearer {canary}"
            self.send_response(200 if authorized else 403)
            self.end_headers()
            self.wfile.write(b"authorized" if authorized else b"denied")

        def log_message(self, _format: str, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockApiHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    api_url = f"http://127.0.0.1:{server.server_port}/verify"
    integration_script = """import os
import sys
import urllib.request

request = urllib.request.Request(
    sys.argv[1],
    headers={"Authorization": f"Bearer {os.environ['TEST_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=3) as response:
    print(response.read().decode("utf-8"))
"""
    (tmp_path / "integration_check.py").write_text(integration_script, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Handshake", "-c", "user.email=test@example.com", "add", "."],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Handshake",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    home = tmp_path / "state"
    if approval_mode is not ApprovalMode.ALWAYS_ASK:
        config_home = Path.home() / ".handshake"
        config_home.mkdir(mode=0o700)
        (config_home / "config.toml").write_text(
            f"[approval]\nmode = '{approval_mode.value}'\n", encoding="utf-8"
        )
    model = _SecretFlagshipModel(api_url)
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: model)
    monkeypatch.setattr(cli.sys, "stdin", _TTY(True))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: canary)

    approval_prompts: list[str] = []

    def answer(prompt: str) -> str:
        if "Execute this command once?" in prompt:
            approval_prompts.append(prompt)
            return "y"
        return "/exit"

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(cli, "_read_approval_answer", answer)
    try:
        exit_code = cli.main(["run", str(tmp_path), "run the authenticated integration test"])
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)

    captured = capsys.readouterr()
    state_store = _store(home)
    run = state_store.list_runs()[0]
    session = state_store.load_session(run.session_id)
    workspace_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file() and ".git" not in path.parts and path != state_store.database_path
    )
    git_diff = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert exit_code == 0
    assert run.status is RunStatus.COMPLETED
    assert len(approval_prompts) == (0 if approval_mode is ApprovalMode.ALLOW_ALL else 1)
    assert run.secret_handles[0].status is SecretHandleStatus.CONSUMED
    assert "Authenticated test passed." in captured.out
    assert canary not in captured.out
    assert canary not in captured.err
    assert canary not in repr(model.contexts)
    assert canary not in repr(history(state_store, session))
    assert canary.encode() not in state_store.database_path.read_bytes()
    assert canary not in workspace_text
    assert canary not in git_diff


def test_sigint_during_approval_exits_without_starting_command(tmp_path: Path) -> None:
    import os
    import select
    import signal
    import time

    script = tmp_path / "approval_scenario.py"
    script.write_text('''
import os
import sys
from pathlib import Path
from unittest.mock import patch
import handshake.cli as cli
from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel

workspace = Path(sys.argv[1])
os.environ["HANDSHAKE_HOME"] = str(workspace / "state")
call = ToolCall("shell", "run_shell", {"argv": [
    sys.executable, "-c",
    "from pathlib import Path; Path('command-started').write_text('started')",
]})
model = ScriptedModel([
    ModelTurn(content=({"type": "tool_use", "id": call.id, "name": call.name,
                        "input": call.arguments},), tool_calls=(call,), stop_reason="tool_use"),
    ModelTurn(content=({"type": "text", "text": "done"},), stop_reason="end_turn"),
])
cli._create_real_model = lambda _name: model
sys.stdin.isatty = lambda: True
with patch.object(Path, "home", return_value=workspace):
    raise SystemExit(cli.main(["run", str(workspace), "test approval cancellation"]))
''', encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", str(script), str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    transcript = b""
    try:
        deadline = time.monotonic() + 5
        while b"Execute this command once?" not in transcript and time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.1)
            if ready:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                transcript += chunk
        assert b"Execute this command once?" in transcript, transcript.decode()
        # A partial line must not make the cancellation-aware reader block on readline().
        assert process.stdin is not None
        process.stdin.write(b"y")
        process.stdin.flush()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=3) == 130
        assert not (tmp_path / "command-started").exists()
        store = _store(tmp_path / "state")
        run = store.list_runs()[0]
        assert run.status is RunStatus.CANCELLED
        assert not run.secret_handles
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.communicate()


def test_cli_cleanup_failure_does_not_replace_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    class ClosingModel(_RecordingFinishModel):
        async def aclose(self) -> None:
            raise RuntimeError("cleanup-sensitive-canary")

    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: ClosingModel("done"))
    assert cli.main(["run", str(tmp_path), "test cleanup"]) == 0
    assert _store(tmp_path / "state").list_runs()[0].status is RunStatus.COMPLETED
    assert "cleanup-sensitive-canary" not in caplog.text
    assert "Model resource cleanup failed (RuntimeError)" in caplog.text


def test_child_answer_resumes_while_sibling_keeps_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import asyncio

    prompting = threading.Event()
    sibling_worked = threading.Event()
    resumed = threading.Event()
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")

    class Model:
        async def complete(self, context, _tools) -> ModelTurn:
            if context.goal == "Explore":
                if any("User response:" in str(b.get("text", ""))
                       for m in context.messages for b in m.content):
                    resumed.set()
                    return _finish("Explorer finished")
                return cli._tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which file?"}))
            if context.goal == "Review":
                if any(b.get("type") == "tool_result"
                       for m in context.messages for b in m.content):
                    sibling_worked.set()
                    assert await asyncio.to_thread(resumed.wait, 2)
                    return _finish("Reviewer finished")
                assert await asyncio.to_thread(prompting.wait, 2)
                return cli._tool_turn(ToolCall("read", "read_file", {"path": "evidence.txt"}))
            if any(b.get("type") == "tool_result"
                   for m in context.messages for b in m.content):
                return _finish("Joined both")
            return cli._tool_turn(
                ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
                ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
            )

    def answer(prompt: str) -> str:
        if prompt != "Response: ":
            return "/exit"
        prompting.set()
        assert sibling_worked.wait(2), "Sibling tools must run while input is pending"
        return "evidence.txt"

    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: Model())
    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr("builtins.input", answer)
    assert cli.main(["run", str(tmp_path), "Investigate"]) == 0
    store = _store(tmp_path / "state")
    parent = next(run for run in store.list_runs() if run.parent_run_id is None)
    parent, explorer, reviewer = store.list_runs(parent.session_id)
    assert all(run.status is RunStatus.COMPLETED for run in (parent, explorer, reviewer))
    assert resumed.is_set()
    output = capsys.readouterr().out
    for run in (parent, explorer, reviewer):
        assert f"agent={run.agent_spec_id} run={run.run_id}" in output
    assert "status=waiting wait=text" in output
    assert (
        f"[agent=reviewer run={reviewer.run_id} status=running] Running tool: read_file"
    ) in output
    assert f"[agent=explorer run={explorer.run_id} status=running] Model turn 2" in output


def _finish(text: str) -> ModelTurn:
    return ModelTurn(content=({"type": "text", "text": text},), stop_reason="end_turn")


@pytest.mark.parametrize("decision", ("y", "n"))
def test_child_approval_wait_allows_sibling_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], decision: str,
) -> None:
    import asyncio
    from dataclasses import replace

    approval_pending = threading.Event()
    sibling_worked = threading.Event()
    marker = tmp_path / "approved-command"
    (tmp_path / "evidence.txt").write_text("evidence", encoding="utf-8")
    original_specs = cli._agent_specs

    def specs(settings, registry):
        result = original_specs(settings, registry)
        spec = result["explorer"]
        result["explorer"] = replace(spec, allowed_tools=spec.allowed_tools | {"run_shell"})
        return result

    class Model:
        async def complete(self, context, _tools) -> ModelTurn:
            tool_results = [b for m in context.messages for b in m.content
                            if b.get("type") == "tool_result"]
            if context.goal == "Explore":
                if tool_results:
                    assert bool(tool_results[-1].get("is_error")) == (decision == "n")
                    return _finish("Explorer finished")
                return cli._tool_turn(ToolCall("shell", "run_shell", {"argv": [
                    sys.executable, "-c",
                    "from pathlib import Path; Path('approved-command').write_text('ok')",
                ]}))
            if context.goal == "Review":
                if tool_results:
                    assert not marker.exists(), "Command must still be waiting for approval"
                    sibling_worked.set()
                    return _finish("Reviewer finished")
                assert await asyncio.to_thread(approval_pending.wait, 2)
                return cli._tool_turn(ToolCall("read", "read_file", {"path": "evidence.txt"}))
            if tool_results:
                return _finish("Joined both")
            return cli._tool_turn(
                ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
                ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
            )

    def answer(_prompt: str) -> str:
        approval_pending.set()
        assert sibling_worked.wait(2), "Approval must not occupy the workspace execution slot"
        return decision

    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: Model())
    monkeypatch.setattr(cli, "_agent_specs", specs)
    monkeypatch.setattr(cli, "_read_approval_answer", answer)
    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")
    assert cli.main(["run", str(tmp_path), "Investigate"]) == 0
    assert marker.exists() == (decision == "y")
    store = _store(tmp_path / "state")
    parent = next(run for run in store.list_runs() if run.parent_run_id is None)
    parent, explorer, reviewer = store.list_runs(parent.session_id)
    assert all(run.status is RunStatus.COMPLETED for run in (parent, explorer, reviewer))
    output = capsys.readouterr().out
    assert f"[agent=explorer run={explorer.run_id} status=waiting wait=approval]" in output
    assert (
        f"[agent=reviewer run={reviewer.run_id} status=running] Running tool: read_file"
    ) in output


def test_text_and_approval_prompts_share_one_input_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from dataclasses import replace

    text_open = threading.Event()
    approval_queued = threading.Event()
    text_finished = threading.Event()
    original_specs = cli._agent_specs
    requires_approval = cli.requires_approval

    def specs(settings, registry):
        result = original_specs(settings, registry)
        spec = result["reviewer"]
        result["reviewer"] = replace(spec, allowed_tools=spec.allowed_tools | {"run_shell"})
        return result

    def approval_policy(*args, **kwargs):
        approval_queued.set()
        return requires_approval(*args, **kwargs)

    class Model:
        async def complete(self, context, _tools) -> ModelTurn:
            if context.goal == "Explore":
                if any("User response:" in str(b.get("text", ""))
                       for m in context.messages for b in m.content):
                    return _finish("Explorer finished")
                return cli._tool_turn(ToolCall("ask", "wait_for_text", {"prompt": "Which color?"}))
            if context.goal == "Review":
                if any(b.get("type") == "tool_result"
                       for m in context.messages for b in m.content):
                    return _finish("Reviewer finished")
                assert await asyncio.to_thread(text_open.wait, 2)
                return cli._tool_turn(ToolCall("shell", "run_shell", {
                    "argv": [sys.executable, "-c", "print('approved')"],
                }))
            if any(b.get("type") == "tool_result"
                   for m in context.messages for b in m.content):
                return _finish("Joined both")
            return cli._tool_turn(
                ToolCall("explore", "delegate", {"agent_spec_id": "explorer", "goal": "Explore"}),
                ToolCall("review", "delegate", {"agent_spec_id": "reviewer", "goal": "Review"}),
            )

    def text_answer(prompt: str) -> str:
        if prompt != "Response: ":
            return "/exit"
        text_open.set()
        assert approval_queued.wait(2)
        text_finished.set()
        return "navy"

    def approval_answer(_prompt: str) -> str:
        assert text_finished.is_set(), "Approval must not read the text answer"
        return "y"

    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(cli, "_agent_specs", specs)
    monkeypatch.setattr(cli, "requires_approval", approval_policy)
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: Model())
    monkeypatch.setattr(cli, "_read_approval_answer", approval_answer)
    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr("builtins.input", text_answer)
    assert cli.main(["run", str(tmp_path), "Investigate"]) == 0
    store = _store(tmp_path / "state")
    explorer = next(run for run in store.list_runs() if run.agent_spec_id == "explorer")
    messages = [
        str(item.message.content) for item in store.read_messages(explorer.context_scope_id)
    ]
    assert any("User response: navy" in message for message in messages)
    assert not any("User response: y" in message for message in messages)


@pytest.mark.parametrize("before_tool_record", (False, True))
def test_resume_secret_command_waiting_for_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    before_tool_record: bool,
) -> None:
    import asyncio

    from handshake.inputs import EphemeralSecretStore
    from handshake.models.scripted import ScriptedModel
    from handshake.runtime.loop import continue_run, start_run
    from handshake.runtime.secret_dependencies import resolve_secret_dependency
    from handshake.runtime.state import Run, Session, ToolRunStatus
    from handshake.tools.secret_command import SecretCommandRunner
    from tests.history_support import TEST_AGENT_SPEC

    home = tmp_path / "state"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    store = _store(home)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    run = Run("Verify an authenticated command", session.session_id, "main")
    spec = cli.main_agent_spec(
        "scripted", TEST_AGENT_SPEC.allowed_tools, TEST_AGENT_SPEC.default_budget
    )
    marker = tmp_path / "executions.txt"
    registry = create_default_registry(lambda _cwd, _argv: True)
    secret = "replacement-secret-canary"
    approval_seen = asyncio.Event()

    async def wait_for_approval(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        approval_seen.set()
        await asyncio.Event().wait()
        return False

    async def interrupt_before_approval() -> None:
        memory = EphemeralSecretStore()
        runner = SecretCommandRunner(memory.consumer(), lambda *_: False, wait_for_approval)
        await start_run(
            session, run, ScriptedModel([cli._tool_turn(ToolCall("ask", "wait_for_secret", {
                "prompt": "Enter the test key", "purpose": "authenticate command",
                "environment_variable": "TEST_API_KEY",
            }))]), registry, store=store, agent_spec=spec, secret_command_runner=runner,
        )
        receipt = resolve_secret_dependency(
            session, run, run.open_dependencies()[0].dependency_id,
            "original-secret", memory.writer(),
        )
        call = ToolCall("authenticated", "run_command_with_secret", {
            "secret_handle": run.secret_handles[0].secret_handle,
            "argv": [sys.executable, "-c", (
                "import os; from pathlib import Path; "
                "assert os.environ['TEST_API_KEY']; "
                "Path('executions.txt').open('a').write('executed\\n'); "
                "print(os.environ['TEST_API_KEY'])"
            )],
        })
        operation = continue_run(
            session, run, ScriptedModel([cli._tool_turn(call)]), registry, store=store,
            agent_spec=spec, pending_messages=(receipt,), secret_command_runner=runner,
        )
        if before_tool_record:
            async def interrupt(*_args: object) -> None:
                raise asyncio.CancelledError()

            with monkeypatch.context() as patch:
                patch.setattr(runtime_loop, "_execute_tool_calls", interrupt)
                with pytest.raises(asyncio.CancelledError):
                    await operation
            return

        task = asyncio.create_task(operation)
        await asyncio.wait_for(approval_seen.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt_before_approval())
    before = store.load_run(run.run_id)
    assert [tool.status for tool in before.tool_runs] == (
        [] if before_tool_record else [ToolRunStatus.PENDING]
    )
    assert not marker.exists()
    message_count = store.message_count(run.context_scope_id)

    model = ScriptedModel([_finish("Verified original command")])
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: model)
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    assert cli.main(["resume", session.session_id]) == 2
    waiting = store.load_run(run.run_id)
    assert waiting.status is RunStatus.WAITING
    assert waiting.dependencies[0].status is DependencyStatus.LOST
    assert waiting.open_dependencies()[0].dependency_id != waiting.dependencies[0].dependency_id
    assert waiting.secret_handles[0].status is SecretHandleStatus.REQUESTED
    assert waiting.secret_handles[0].secret_handle == before.secret_handles[0].secret_handle
    assert waiting.tool_runs == before.tool_runs
    assert store.message_count(run.context_scope_id) == message_count
    assert not model.contexts

    def approve(_prompt: str) -> str:
        snapshot = store.load_run(run.run_id)
        assert not snapshot.open_dependencies()
        assert snapshot.tool_runs[0].status is ToolRunStatus.PENDING
        assert not marker.exists()
        return "y"

    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: secret)
    monkeypatch.setattr(cli, "_read_approval_answer", approve)
    monkeypatch.setattr("builtins.input", lambda _prompt: "/exit")
    assert cli.main(["resume", session.session_id]) == 0
    completed = store.load_run(run.run_id)
    assert completed.status is RunStatus.COMPLETED
    assert completed.secret_handles[0].status is SecretHandleStatus.CONSUMED
    assert len(completed.tool_runs) == 1
    result = completed.tool_runs[0]
    assert result.tool_call_id == "authenticated"
    if before.tool_runs:
        assert result.tool_run_id == before.tool_runs[0].tool_run_id
        assert result.arguments == before.tool_runs[0].arguments
    assert result.status is ToolRunStatus.SUCCEEDED
    assert marker.read_text() == "executed\n"
    transcript = history(store, session)
    assert transcript[-3].role == "assistant"
    assert transcript[-3].content[0]["id"] == "authenticated"
    assert transcript[-2].content[0]["tool_use_id"] == "authenticated"
    assert "[REDACTED]" in transcript[-2].content[0]["content"]
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + repr(model.contexts)
    assert secret.encode() not in store.database_path.read_bytes()
