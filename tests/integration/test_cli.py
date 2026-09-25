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
from handshake.config import credentials as provider_credentials
from handshake.config.approval import ApprovalMode
from handshake.context import AgentContextManager
from handshake.models.protocol import ModelContext, ModelTurn, ToolCall
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


class _SecretFlagshipModel:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.contexts: list[ModelContext] = []

    def complete(self, context: ModelContext, tools: tuple[Any, ...]) -> ModelTurn:
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
                if block.get("type") == "text"
                and "Secret handle:" in str(block.get("text"))
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


def test_real_run_requires_api_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
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
        "[model]\nmax_tokens = 4096\n"
        "[context]\ncontext_window_tokens = 32768\n",
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
    monkeypatch.setattr(cli, "AgentContextManager", manager_factory)

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
    config_path.write_text(
        "[context]\ncontext_window_tokens = 131072\n", encoding="utf-8"
    )
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    monkeypatch.setattr(cli, "_create_real_model", lambda _name: _WaitThenFinishModel())
    monkeypatch.setattr(sys, "stdin", _TTY(False))
    policies: list[Any] = []
    context_manager = AgentContextManager

    def manager_factory(*args: Any, **kwargs: Any) -> Any:
        policies.append(kwargs.get("policy"))
        return context_manager(*args, **kwargs)

    monkeypatch.setattr(cli, "AgentContextManager", manager_factory)

    assert cli.main(["run", str(tmp_path), "style the button"]) == 2
    session_id = _store(home).list_sessions()[0].session_id
    config_path.write_text(
        "[context]\ncontext_window_tokens = 65536\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "stdin", _TTY(True))
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "navy" if prompt == "Response: " else "/exit",
    )

    assert cli.main(["resume", session_id]) == 0
    assert [policy.context_window_tokens for policy in policies] == [131072, 65536]


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
        warnings.warn(
            "terminal echo would be enabled", cli.getpass.GetPassWarning, stacklevel=2
        )
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
    (tmp_path / "integration_check.py").write_text(
        integration_script, encoding="utf-8"
    )
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
    try:
        exit_code = cli.main(
            ["run", str(tmp_path), "run the authenticated integration test"]
        )
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
