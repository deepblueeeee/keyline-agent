from __future__ import annotations

from pathlib import Path

import pytest

from handshake.config.approval import ApprovalMode
from handshake.config.settings import load_runtime_settings, settings_path
from handshake.context.budget import ContextBudgetPolicy
from handshake.errors import HandshakeError


def _write_config(home: Path, content: str) -> None:
    home.mkdir(mode=0o700)
    (home / "config.toml").write_text(content, encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_defaults_and_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / ".handshake"
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "workspace-state"))

    settings = load_runtime_settings(model_override="chosen-model")

    assert settings_path() == home / "config.toml"
    assert settings.provider.model == "chosen-model"
    assert settings.provider.max_tokens == 16_384
    assert settings.context.context_window_tokens == 224_000
    assert settings.context.reserved_output_tokens == 16_384
    assert settings.context.pressure_trigger_ratio == 0.8
    assert settings.context.pressure_target_ratio == 0.5
    assert settings.approval_mode is ApprovalMode.ALWAYS_ASK


def test_only_public_token_limits_override_internal_defaults(
    tmp_path: Path,
) -> None:
    home = tmp_path / ".handshake"
    _write_config(
        home,
        """[model]
max_tokens = 4096

[context]
context_window_tokens = 32768
""",
    )

    settings = load_runtime_settings()
    defaults = ContextBudgetPolicy()

    assert settings.provider.max_tokens == 4096
    assert settings.context.reserved_output_tokens == 4096
    assert settings.context.hard_input_tokens == 28672
    assert settings.context.condenser_input_tokens == 28672
    assert settings.context.pressure_trigger_ratio == defaults.pressure_trigger_ratio
    assert settings.context.pressure_target_ratio == defaults.pressure_target_ratio
    assert settings.context.protected_recent_messages == defaults.protected_recent_messages
    assert settings.context.max_tool_result_tokens == defaults.max_tool_result_tokens


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[model]\nmax_tokens = true\n", "model.max_tokens"),
        ("[model]\nmax_tokens = 0\n", "model.max_tokens"),
        ("[context]\ncontext_window_tokens = 16000\n", "context_window_tokens"),
        ("[context]\ncontext_window_tokens = true\n", "context_window_tokens"),
        (
            "[model]\nmax_tokens = 8192\n[context]\ncontext_window_tokens = 8192\n",
            "must exceed",
        ),
        ("[context]\npressure_trigger_ratio = 0.9\n", "pressure_trigger_ratio"),
        ("[context]\nreserved_output_tokens = 20000\n", "reserved_output_tokens"),
        ("[context]\ncondenser_input_tokens = 20000\n", "condenser_input_tokens"),
        ("[context]\nunknown_key = 42\n", "unknown_key"),
        ("[approval]\nmode = 'unknown'\n", "approval.mode"),
        ("[approval]\nmode = true\n", "approval.mode"),
        ("[approval]\nextra = 1\n", "extra"),
        ("[context\n", "Cannot read configuration"),
    ],
)
def test_invalid_settings_fail_before_run(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    home = tmp_path / ".handshake"
    _write_config(home, content)

    with pytest.raises(HandshakeError, match=message):
        load_runtime_settings()


@pytest.mark.parametrize("mode", (ApprovalMode.DANGEROUS_ONLY, ApprovalMode.ALLOW_ALL))
def test_approval_mode_loads(tmp_path: Path, mode: ApprovalMode) -> None:
    _write_config(tmp_path / ".handshake", f"[approval]\nmode = '{mode.value}'\n")

    assert load_runtime_settings().approval_mode is mode


def test_config_file_must_not_be_workspace_redirect_or_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_state = tmp_path / "workspace-state"
    workspace_state.mkdir()
    (workspace_state / "config.toml").write_text(
        "[approval]\nmode = 'allow_all'\n", encoding="utf-8"
    )
    monkeypatch.setenv("HANDSHAKE_HOME", str(workspace_state))
    assert load_runtime_settings().approval_mode is ApprovalMode.ALWAYS_ASK

    home = tmp_path / ".handshake"
    home.mkdir(mode=0o700)
    (home / "config.toml").symlink_to(workspace_state / "config.toml")
    with pytest.raises(HandshakeError, match="unsafe permissions"):
        load_runtime_settings()


def test_non_utf8_config_reports_domain_error(tmp_path: Path) -> None:
    home = tmp_path / ".handshake"
    home.mkdir(mode=0o700)
    (home / "config.toml").write_bytes(b"\xff")

    with pytest.raises(HandshakeError, match="Cannot read configuration"):
        load_runtime_settings()
