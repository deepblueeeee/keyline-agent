from __future__ import annotations

from pathlib import Path

import pytest

from handshake.config import credentials as provider_credentials
from handshake.config import read_deepseek_api_key
from handshake.runtime.errors import HandshakeError


@pytest.fixture
def credential_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".handshake" / "credentials" / "deepseek_api_key"
    monkeypatch.setattr(provider_credentials, "_credential_path", lambda: path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    return path


def test_save_and_read_key_with_private_permissions(credential_path: Path) -> None:
    provider_credentials.save_deepseek_api_key("first-canary-key")

    assert credential_path.read_text(encoding="utf-8") == "first-canary-key"
    assert credential_path.stat().st_mode & 0o777 == 0o600
    assert credential_path.parent.stat().st_mode & 0o777 == 0o700
    assert read_deepseek_api_key() == "first-canary-key"

    provider_credentials.save_deepseek_api_key("second-canary-key")
    assert read_deepseek_api_key() == "second-canary-key"
    assert not list(credential_path.parent.glob(".deepseek-key-*"))


def test_environment_key_takes_precedence(
    credential_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider_credentials.save_deepseek_api_key("saved-canary-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-canary-key")

    assert read_deepseek_api_key() == "environment-canary-key"


def test_missing_key_gives_setup_instruction(credential_path: Path) -> None:
    assert not credential_path.exists()

    with pytest.raises(HandshakeError, match="handshake config set-api-key"):
        read_deepseek_api_key()


def test_saved_key_rejects_loose_permissions_and_symlinks(credential_path: Path) -> None:
    provider_credentials.save_deepseek_api_key("private-canary-key")
    credential_path.chmod(0o644)

    with pytest.raises(HandshakeError, match="mode 0600") as captured:
        provider_credentials.read_saved_deepseek_api_key()
    assert "private-canary-key" not in str(captured.value)

    credential_path.unlink()
    other = credential_path.parent / "other"
    other.write_text("other-canary-key", encoding="utf-8")
    credential_path.symlink_to(other)
    with pytest.raises(HandshakeError, match="private regular file"):
        provider_credentials.read_saved_deepseek_api_key()
    with pytest.raises(HandshakeError, match="not a regular file"):
        provider_credentials.save_deepseek_api_key("replacement-canary-key")
    assert other.read_text(encoding="utf-8") == "other-canary-key"


def test_invalid_input_is_not_written(credential_path: Path) -> None:
    for value in ("", " leading-space", "two\nlines"):
        with pytest.raises(HandshakeError, match="single-line"):
            provider_credentials.save_deepseek_api_key(value)
    assert not credential_path.exists()


def test_private_directory_must_not_be_world_accessible(credential_path: Path) -> None:
    provider_credentials.save_deepseek_api_key("private-canary-key")
    credential_path.parent.chmod(0o755)

    with pytest.raises(HandshakeError, match="unsafe permissions"):
        provider_credentials.read_saved_deepseek_api_key()


def test_saved_key_does_not_follow_credential_directory_symlink(
    credential_path: Path,
) -> None:
    elsewhere = credential_path.parent.parent / "elsewhere"
    elsewhere.mkdir(parents=True, mode=0o700)
    credential_path.parent.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(HandshakeError, match="unsafe permissions"):
        provider_credentials.save_deepseek_api_key("private-canary-key")
    assert not (elsewhere / credential_path.name).exists()
