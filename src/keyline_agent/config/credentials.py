"""Private, user-local storage for the DeepSeek credential."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path

from keyline_agent.errors import KeylineError

_MAX_KEY_BYTES = 16_384


def _credential_path() -> Path:
    # Deliberately independent of KEYLINE_HOME, which may point into a workspace.
    return Path.home() / ".keyline-agent" / "credentials" / "deepseek_api_key"


def _check_directory(path: Path, *, private: bool) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise KeylineError("Cannot inspect the local credential directory") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & (0o077 if private else 0o022):
        raise KeylineError("Local credential directory has unsafe permissions")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise KeylineError("Local credential directory is not owned by this user")
    return True


def _check_parent(path: Path, *, create: bool) -> bool:
    root = path.parent.parent
    if create:
        try:
            root.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            raise KeylineError("Cannot create the local credential directory") from None
    if not _check_directory(root, private=False):
        return False
    if create:
        try:
            path.parent.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            raise KeylineError("Cannot create the local credential directory") from None
    return _check_directory(path.parent, private=True)


def save_deepseek_api_key(value: str) -> None:
    """Atomically replace the saved key; never accept it through CLI arguments."""
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise KeylineError("DeepSeek API key must be a non-empty single-line value")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise KeylineError("DeepSeek API key is not valid UTF-8") from None
    if len(encoded) > _MAX_KEY_BYTES:
        raise KeylineError("DeepSeek API key is too long")

    path = _credential_path()
    if not _check_parent(path, create=True):
        raise KeylineError("Cannot create the local credential directory")
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    except OSError:
        raise KeylineError("Cannot inspect the saved DeepSeek API key") from None
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise KeylineError("Saved DeepSeek API key path is not a regular file")

    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".deepseek-key-", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        raise KeylineError("Cannot save the local DeepSeek API key") from None
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


def read_saved_deepseek_api_key() -> str | None:
    """Read a private regular file, rejecting symlinks and loose permissions."""
    path = _credential_path()
    if not _check_parent(path, create=False):
        return None
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise KeylineError("Cannot inspect the saved DeepSeek API key") from None
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise KeylineError("Saved DeepSeek API key must be a private regular file (mode 0600)")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise KeylineError("Saved DeepSeek API key is not owned by this user")

    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_mode & 0o077:
                raise KeylineError("Saved DeepSeek API key has unsafe permissions")
            if hasattr(os, "getuid") and opened.st_uid != os.getuid():
                raise KeylineError("Saved DeepSeek API key is not owned by this user")
            raw = stream.read(_MAX_KEY_BYTES + 1)
    except OSError:
        raise KeylineError("Cannot read the local DeepSeek API key") from None
    if len(raw) > _MAX_KEY_BYTES:
        raise KeylineError("Saved DeepSeek API key is too long")
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise KeylineError("Saved DeepSeek API key is not valid UTF-8") from None
    if not value or "\n" in value or "\r" in value:
        raise KeylineError("Saved DeepSeek API key is empty or malformed")
    return value
