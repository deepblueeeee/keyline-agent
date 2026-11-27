"""Local paths for durable Keyline state and artifacts."""

from __future__ import annotations

import os
from pathlib import Path

_STATE_DIR = ".keyline-agent"


def private_state_roots() -> tuple[Path, Path]:
    """Protect credentials left by the previous application as well as current state."""
    home = Path.home()
    return (home / _STATE_DIR).resolve(), (home / ".handshake").resolve()


def keyline_home() -> Path:
    """Return the local durable-state directory without reading any secret."""
    configured = os.environ.get("KEYLINE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / _STATE_DIR).resolve()


def artifact_directory() -> Path:
    """Return the private directory backing model-readable artifact references."""
    return keyline_home() / "artifacts"
