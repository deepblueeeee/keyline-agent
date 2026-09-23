"""Local paths for durable Handshake state and artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def handshake_home() -> Path:
    """Return the local durable-state directory without reading any secret."""
    configured = os.environ.get("HANDSHAKE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".handshake").resolve()


def artifact_directory() -> Path:
    """Return the private directory backing model-readable artifact references."""
    return handshake_home() / "artifacts"
