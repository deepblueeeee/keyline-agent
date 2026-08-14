"""Non-secret provider configuration and environment lookup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from handshake.runtime.errors import HandshakeError


@dataclass(slots=True, frozen=True)
class DeepSeekConfig:
    base_url: str = "https://api.deepseek.com/anthropic"
    model: str = "deepseek-v4-flash"
    max_tokens: int = 8192
    timeout_seconds: float = 120
    max_retries: int = 2

    @classmethod
    def from_environment(cls, *, model_override: str | None = None) -> DeepSeekConfig:
        return cls(
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"
            ),
            model=model_override
            or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )


def read_deepseek_api_key() -> str:
    """Read the key without placing it in normal task/config state."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HandshakeError("DEEPSEEK_API_KEY is required for real-model runs")
    return api_key


def handshake_home() -> Path:
    """Return the local durable-state directory without reading any secret."""
    configured = os.environ.get("HANDSHAKE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".handshake").resolve()
