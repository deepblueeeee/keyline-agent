"""DeepSeek provider settings and credential lookup."""

from __future__ import annotations

import os
from dataclasses import dataclass

from handshake.config.credentials import read_saved_deepseek_api_key
from handshake.errors import HandshakeError


@dataclass(slots=True, frozen=True)
class DeepSeekConfig:
    base_url: str = "https://api.deepseek.com/anthropic"
    model: str = "deepseek-v4-flash"
    max_tokens: int = 16_384
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
    """Prefer an environment override, then the private user-local key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        return api_key
    saved = read_saved_deepseek_api_key()
    if saved is not None:
        return saved
    raise HandshakeError(
        "DeepSeek API key is required: set DEEPSEEK_API_KEY or run "
        "'handshake config set-api-key'"
    )
