"""Deterministic local estimates for model-visible context pressure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from handshake.models.base import ToolDefinition
from handshake.models.protocol import ModelContext


class TokenEstimator(Protocol):
    """Estimate tokens without making a provider request."""

    def estimate_text(self, text: str) -> int:
        """Return a deterministic token estimate for text."""
        ...


class Utf8TokenEstimator:
    """Conservatively approximate tokens from UTF-8 bytes."""

    def estimate_text(self, text: str) -> int:
        size = len(text.encode("utf-8"))
        return (size + 2) // 3


@dataclass(frozen=True, slots=True)
class ContextBudgetPolicy:
    """Token pressure thresholds for one provider request."""

    context_window_tokens: int = 1_000_000
    reserved_output_tokens: int = 16_384
    pressure_trigger_ratio: float = 0.80
    pressure_target_ratio: float = 0.50
    raw_target_tokens: int = 48_000
    background_episode_tokens: int = 8_000
    rollup_episode_tokens: int = 48_000
    condenser_input_tokens: int = 64_000
    protected_recent_messages: int = 8
    protected_recent_tokens: int = 16_000
    max_tool_result_tokens: int = 16_000

    @property
    def hard_input_tokens(self) -> int:
        return self.context_window_tokens - self.reserved_output_tokens

    @property
    def pressure_trigger_tokens(self) -> int:
        return int(self.hard_input_tokens * self.pressure_trigger_ratio)

    @property
    def pressure_target_tokens(self) -> int:
        return int(self.hard_input_tokens * self.pressure_target_ratio)


def compact_json(value: Any) -> str:
    """Serialize provider-visible data consistently for estimation."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def estimate_model_context(
    context: ModelContext,
    tools: tuple[ToolDefinition, ...],
    estimator: TokenEstimator,
) -> int:
    payload = {
        "system": context.system_prompt,
        "messages": [message.as_dict() for message in context.messages],
        "tools": [tool.as_anthropic() for tool in tools],
    }
    return estimator.estimate_text(compact_json(payload))
