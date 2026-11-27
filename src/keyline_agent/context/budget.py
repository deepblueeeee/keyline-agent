"""Deterministic local estimates for model-visible context pressure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ModelContext, ModelMessage, ModelTurn


@dataclass(frozen=True, slots=True)
class ContextTokenMeasurement:
    """Provider-measured message prefix plus estimated request header overhead.

    Fingerprints bind counts to the actual visible history, including projections
    and summaries. No prompt text is retained in persistence metadata.
    """

    tokens: int
    message_count: int
    messages_fingerprint: str
    header_estimate: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.tokens,
                    self.message_count,
                    self.header_estimate,
                )
            )
            or not isinstance(self.messages_fingerprint, str)
            or (
                len(self.messages_fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in self.messages_fingerprint)
            )
        ):
            raise ValueError("Invalid context token measurement")


def _messages_fingerprint(messages: tuple[ModelMessage, ...]) -> str:
    return sha256(compact_json([message.as_dict() for message in messages]).encode()).hexdigest()


def _header_estimate(
    context: ModelContext,
    tools: tuple[ToolDefinition, ...],
    estimator: TokenEstimator,
) -> int:
    return estimator.estimate_text(
        compact_json(
            {
                "system": context.system_prompt,
                "tools": [tool.as_anthropic() for tool in tools],
            }
        )
    )


def measure_model_turn(
    context: ModelContext,
    tools: tuple[ToolDefinition, ...],
    turn: ModelTurn,
    estimator: TokenEstimator,
) -> ContextTokenMeasurement | None:
    """Replace the previous baseline: each input count already includes history."""
    if turn.usage is None:
        return None
    messages = (*context.messages, ModelMessage("assistant", turn.content))
    return ContextTokenMeasurement(
        tokens=turn.usage.total_input_tokens + turn.usage.output_tokens,
        message_count=len(messages),
        messages_fingerprint=_messages_fingerprint(messages),
        header_estimate=_header_estimate(context, tools, estimator),
    )


def measure_model_context(
    context: ModelContext,
    tools: tuple[ToolDefinition, ...],
    estimator: TokenEstimator,
    measurement: ContextTokenMeasurement | None,
) -> int:
    """Use measured history, estimating only new messages and header changes.

    Masking, projection changes or semantic compression invalidate the prefix;
    the resulting context is estimated until the next response recalibrates it.
    """
    if (
        measurement is None
        or len(context.messages) < measurement.message_count
        or (
            _messages_fingerprint(context.messages[: measurement.message_count])
            != measurement.messages_fingerprint
        )
    ):
        return estimate_model_context(context, tools, estimator)
    added = context.messages[measurement.message_count :]
    added_tokens = sum(
        estimator.estimate_text(compact_json(message.as_dict())) for message in added
    )
    return max(
        0,
        measurement.tokens
        + added_tokens
        + _header_estimate(context, tools, estimator)
        - measurement.header_estimate,
    )


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

    context_window_tokens: int = 224_000
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
