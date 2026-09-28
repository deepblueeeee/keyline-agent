"""Anthropic Messages conversation types persisted by Handshake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ContentBlock = dict[str, Any]


@dataclass(slots=True, frozen=True)
class ModelMessage:
    """One protocol message with the ID of its originating run."""

    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]
    run_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize only Anthropic-visible message fields."""
        return {"role": self.role, "content": list(self.content)}


@dataclass(slots=True, frozen=True)
class ModelContext:
    system_prompt: str
    goal: str
    workspace: str
    plan: tuple[str, ...]
    messages: tuple[ModelMessage, ...]


@dataclass(slots=True, frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ModelUsage:
    """Token counts reported for one Anthropic Messages request."""

    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cache_write_tokens,
                self.cache_read_tokens,
            )
        ):
            raise ValueError("Model usage counts must be non-negative integers")

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass(slots=True, frozen=True)
class ModelTurn:
    content: tuple[ContentBlock, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None
    usage: ModelUsage | None = None

    @property
    def text(self) -> str:
        return "\n".join(
            str(block["text"])
            for block in self.content
            if block.get("type") == "text" and block.get("text")
        ).strip()
