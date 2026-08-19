"""Provider-neutral model conversation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ContentBlock = dict[str, Any]


@dataclass(slots=True, frozen=True)
class ModelMessage:
    """One persisted session message and its originating run."""

    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]
    run_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize only provider-visible message fields."""
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
class ModelTurn:
    content: tuple[ContentBlock, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(
            str(block["text"])
            for block in self.content
            if block.get("type") == "text" and block.get("text")
        ).strip()
