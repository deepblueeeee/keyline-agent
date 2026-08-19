"""Provider-neutral model and tool interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from handshake.models.protocol import ModelContext, ModelTurn


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ModelAdapter(Protocol):
    """The narrow provider boundary used by the agent loop."""

    def complete(
        self, context: ModelContext, tools: tuple[ToolDefinition, ...]
    ) -> ModelTurn:
        """Produce the next assistant turn."""
        ...
