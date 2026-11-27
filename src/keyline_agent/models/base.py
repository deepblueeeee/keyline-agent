"""Model and tool interfaces for the Anthropic Messages protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from keyline_agent.models.protocol import ModelContext, ModelTurn


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
    """An Anthropic Messages model used by the agent loop."""

    async def complete(
        self, context: ModelContext, tools: tuple[ToolDefinition, ...]
    ) -> ModelTurn:
        """Produce the next assistant turn."""
        ...


async def close_model(model: ModelAdapter) -> None:
    """Release an adapter's optional process-local network resources."""
    close = getattr(model, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception as exc:
            # Cleanup cannot replace a committed outcome or a primary execution error.
            logging.getLogger(__name__).warning(
                "Model resource cleanup failed (%s)", type(exc).__name__
            )
