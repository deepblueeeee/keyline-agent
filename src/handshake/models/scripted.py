"""Deterministic model used for runtime and scenario tests."""

from __future__ import annotations

from collections.abc import Iterable

from handshake.models.base import ToolDefinition
from handshake.models.errors import ModelProviderError
from handshake.models.protocol import ModelContext, ModelTurn


class ScriptedModel:
    def __init__(self, turns: Iterable[ModelTurn]) -> None:
        self._turns = list(turns)
        self.contexts: list[ModelContext] = []
        self.tool_sets: list[tuple[ToolDefinition, ...]] = []

    async def complete(
        self, context: ModelContext, tools: tuple[ToolDefinition, ...]
    ) -> ModelTurn:
        self.contexts.append(context)
        self.tool_sets.append(tools)
        if not self._turns:
            raise ModelProviderError("Scripted model ran out of turns before completion")
        return self._turns.pop(0)
