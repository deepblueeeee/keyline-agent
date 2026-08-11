"""A small registry for explicit tool definition/handler bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handshake.models.base import ToolDefinition
from handshake.runtime.errors import ToolExecutionError, UserCancelledError
from handshake.runtime.state import ToolResult

ToolHandler = Callable[[Path, Mapping[str, Any]], ToolResult]


@dataclass(slots=True, frozen=True)
class ToolBinding:
    """Keep a model-facing definition attached to its runtime handler."""

    definition: ToolDefinition
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, ToolBinding] = {}

    def register(self, binding: ToolBinding) -> None:
        name = binding.definition.name
        if name in self._bindings:
            raise ValueError(f"Tool already registered: {name}")
        self._bindings[name] = binding

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(binding.definition for binding in self._bindings.values())

    def execute(
        self, name: str, workspace: Path, arguments: Mapping[str, Any]
    ) -> ToolResult:
        binding = self._bindings.get(name)
        if binding is None:
            return ToolResult(ok=False, stderr=f"Unknown tool: {name}")
        try:
            return binding.handler(workspace, arguments)
        except UserCancelledError:
            raise
        except (ToolExecutionError, OSError, UnicodeError) as exc:
            return ToolResult(ok=False, stderr=str(exc))
