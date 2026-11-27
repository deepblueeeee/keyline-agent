"""A small registry for explicit tool definition/handler bindings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyline_agent.models.base import ToolDefinition
from keyline_agent.runtime.errors import ToolExecutionError, UserCancelledError
from keyline_agent.runtime.state import ReplayPolicy, ToolResult

ToolHandler = Callable[[Path, Mapping[str, Any]], ToolResult]
PreparedTool = Callable[[], ToolResult] | ToolResult
ToolPreparation = PreparedTool | Awaitable[PreparedTool]
ToolPreparer = Callable[[Path, Mapping[str, Any]], ToolPreparation]


@dataclass(slots=True, frozen=True)
class ToolBinding:
    """Keep a model-facing definition attached to its runtime handler."""

    definition: ToolDefinition
    handler: ToolHandler
    replay_policy: ReplayPolicy = ReplayPolicy.MANUAL
    prepare: ToolPreparer | None = None


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

    def replay_policy(self, name: str) -> ReplayPolicy:
        binding = self._bindings.get(name)
        return ReplayPolicy.AUTO if binding is None else binding.replay_policy

    def prepare(
        self, name: str, workspace: Path, arguments: Mapping[str, Any]
    ) -> ToolPreparation:
        """Resolve validation and approval before acquiring the workspace execution slot."""
        binding = self._bindings.get(name)
        if binding is None or binding.prepare is None:
            return lambda: self.execute(name, workspace, arguments)
        try:
            return binding.prepare(workspace, arguments)
        except UserCancelledError:
            raise
        except (ToolExecutionError, OSError, UnicodeError) as exc:
            return ToolResult(ok=False, stderr=str(exc))

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
