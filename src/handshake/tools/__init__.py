"""Composition root for workspace tools exposed to models."""

from handshake.tools.files import file_tool_bindings
from handshake.tools.registry import ToolRegistry
from handshake.tools.shell import ApprovalCallback, process_tool_bindings


def create_default_registry(approval: ApprovalCallback) -> ToolRegistry:
    """Create the explicit MVP tool registry."""
    registry = ToolRegistry()
    bindings = (*file_tool_bindings(), *process_tool_bindings(approval))
    for binding in bindings:
        registry.register(binding)
    return registry


__all__ = ["ApprovalCallback", "ToolRegistry", "create_default_registry"]
