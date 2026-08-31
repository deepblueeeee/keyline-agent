"""Composition root for workspace tools exposed to models."""

from handshake.models.base import ToolDefinition
from handshake.tools.files import file_tool_bindings
from handshake.tools.plan import PLAN_TOOL, handle_plan
from handshake.tools.registry import ToolRegistry
from handshake.tools.secret_command import SECRET_COMMAND_TOOL
from handshake.tools.secret_wait import SECRET_WAIT_TOOL, prepare_secret_dependency
from handshake.tools.shell import ApprovalCallback, process_tool_bindings
from handshake.tools.text_wait import TEXT_WAIT_TOOL, prepare_text_dependency


def create_default_registry(approval: ApprovalCallback) -> ToolRegistry:
    """Create the explicit MVP tool registry."""
    registry = ToolRegistry()
    bindings = (*file_tool_bindings(), *process_tool_bindings(approval))
    for binding in bindings:
        registry.register(binding)
    return registry


def runtime_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Return tools whose lifecycle is coordinated directly by the runtime."""
    return (PLAN_TOOL, TEXT_WAIT_TOOL, SECRET_WAIT_TOOL, SECRET_COMMAND_TOOL)


__all__ = [
    "ApprovalCallback",
    "PLAN_TOOL",
    "SECRET_WAIT_TOOL",
    "SECRET_COMMAND_TOOL",
    "TEXT_WAIT_TOOL",
    "ToolRegistry",
    "create_default_registry",
    "handle_plan",
    "prepare_secret_dependency",
    "prepare_text_dependency",
    "runtime_tool_definitions",
]
