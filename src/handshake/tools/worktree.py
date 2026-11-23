"""Model-facing tools for accepting, inspecting and cleaning up child changes."""

from handshake.models.base import ToolDefinition

_CHILD = {"type": "string", "minLength": 1, "description": "Run ID of the owning implementer."}

INSPECT_CHANGES = ToolDefinition(
    "inspect_changes",
    "Read a saved implementer change set, including test outcomes and a page of its Git diff.",
    {
        "type": "object",
        "properties": {
            "child_run_id": _CHILD,
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["child_run_id"],
        "additionalProperties": False,
    },
)

INTEGRATE_CHANGES = ToolDefinition(
    "integrate_changes",
    "Integrate one completed implementer's saved changes into the clean main workspace. "
    "Git reports conflicts without changing the main workspace. Choose the order explicitly.",
    {
        "type": "object", "properties": {"child_run_id": _CHILD},
        "required": ["child_run_id"], "additionalProperties": False,
    },
)

CLEANUP_WORKTREE = ToolDefinition(
    "cleanup_worktree",
    "Remove an integrated child worktree, or explicitly abandon it with abandon=true. "
    "The branch is retained as a backup.",
    {
        "type": "object", "properties": {
            "child_run_id": _CHILD, "abandon": {"type": "boolean", "default": False},
        },
        "required": ["child_run_id"], "additionalProperties": False,
    },
)

WORKTREE_TOOLS = (INSPECT_CHANGES, INTEGRATE_CHANGES, CLEANUP_WORKTREE)
WORKTREE_TOOL_NAMES = frozenset(tool.name for tool in WORKTREE_TOOLS)
