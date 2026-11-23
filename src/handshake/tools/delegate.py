"""Model-facing request to run a configured child agent."""

from __future__ import annotations

from handshake.models.base import ToolDefinition

DELEGATE_TOOL = ToolDefinition(
    name="delegate",
    description=(
        "Run a configured child agent on a specific goal in its own model context. "
        "Implementers get separate Git worktrees (up to three per batch). "
        "Explorers and reviewers read the main workspace or a completed implementer's worktree. "
        "A tool turn may delegate to several child agents and no other tools. "
        "Their structured results return together after every child finishes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "agent_spec_id": {
                "type": "string",
                "minLength": 1,
                "description": "The configured child agent to run.",
            },
            "target_run_id": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Optional owning implementer Run ID for an explorer or reviewer. "
                    "Omit to use the main workspace."
                ),
            },
            "goal": {
                "type": "string",
                "minLength": 1,
                "description": "A self-contained goal for the child agent.",
            },
        },
        "required": ["agent_spec_id", "goal"],
        "additionalProperties": False,
    },
)
