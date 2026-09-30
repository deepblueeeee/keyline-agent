"""Model-facing request to run a configured child agent."""

from __future__ import annotations

from handshake.models.base import ToolDefinition

DELEGATE_TOOL = ToolDefinition(
    name="delegate",
    description=(
        "Run a configured child agent on a specific goal in its own model context. "
        "The child shares the workspace, but not your message history. "
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
