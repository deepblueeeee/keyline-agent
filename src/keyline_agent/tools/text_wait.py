"""Definition and dependency preparation for the model-facing text wait tool."""

from __future__ import annotations

from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ToolCall
from keyline_agent.runtime.state import Dependency, ToolResult

TEXT_WAIT_TOOL = ToolDefinition(
    name="wait_for_text",
    description=(
        "Pause the run to ask for a non-sensitive fact that cannot be inferred from "
        "the workspace. Never use this for passwords, API keys, tokens, or other secrets. "
        "It may be combined only with other wait requests."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "description": "The concise, non-sensitive question to show the user.",
            }
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
)


def prepare_text_dependency(call: ToolCall) -> Dependency | ToolResult:
    """Validate one request and describe the dependency blocking the run."""
    prompt = call.arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolResult(ok=False, stderr="prompt must be a non-empty string")
    normalized_prompt = prompt.strip()
    return Dependency(
        tool_call_id=call.id,
        prompt=normalized_prompt,
    )
