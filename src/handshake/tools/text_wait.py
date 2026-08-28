"""Definition and dependency preparation for the model-facing text wait tool."""

from __future__ import annotations

from handshake.models.base import ToolDefinition
from handshake.models.protocol import ToolCall
from handshake.runtime.state import Dependency, ToolResult, WorkItem

TEXT_WAIT_TOOL = ToolDefinition(
    name="wait_for_text",
    description=(
        "Pause the current work to ask for a non-sensitive fact that cannot be inferred from "
        "the workspace. Never use this for passwords, API keys, tokens, or other secrets. "
        "This must be the only tool call in the assistant turn."
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


def prepare_text_dependency(
    call: ToolCall, work_item: WorkItem
) -> Dependency | ToolResult:
    """Validate one request and describe the dependency blocking current work."""
    prompt = call.arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolResult(ok=False, stderr="prompt must be a non-empty string")
    normalized_prompt = prompt.strip()
    return Dependency(
        work_id=work_item.work_id,
        tool_call_id=call.id,
        prompt=normalized_prompt,
    )
