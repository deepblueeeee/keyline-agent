"""Definition and state preparation for the model-facing text wait tool."""

from __future__ import annotations

from handshake.models.base import ToolDefinition
from handshake.runtime.state import ToolCall, ToolResult, WaitCondition, WorkItem

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


def prepare_text_wait(call: ToolCall) -> tuple[WorkItem, WaitCondition] | ToolResult:
    """Validate one request and bind its wait state to a minimal work item."""
    prompt = call.arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolResult(ok=False, stderr="prompt must be a non-empty string")
    normalized_prompt = prompt.strip()
    work_item = WorkItem(description=normalized_prompt, tool_call_id=call.id)
    wait = WaitCondition(
        work_id=work_item.work_id,
        tool_call_id=call.id,
        prompt=normalized_prompt,
    )
    return work_item, wait
