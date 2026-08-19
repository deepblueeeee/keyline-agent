"""Definition and handler for the model-facing plan tool."""

from __future__ import annotations

from handshake.models.base import ToolDefinition
from handshake.models.protocol import ToolCall
from handshake.runtime.state import Run, ToolResult

PLAN_TOOL = ToolDefinition(
    name="update_plan",
    description=(
        "Replace the current run's short plan. Use this after initial investigation and when the "
        "approach materially changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
)


def handle_plan(run: Run, call: ToolCall) -> ToolResult:
    """Validate and replace the run's current short plan."""
    raw_steps = call.arguments.get("steps")
    if (
        not isinstance(raw_steps, list)
        or not raw_steps
        or any(not isinstance(step, str) or not step.strip() for step in raw_steps)
    ):
        return ToolResult(
            ok=False,
            stderr="steps must be a non-empty array of non-empty strings",
        )
    run.plan = [step.strip() for step in raw_steps]
    return ToolResult(ok=True, stdout="Plan updated")
