"""Definition and handler for the model-facing plan tool."""

from __future__ import annotations

from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ToolCall
from keyline_agent.runtime.state import Run, ToolResult

PLAN_TOOL = ToolDefinition(
    name="update_plan",
    description=(
        "Replace the current run's optional short plan. Use this for non-trivial, multi-step work "
        "after initial investigation, and update it only when the approach materially changes. "
        "Do not call this tool for simple work that does not benefit from progress tracking."
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
