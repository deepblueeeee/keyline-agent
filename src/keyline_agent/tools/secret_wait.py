"""Model-facing Secret dependency request and its safe metadata."""

from __future__ import annotations

from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ToolCall
from keyline_agent.runtime.state import (
    Dependency,
    DependencyKind,
    SecretHandle,
    ToolResult,
)

SECRET_COMMAND_CONSUMER = "run_command_with_secret"
SECRET_ENVIRONMENT_VARIABLES = frozenset(
    {"TEST_API_KEY", "TEST_ACCESS_TOKEN", "TEST_PASSWORD"}
)

SECRET_WAIT_TOOL = ToolDefinition(
    name="wait_for_secret",
    description=(
        "Request a credential through Keyline's out-of-band, no-echo Secret channel. "
        "Use only when run_command_with_secret needs one of the supported environment variables. "
        "It may be combined only with other wait requests."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "minLength": 1,
                "description": "The concise Secret request shown only by the CLI.",
            },
            "purpose": {
                "type": "string",
                "minLength": 1,
                "description": "A non-sensitive explanation of why the test needs it.",
            },
            "environment_variable": {
                "type": "string",
                "enum": sorted(SECRET_ENVIRONMENT_VARIABLES),
            },
        },
        "required": ["prompt", "purpose", "environment_variable"],
        "additionalProperties": False,
    },
)


def prepare_secret_dependency(call: ToolCall) -> tuple[Dependency, SecretHandle] | ToolResult:
    """Validate a Secret request without obtaining or storing its raw value."""
    prompt = call.arguments.get("prompt")
    purpose = call.arguments.get("purpose")
    environment_variable = call.arguments.get("environment_variable")
    if not isinstance(prompt, str) or not prompt.strip():
        return ToolResult(ok=False, stderr="prompt must be a non-empty string")
    if not isinstance(purpose, str) or not purpose.strip():
        return ToolResult(ok=False, stderr="purpose must be a non-empty string")
    if environment_variable not in SECRET_ENVIRONMENT_VARIABLES:
        return ToolResult(
            ok=False,
            stderr="environment_variable must be one of the supported test variables",
        )
    dependency = Dependency(
        tool_call_id=call.id,
        prompt=prompt.strip(),
        kind=DependencyKind.SECRET,
    )
    handle = SecretHandle(
        dependency_id=dependency.dependency_id,
        purpose=purpose.strip(),
        consumer=SECRET_COMMAND_CONSUMER,
        environment_variable=environment_variable,
    )
    return dependency, handle
