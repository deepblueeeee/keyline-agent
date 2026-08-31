"""Model-facing Secret dependency request and its safe metadata."""

from __future__ import annotations

from handshake.models.base import ToolDefinition
from handshake.models.protocol import ToolCall
from handshake.runtime.state import (
    Dependency,
    DependencyKind,
    SecretHandle,
    ToolResult,
    WorkItem,
)

SECRET_COMMAND_CONSUMER = "run_command_with_secret"
SECRET_ENVIRONMENT_VARIABLES = frozenset(
    {"TEST_API_KEY", "TEST_ACCESS_TOKEN", "TEST_PASSWORD"}
)

SECRET_WAIT_TOOL = ToolDefinition(
    name="wait_for_secret",
    description=(
        "Request a credential through Handshake's out-of-band, no-echo Secret channel. "
        "Use only when run_command_with_secret needs one of the supported environment variables. "
        "This must be the only tool call in the assistant turn."
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


def prepare_secret_dependency(
    call: ToolCall, work_item: WorkItem
) -> tuple[Dependency, SecretHandle] | ToolResult:
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
        work_id=work_item.work_id,
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
