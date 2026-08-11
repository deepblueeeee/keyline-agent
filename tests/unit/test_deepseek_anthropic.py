from pathlib import Path
from typing import Any

import pytest

from handshake.models.base import ToolDefinition
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.runtime.errors import ModelProviderError
from handshake.runtime.loop import run_agent
from handshake.runtime.state import ModelContext, ModelMessage, Task, TaskStatus
from handshake.tools import create_default_registry


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)


def test_normalizes_thinking_text_and_tool_use() -> None:
    client = FakeClient(
        [
            {
                "content": [
                    {"type": "thinking", "thinking": "inspect first", "signature": "sig"},
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "read_file",
                        "input": {"path": "x.py"},
                    },
                ],
                "stop_reason": "tool_use",
            }
        ]
    )
    model = DeepSeekAnthropicModel(api_key="canary", client=client)
    context = ModelContext(
        system_prompt="system",
        goal="goal",
        workspace="/tmp/work",
        plan=(),
        messages=(ModelMessage("user", ({"type": "text", "text": "goal"},)),),
    )
    definition = ToolDefinition("read_file", "read", {"type": "object"})

    turn = model.complete(context, (definition,))

    assert turn.text == "Checking."
    assert turn.tool_calls[0].arguments == {"path": "x.py"}
    assert turn.content[0]["signature"] == "sig"
    request = client.messages.requests[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["system"] == "system"
    assert request["tools"][0]["name"] == "read_file"


def test_loop_round_trips_thinking_and_tool_result(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("value = 1\n", encoding="utf-8")
    client = FakeClient(
        [
            {
                "content": [
                    {"type": "thinking", "thinking": "read it", "signature": "signed"},
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "read_file",
                        "input": {"path": "x.py"},
                    },
                ],
                "stop_reason": "tool_use",
            },
            {
                "content": [{"type": "text", "text": "Inspected x.py."}],
                "stop_reason": "end_turn",
            },
        ]
    )
    task = Task(goal="inspect", workspace=tmp_path)

    run_agent(
        task,
        DeepSeekAnthropicModel(api_key="canary", client=client),
        create_default_registry(lambda _cwd, _argv: True),
    )

    assert task.status is TaskStatus.COMPLETED
    second_messages = client.messages.requests[1]["messages"]
    assert second_messages[-2]["content"][0]["type"] == "thinking"
    assert second_messages[-2]["content"][0]["signature"] == "signed"
    assert second_messages[-1]["content"][0]["tool_use_id"] == "call-1"


def test_provider_error_does_not_leak_api_key() -> None:
    secret = "canary-super-secret"
    client = FakeClient([RuntimeError(secret)])
    model = DeepSeekAnthropicModel(api_key=secret, client=client)
    context = ModelContext("system", "goal", "/tmp/work", (), ())

    with pytest.raises(ModelProviderError) as raised:
        model.complete(context, ())

    assert secret not in str(raised.value)
    assert secret not in repr(model)
    assert secret not in vars(model)
