from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message

from handshake.models.base import ToolDefinition
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import ModelContext, ModelMessage, ModelUsage
from handshake.runtime.loop import start_run
from handshake.runtime.state import Run, RunStatus, Session
from handshake.tools import create_default_registry
from tests.history_support import TEST_AGENT_SPEC, fresh_store, run_async


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessages(responses)


@run_async
async def test_normalizes_thinking_text_and_tool_use() -> None:
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
        messages=(
            ModelMessage(
                "user",
                ({"type": "text", "text": "goal"},),
                run_id="internal-run-id",
            ),
        ),
    )
    definition = ToolDefinition("read_file", "read", {"type": "object"})

    turn = await model.complete(context, (definition,))

    assert turn.text == "Checking."
    assert turn.tool_calls[0].arguments == {"path": "x.py"}
    assert turn.content[0]["signature"] == "sig"
    request = client.messages.requests[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["max_tokens"] == 16_384
    assert request["system"] == "system"
    assert request["tools"][0]["name"] == "read_file"
    assert "run_id" not in request["messages"][0]


@run_async
async def test_loop_round_trips_thinking_and_tool_result(tmp_path: Path) -> None:
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
    session = Session(workspace=tmp_path)
    run = Run(goal="inspect", session_id=session.session_id, agent_spec_id="test")

    await start_run(
        session,
        run,
        DeepSeekAnthropicModel(api_key="canary", client=client),
        create_default_registry(lambda _cwd, _argv: True),
        store=fresh_store(tmp_path, session),
        agent_spec=TEST_AGENT_SPEC,
    )

    assert run.status is RunStatus.COMPLETED
    second_messages = client.messages.requests[1]["messages"]
    assert second_messages[-2]["content"][0]["type"] == "thinking"
    assert second_messages[-2]["content"][0]["signature"] == "signed"
    assert second_messages[-1]["content"][0]["tool_use_id"] == "call-1"


@run_async
async def test_adapter_sends_consecutive_user_messages_without_domain_normalization() -> None:
    client = FakeClient(
        [
            {
                "content": [{"type": "text", "text": "continued"}],
                "stop_reason": "end_turn",
            }
        ]
    )
    model = DeepSeekAnthropicModel(api_key="canary", client=client)
    context = ModelContext(
        system_prompt="system",
        goal="goal",
        workspace="/tmp/work",
        plan=(),
        messages=(
            ModelMessage(
                "user",
                (
                    {
                        "type": "tool_result",
                        "tool_use_id": "ask-1",
                        "content": "dependency registered",
                    },
                ),
                run_id="run",
            ),
            ModelMessage(
                "user",
                ({"type": "text", "text": "User response: navy"},),
                run_id="run",
            ),
        ),
    )

    await model.complete(context, ())

    messages = client.messages.requests[0]["messages"]
    assert [message["role"] for message in messages] == ["user", "user"]
    assert messages[0]["content"][0]["tool_use_id"] == "ask-1"
    assert messages[1]["content"][0]["text"] == "User response: navy"
    assert all("run_id" not in message for message in messages)


@run_async
async def test_provider_error_does_not_leak_api_key() -> None:
    secret = "canary-super-secret"
    client = FakeClient([RuntimeError(secret)])
    model = DeepSeekAnthropicModel(api_key=secret, client=client)
    context = ModelContext("system", "goal", "/tmp/work", (), ())

    with pytest.raises(ModelProviderError) as raised:
        await model.complete(context, ())

    assert secret not in str(raised.value)
    assert secret not in repr(model)
    assert secret not in vars(model)


@pytest.mark.parametrize("sdk_response", [False, True])
@run_async
async def test_usage_is_normalized_from_mapping_and_sdk_message(sdk_response: bool) -> None:
    payload: dict[str, Any] = {
        "id": "msg-1", "type": "message", "role": "assistant", "model": "test",
        "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_creation_input_tokens": 30, "cache_read_input_tokens": 40},
    }
    response = Message.model_validate(payload) if sdk_response else payload
    model = DeepSeekAnthropicModel(api_key="canary", client=FakeClient([response]))
    turn = await model.complete(ModelContext("system", "goal", "/tmp", (), ()), ())
    assert turn.usage == ModelUsage(
        input_tokens=100, output_tokens=20, cache_write_tokens=30, cache_read_tokens=40,
    )
    assert turn.usage.total_input_tokens == 170


@pytest.mark.parametrize("usage", [
    {}, {"input_tokens": 1}, {"input_tokens": -1, "output_tokens": 2},
    {"input_tokens": True, "output_tokens": 2},
    {"input_tokens": 1.5, "output_tokens": 2},
    {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": False},
    {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": "100"},
])
@run_async
async def test_invalid_usage_is_rejected(usage: dict[str, Any]) -> None:
    model = DeepSeekAnthropicModel(api_key="canary", client=FakeClient([{
        "content": [{"type": "text", "text": "done"}], "usage": usage,
    }]))
    with pytest.raises(ModelProtocolError, match="token usage"):
        await model.complete(ModelContext("system", "goal", "/tmp", (), ()), ())
