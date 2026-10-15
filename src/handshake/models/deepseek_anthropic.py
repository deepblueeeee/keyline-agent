"""DeepSeek adapter for its Anthropic-compatible Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import anthropic

from handshake.models.base import ToolDefinition
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import (
    ContentBlock,
    ModelContext,
    ModelTurn,
    ModelUsage,
    ToolCall,
)


class DeepSeekAnthropicModel:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/anthropic",
        model: str = "deepseek-v4-flash",
        max_tokens: int = 16_384,
        timeout_seconds: float = 120,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self.model = model
        self.max_tokens = max_tokens
        self._owns_client = client is None
        self._client: Any = client or anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def complete(
        self, context: ModelContext, tools: tuple[ToolDefinition, ...]
    ) -> ModelTurn:
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=context.system_prompt,
                messages=[message.as_dict() for message in context.messages],
                tools=[tool.as_anthropic() for tool in tools],
            )
            return _normalize_response(response)
        except ModelProtocolError:
            raise
        except anthropic.APIError as exc:
            status = getattr(exc, "status_code", None)
            suffix = f", HTTP {status}" if status is not None else ""
            raise ModelProviderError(
                f"DeepSeek request failed ({type(exc).__name__}{suffix})"
            ) from None
        except Exception as exc:
            raise ModelProviderError(
                f"DeepSeek request failed ({type(exc).__name__})"
            ) from None


def _as_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    raise ModelProtocolError(f"DeepSeek returned an invalid {description}")


def _normalize_response(response: Any) -> ModelTurn:
    payload = _as_mapping(response, "message")
    raw_content = payload.get("content")
    if not isinstance(raw_content, list) or not raw_content:
        raise ModelProtocolError("DeepSeek returned an empty message")

    content: list[ContentBlock] = []
    tool_calls: list[ToolCall] = []
    for raw_block in raw_content:
        block = dict(_as_mapping(raw_block, "content block"))
        block_type = block.get("type")
        if block_type not in {"text", "thinking", "tool_use"}:
            raise ModelProtocolError(f"Unsupported DeepSeek content block: {block_type!r}")
        if block_type == "text" and not isinstance(block.get("text"), str):
            raise ModelProtocolError("DeepSeek text block is missing text")
        if block_type == "thinking" and not isinstance(block.get("thinking"), str):
            raise ModelProtocolError("DeepSeek thinking block is missing thinking content")
        if block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(name, str)
                or not name
                or not isinstance(arguments, Mapping)
            ):
                raise ModelProtocolError("DeepSeek returned a malformed tool_use block")
            tool_calls.append(ToolCall(call_id, name, dict(arguments)))
        content.append(block)

    raw_stop_reason = payload.get("stop_reason")
    stop_reason = raw_stop_reason if isinstance(raw_stop_reason, str) else None
    usage = None
    if payload.get("usage") is not None:
        raw_usage = _as_mapping(payload["usage"], "usage")
        cache_write = raw_usage.get("cache_creation_input_tokens")
        cache_read = raw_usage.get("cache_read_input_tokens")
        try:
            usage = ModelUsage(
                input_tokens=raw_usage["input_tokens"],
                output_tokens=raw_usage["output_tokens"],
                cache_write_tokens=0 if cache_write is None else cache_write,
                cache_read_tokens=0 if cache_read is None else cache_read,
            )
        except (KeyError, ValueError):
            raise ModelProtocolError("DeepSeek returned invalid token usage") from None
    return ModelTurn(tuple(content), tuple(tool_calls), stop_reason, usage)
