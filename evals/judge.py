"""Make one tool-free JSON Judge request and report provider usage."""

import json
import os
import re
from dataclasses import asdict
from typing import Any

from handshake.models.base import ModelAdapter
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import ModelContext, ModelMessage


def check_judge_input(payload: str, api_key: str) -> None:
    secrets = [
        api_key,
        *(value for name, value in os.environ.items()
          if len(value) >= 8 and re.search(r"API_?KEY|TOKEN|SECRET|PASSWORD", name, re.I)),
    ]
    if any(secret and secret in payload for secret in secrets) or re.search(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[\w.-]{8,}|\bsk-[\w-]{12,}|"
        r"\b(?:api[_-]?key|password|secret|token)['\"]?\s*[:=]\s*['\"]?[\w./+-]{8,}",
        payload, re.I,
    ):
        raise ValueError("Judge input contains a possible credential")


async def judge_json(
    model: ModelAdapter, rubric: str, payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, int] | None, str | None]:
    message = ModelMessage("user", ({"type": "text", "text": json.dumps(payload)},))
    try:
        turn = await model.complete(ModelContext(rubric, "", "", (), (message,)), ())
    except (ModelProtocolError, ModelProviderError) as exc:
        return None, None, type(exc).__name__
    usage = asdict(turn.usage) if turn.usage is not None else None
    try:
        data = json.loads(turn.text.removeprefix("```json\n").removesuffix("\n```"))
    except json.JSONDecodeError:
        return None, usage, "invalid_json"
    return (data, usage, None) if isinstance(data, dict) else (None, usage, "expected_object")
