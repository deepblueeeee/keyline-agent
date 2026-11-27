"""Bounded evidence from the actual model request for a tool decision."""

import json
from typing import Any

from keyline_agent.models.protocol import ModelContext

VISIBLE_BLOCKS = 12
EXCERPT_CHARS = 2000


def visible_context(context: ModelContext) -> dict[str, Any]:
    blocks = [
        (message.role, block)
        for message in context.messages
        for block in message.content
        if block["type"] in {"text", "tool_use", "tool_result"}
    ]
    visible = []
    for role, block in blocks[-VISIBLE_BLOCKS:]:
        excerpt = block.get("text", block.get("content", block.get("input")))
        if not isinstance(excerpt, str):
            excerpt = json.dumps(excerpt, ensure_ascii=False)
        visible.append({
            "role": role,
            **{key: block[key] for key in ("type", "id", "name", "tool_use_id") if key in block},
            "excerpt": excerpt[:EXCERPT_CHARS],
            "truncated": len(excerpt) > EXCERPT_CHARS,
        })
    return {
        "system_prompt": context.system_prompt,
        "visible": visible,
        "omitted_blocks": len(blocks) - len(visible),
    }
