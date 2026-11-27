"""Recoverable files for tool results that are too large for model ingress."""

from __future__ import annotations

import json

from keyline_agent.artifacts import StoredTextArtifact, store_text_artifact


def readable_tool_result(content: str) -> str:
    """Turn Keyline's JSON result envelope into line-readable lossless text."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if (
        not isinstance(value, dict)
        or set(value) != {"ok", "stdout", "stderr", "exit_code"}
        or not isinstance(value["ok"], bool)
        or not isinstance(value["stdout"], str)
        or not isinstance(value["stderr"], str)
        or (value["exit_code"] is not None and not isinstance(value["exit_code"], int))
    ):
        return content
    stdout = value["stdout"]
    stderr = value["stderr"]
    stdout_suffix = "" if not stdout or stdout.endswith("\n") else "\n"
    return (
        "[tool result]\n"
        f"ok: {str(value['ok']).lower()}\n"
        f"exit_code: {value['exit_code']}\n"
        "[stdout]\n"
        f"{stdout}{stdout_suffix}"
        "[stderr]\n"
        f"{stderr}"
    )


def externalize_tool_result(
    run_id: str,
    sequence: int,
    block_index: int,
    content: str,
) -> StoredTextArtifact:
    """Store one tool result as a generic text artifact."""
    return store_text_artifact(
        f"tool-result\0{run_id}\0{sequence}\0{block_index}",
        readable_tool_result(content),
    )
