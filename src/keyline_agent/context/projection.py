"""Project durable messages into the bounded model-visible representation."""

from __future__ import annotations

from copy import deepcopy

from keyline_agent.artifacts import ArtifactReferenceError, StoredTextArtifact
from keyline_agent.context.budget import TokenEstimator, compact_json
from keyline_agent.context.errors import ContextPressureError, ContextViewError
from keyline_agent.context.outputs import externalize_tool_result
from keyline_agent.models.protocol import ModelMessage


def project_message(
    raw: ModelMessage,
    *,
    run_id: str,
    sequence: int,
    stale_mask_through: tuple[int, int] | None,
    max_tool_result_tokens: int,
    estimator: TokenEstimator,
) -> ModelMessage:
    message = deepcopy(raw)
    if message.role != "user":
        return message
    for index, block in enumerate(message.content):
        if block.get("type") != "tool_result":
            continue
        if stale_mask_through is not None and (sequence, index) <= stale_mask_through:
            block["content"] = stale_marker(sequence, block.get("is_error"))
            continue
        call_id = block.get("tool_use_id")
        if not isinstance(call_id, str):
            continue
        content = block.get("content", "")
        text = content if isinstance(content, str) else compact_json(content)
        replacement = _ingress_content(
            run_id=run_id,
            sequence=sequence,
            block_index=index,
            call_id=call_id,
            is_error=block.get("is_error") is not False,
            content=text,
            max_tool_result_tokens=max_tool_result_tokens,
            estimator=estimator,
        )
        if replacement is not None:
            block["content"] = replacement
    return message


def stale_marker(sequence: int, is_error: object) -> str:
    status = "failure" if is_error is not False else "success"
    return (
        f"[stale tool result masked; status={status}; "
        f"source_sequence={sequence}; details omitted from active context]"
    )


def _ingress_content(
    *,
    run_id: str,
    sequence: int,
    block_index: int,
    call_id: str,
    is_error: bool,
    content: str,
    max_tool_result_tokens: int,
    estimator: TokenEstimator,
) -> str | None:
    original_tokens = _result_tokens(call_id, is_error, content, estimator)
    if original_tokens <= max_tool_result_tokens:
        return None
    try:
        external = externalize_tool_result(run_id, sequence, block_index, content)
    except (OSError, ArtifactReferenceError) as exc:
        raise ContextViewError(
            f"Could not externalize oversized tool result at source sequence {sequence}"
        ) from exc
    marker = _external_result_marker(
        external,
        sequence=sequence,
        is_error=is_error,
        estimated_tokens=original_tokens,
    )
    if _result_tokens(call_id, is_error, marker, estimator) > max_tool_result_tokens:
        raise ContextPressureError(
            "Tool-result ingress limit is too small to expose its recovery path"
        )
    return marker


def _external_result_marker(
    external: StoredTextArtifact,
    *,
    sequence: int,
    is_error: bool,
    estimated_tokens: int,
) -> str:
    return (
        "[tool result externalized; "
        f"status={'failure' if is_error else 'success'}; "
        f"source_sequence={sequence}; estimated_tokens={estimated_tokens}; "
        f"bytes={external.byte_count}; characters={external.character_count}; "
        f"lines={external.line_count}; path={external.reference}]\n"
        "Read the complete output with read_file. Ranges are 1-based and inclusive. "
        "Use one mode only: start_line/end_line OR start_character/end_character."
    )


def _result_tokens(
    call_id: str,
    is_error: bool,
    content: str,
    estimator: TokenEstimator,
) -> int:
    block = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
        "is_error": is_error,
    }
    return estimator.estimate_text(compact_json(block))
