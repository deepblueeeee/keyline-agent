"""Pure validation and provider-neutral rendering for ContextView."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

from handshake.context.errors import ContextViewError
from handshake.context.state import (
    ContextView,
    ExactContext,
    HistoryEpisode,
    HistorySegment,
    RawHistorySegment,
    SequencedMessage,
    SourceRange,
)
from handshake.models.protocol import ModelContext, ModelMessage


def render_context_view(view: ContextView) -> ModelContext:
    """Validate and render a ContextView without changing its source data."""
    _validate_view(view)
    exact = view.exact
    prompt = _system_prompt(exact)
    state = {
        "constraints": exact.constraints,
        "work_items": [asdict(item) for item in exact.work_items],
        "dependencies": [asdict(item) for item in exact.dependencies],
    }
    prompt += "\nExact current state:\n" + json.dumps(state, ensure_ascii=False)

    messages: list[ModelMessage] = []
    for segment in view.history:
        if isinstance(segment, RawHistorySegment):
            messages.extend(item.message for item in segment.messages)
        else:
            messages.append(_render_episode(segment))
    messages.extend(item.message for item in view.protected_recent.messages)

    return ModelContext(
        system_prompt=prompt,
        goal=exact.goal,
        workspace=exact.workspace,
        plan=exact.plan,
        messages=deepcopy(tuple(messages)),
    )


def _system_prompt(exact: ExactContext) -> str:
    plan = "\n".join(f"- {step}" for step in exact.plan) or "(none)"
    return f"""You are Handshake, a local coding agent operating on one workspace.

Current goal:
{exact.goal}

Workspace:
{exact.workspace}

Optional current plan:
{plan}

Rules:
- Investigate enough repository context before choosing an approach or modifying files.
- Use update_plan for non-trivial, multi-step work when it improves progress tracking.
- Simple requests may be completed without creating a plan.
- Treat the plan as provisional and revise it when tool results materially change the approach.
- Use workspace-relative paths for repository files. A large tool result may provide an opaque
  @handshake/artifacts reference; only read that reference with read_file.
- Keep context use focused: discover with list_files/search_files, read only relevant line ranges,
  use read_file's character-range mode for very long lines, and prefer edit_file for targeted
  changes to existing files.
- Make targeted changes and run appropriate validation afterwards.
- Shell tools require user approval. If denied, adapt instead of pretending they ran.
- Ask for a non-sensitive user fact with wait_for_text only when it cannot be inferred from the
  repository. Call wait_for_text by itself in a turn.
- Never request passwords, API keys, tokens, or other secrets with wait_for_text.
- A successful wait_for_text result means an input dependency was registered. The user's later
  answer arrives as a new user continuation message, not as another result for the same tool call.
- Request a test credential with wait_for_secret, never through normal messages or file tools.
  Call wait_for_secret by itself. Its successful result provides an opaque handle, not the Secret.
- Pass that opaque handle only to run_command_with_secret when the requested credential is
  available. Handles are one-time: request a new one before rerunning a command that started.
- Use tool results as evidence and correct failures when possible.
- Do not claim a test passed unless its tool result says it passed.
- When the current request is complete, respond without tool calls with a concise summary of
  changes, validation performed, and any unresolved issue.
"""


def _segment_range(segment: HistorySegment) -> SourceRange:
    if isinstance(segment, HistoryEpisode):
        return segment.source_range
    if not segment.messages:
        raise ContextViewError("A raw history segment cannot be empty")
    return SourceRange(segment.messages[0].sequence, segment.messages[-1].sequence)


def _validate_sequenced_messages(
    messages: tuple[SequencedMessage, ...], description: str
) -> None:
    if not messages:
        return
    expected = messages[0].sequence
    for item in messages:
        if item.sequence != expected:
            raise ContextViewError(f"{description} messages must have continuous sequences")
        expected += 1


def _validate_view(view: ContextView) -> None:
    if view.source_message_count < 0:
        raise ContextViewError("source_message_count cannot be negative")

    expected_sequence = 0
    for segment in view.history:
        if isinstance(segment, RawHistorySegment):
            _validate_sequenced_messages(segment.messages, "Raw history")
        source_range = _segment_range(segment)
        if (
            source_range.start_sequence < 0
            or source_range.end_sequence < source_range.start_sequence
        ):
            raise ContextViewError("History segments must use valid source ranges")
        if source_range.start_sequence != expected_sequence:
            raise ContextViewError(
                "Context history must cover source sequences in order without gaps or overlaps"
            )
        expected_sequence = source_range.end_sequence + 1

    recent = view.protected_recent.messages
    _validate_sequenced_messages(recent, "Protected recent")
    if recent:
        if recent[0].sequence != expected_sequence:
            raise ContextViewError(
                "Protected recent history must follow older history without gaps or overlaps"
            )
        expected_sequence = recent[-1].sequence + 1

    if expected_sequence != view.source_message_count:
        raise ContextViewError("ContextView must cover every source message exactly once")


def _render_episode(episode: HistoryEpisode) -> ModelMessage:
    source_range = episode.source_range
    text = (
        "[Compacted history episode; source messages "
        f"{source_range.start_sequence}-{source_range.end_sequence}]\n"
        f"Title: {episode.title}\n"
        f"Summary: {episode.summary}"
    )
    return ModelMessage(role="user", content=({"type": "text", "text": text},))
