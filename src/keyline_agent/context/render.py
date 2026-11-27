"""Pure validation and Anthropic Messages rendering for ContextView."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

from keyline_agent.context.errors import ContextViewError
from keyline_agent.context.state import (
    ContextView,
    ExactContext,
    HistoryEpisode,
    HistorySegment,
    RawHistorySegment,
    SequencedMessage,
    SourceRange,
)
from keyline_agent.models.base import ToolDefinition
from keyline_agent.models.protocol import ModelContext, ModelMessage


def render_context_view(
    view: ContextView,
    *,
    instructions: str = "",
    tools: tuple[ToolDefinition, ...] = (),
) -> ModelContext:
    """Validate and render a ContextView without changing its source data."""
    _validate_view(view)
    exact = view.exact
    prompt = _system_prompt(exact, instructions, tools)

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


def _system_prompt(
    exact: ExactContext,
    instructions: str,
    tools: tuple[ToolDefinition, ...],
) -> str:
    tool_names = {tool.name for tool in tools}
    protocol = [
        "- Use only the tools provided with this request; never invent a tool or its result.",
        "- Treat tool results as evidence. Do not claim an action succeeded without its result.",
        "- Never put secret values in ordinary messages, model requests, workspace files, or logs.",
        "- When finished, reply with text and no tool calls.",
    ]
    if "update_plan" in tool_names:
        protocol.append("- update_plan records a provisional plan; revise it as work changes.")
    if "read_file" in tool_names:
        protocol.append(
            "- A tool result may contain an opaque @keyline/artifacts reference; "
            "read it only with read_file."
        )
    if tool_names & {"run_shell", "run_tests", "run_command_with_secret"}:
        protocol.append(
            "- Commands follow the user's local approval policy. If denied, do not claim they ran."
        )
    if "wait_for_text" in tool_names:
        protocol.extend(
            (
                "- Use wait_for_text for non-sensitive user input; never use it "
                "for credentials.",
                "- Its tool result registers a dependency. The answer arrives later as "
                "a user continuation message.",
            )
        )
    if "wait_for_secret" in tool_names:
        protocol.append(
            "- Use wait_for_secret for a credential. Its result contains an "
            "opaque handle, never the secret value."
        )
    wait_tools = tuple(
        name for name in ("wait_for_text", "wait_for_secret") if name in tool_names
    )
    if wait_tools:
        protocol.append(
            "- A turn may request multiple waits, but it must contain only "
            + " or ".join(wait_tools)
            + " tool calls. The run continues after every requested input is resolved."
        )
    if "run_command_with_secret" in tool_names:
        protocol.append(
            "- Pass an available one-time secret handle only to run_command_with_secret. "
            "Request a new handle before rerunning a command that started."
        )
    if "delegate" in tool_names:
        protocol.append(
            "- A turn may contain multiple delegate calls, but no other tools. Each child "
            "uses its own message history; results arrive together after every child finishes."
        )
    state = {
        "goal": exact.goal,
        "workspace": exact.workspace,
        "plan": exact.plan,
        "constraints": exact.constraints,
        "dependencies": [asdict(item) for item in exact.dependencies],
    }
    return (
        "Harness protocol\n"
        + "\n".join(protocol)
        + "\n\nAgent instructions\n"
        + (instructions.strip() or "(none)")
        + "\n\nRuntime state\n"
        + json.dumps(state, ensure_ascii=False)
    )


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
