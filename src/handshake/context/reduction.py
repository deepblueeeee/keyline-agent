"""Deterministic masking and two-stage semantic context reduction."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

from handshake.context.budget import (
    ContextBudgetPolicy,
    TokenEstimator,
    compact_json,
    estimate_model_context,
)
from handshake.context.condenser import (
    HistoryCondenser,
    condenser_context,
    episode_data,
    raw_data,
)
from handshake.context.errors import ContextViewError
from handshake.context.managed_state import ManagedContextState, compacted_end
from handshake.context.projection import stale_marker
from handshake.context.render import render_context_view
from handshake.context.state import (
    ContextView,
    HistoryEpisode,
    HistorySegment,
    RawHistorySegment,
    SequencedMessage,
)
from handshake.models.base import ToolDefinition
from handshake.models.errors import ModelProtocolError, ModelProviderError
from handshake.models.protocol import ModelMessage

RawMessageReader = Callable[[str, int], ModelMessage]
ContextViewReader = Callable[[], ContextView]


def mask_oldest_stale_result(state: ManagedContextState) -> bool:
    """Mask one oldest tool result outside protected recent context."""
    for sequence in range(compacted_end(state), state.recent_start):
        message = state.messages[sequence]
        if message.role != "user":
            continue
        for index, block in enumerate(message.content):
            if block.get("type") != "tool_result":
                continue
            if (
                state.stale_mask_through is not None
                and (sequence, index) <= state.stale_mask_through
            ):
                continue
            block["content"] = stale_marker(sequence, block.get("is_error"))
            state.stale_mask_through = (sequence, index)
            state.reduction_dirty = True
            return True
    return False


async def compact_semantic_context(
    state: ManagedContextState,
    *,
    run_id: str,
    policy: ContextBudgetPolicy,
    estimator: TokenEstimator,
    condenser: HistoryCondenser,
    tools: tuple[ToolDefinition, ...],
    instructions: str,
    measure_pressure: Callable[[], int],
    capture_view: ContextViewReader,
    raw_message_reader: RawMessageReader | None,
) -> None:
    """Advance raw compaction, then roll up Episodes while pressure remains high."""
    while measure_pressure() > policy.pressure_target_tokens:
        background, target = select_raw_candidate(
            state,
            run_id=run_id,
            policy=policy,
            estimator=estimator,
            raw_message_reader=raw_message_reader,
        )
        if not target:
            break
        try:
            episodes = await condenser.compact_raw(background, target)
            if not episodes:
                break
            expected = target[0].sequence
            for episode in episodes:
                validate_episode(episode, expected, target[-1].sequence)
                end = episode.source_range.end_sequence
                if end + 1 < state.message_count and same_tool_round(
                    SequencedMessage(end, state.messages[end]),
                    SequencedMessage(end + 1, state.messages[end + 1]),
                ):
                    raise ContextViewError("Episode splits a tool round")
                expected = end + 1
        except (ContextViewError, ModelProtocolError, ModelProviderError):
            return
        if not _accept_episodes(
            state,
            tools=tools,
            instructions=instructions,
            estimator=estimator,
            capture_view=capture_view,
            episodes=state.episodes + episodes,
        ):
            break

    while measure_pressure() > policy.pressure_target_tokens:
        selected: tuple[HistoryEpisode, ...] = ()
        for episode in state.episodes:
            proposed = (*selected, episode)
            if (
                estimator.estimate_text(compact_json(episode_data(proposed)))
                > policy.rollup_episode_tokens
                or not _fits_request(proposed, None, policy, estimator)
            ):
                break
            selected = proposed
        if len(selected) < 2:
            break
        try:
            merged = await condenser.roll_up(selected)
            if merged is None:
                break
            start = selected[0].source_range.start_sequence
            end = selected[-1].source_range.end_sequence
            validate_episode(merged, start, end)
            if merged.source_range.end_sequence != end:
                raise ContextViewError("Roll-up must cover its complete input")
        except (ContextViewError, ModelProtocolError, ModelProviderError):
            return
        if not _accept_episodes(
            state,
            tools=tools,
            instructions=instructions,
            estimator=estimator,
            capture_view=capture_view,
            episodes=(merged, *state.episodes[len(selected) :]),
        ):
            break


def same_tool_round(previous: SequencedMessage, current: SequencedMessage) -> bool:
    if previous.message.role != "assistant" or current.message.role != "user":
        return False
    if previous.message.run_id != current.message.run_id:
        return False
    calls = {
        block.get("id")
        for block in previous.message.content
        if block.get("type") == "tool_use"
    }
    results = {
        block.get("tool_use_id")
        for block in current.message.content
        if block.get("type") == "tool_result"
    }
    return bool(calls & results)


def validate_episode(episode: HistoryEpisode, start: int, maximum: int) -> None:
    source = episode.source_range
    if (
        type(source.start_sequence) is not int
        or type(source.end_sequence) is not int
        or source.start_sequence != start
        or not start <= source.end_sequence <= maximum
        or not isinstance(episode.title, str)
        or not episode.title.strip()
        or not isinstance(episode.summary, str)
        or not episode.summary.strip()
    ):
        raise ContextViewError("Invalid condenser episode range or content")


def _fits_request(
    episodes: tuple[HistoryEpisode, ...],
    target: tuple[SequencedMessage, ...] | None,
    policy: ContextBudgetPolicy,
    estimator: TokenEstimator,
) -> bool:
    return (
        estimate_model_context(condenser_context(episodes, target), (), estimator)
        <= policy.condenser_input_tokens
    )


def select_raw_candidate(
    state: ManagedContextState,
    *,
    run_id: str,
    policy: ContextBudgetPolicy,
    estimator: TokenEstimator,
    raw_message_reader: RawMessageReader | None,
) -> tuple[tuple[HistoryEpisode, ...], tuple[SequencedMessage, ...]]:
    """Select bounded Episode context and the next uncompacted raw prefix."""
    background: tuple[HistoryEpisode, ...] = ()
    for episode in reversed(state.episodes):
        proposed = (episode, *background)
        if estimator.estimate_text(compact_json(episode_data(proposed))) > (
            policy.background_episode_tokens
        ):
            break
        background = proposed
    target: tuple[SequencedMessage, ...] = ()
    pending: list[SequencedMessage] = []
    for sequence in range(compacted_end(state), state.recent_start):
        message = deepcopy(state.messages[sequence])
        if message.role == "user":
            for index, block in enumerate(message.content):
                if block.get("type") == "tool_result" and block.get("is_error") is True:
                    raw = (
                        raw_message_reader(run_id, sequence)
                        if raw_message_reader is not None
                        else state.messages[sequence]
                    )
                    block["content"] = deepcopy(raw.content[index].get("content", ""))
        pending.append(SequencedMessage(sequence, message))
        if sequence + 1 < state.message_count and same_tool_round(
            pending[-1], SequencedMessage(sequence + 1, state.messages[sequence + 1])
        ):
            continue
        proposed_target = (*target, *pending)
        if estimator.estimate_text(compact_json(raw_data(proposed_target))) > (
            policy.raw_target_tokens
        ):
            break
        while background and not _fits_request(
            background, proposed_target, policy, estimator
        ):
            background = background[1:]
        if not _fits_request(background, proposed_target, policy, estimator):
            break
        target = proposed_target
        pending = []
    return background, target


def _accept_episodes(
    state: ManagedContextState,
    *,
    tools: tuple[ToolDefinition, ...],
    instructions: str,
    estimator: TokenEstimator,
    capture_view: ContextViewReader,
    episodes: tuple[HistoryEpisode, ...],
) -> bool:
    view = capture_view()
    end = episodes[-1].source_range.end_sequence + 1
    history: tuple[HistorySegment, ...] = episodes
    if end < state.recent_start:
        history += (
            RawHistorySegment(
                tuple(
                    SequencedMessage(sequence, state.messages[sequence])
                    for sequence in range(end, state.recent_start)
                )
            ),
        )
    proposed = render_context_view(
        replace(view, history=history), instructions=instructions, tools=tools
    )
    if estimate_model_context(proposed, tools, estimator) >= estimate_model_context(
        render_context_view(view, instructions=instructions, tools=tools),
        tools,
        estimator,
    ):
        return False
    state.episodes = episodes
    for sequence in tuple(state.messages):
        if sequence < end:
            del state.messages[sequence]
    state.reduction_dirty = True
    return True
