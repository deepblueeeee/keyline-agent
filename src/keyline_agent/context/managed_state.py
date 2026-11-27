"""Mutable bounded state owned by AgentContextManager."""

from __future__ import annotations

from dataclasses import dataclass, field

from keyline_agent.context.budget import ContextTokenMeasurement
from keyline_agent.context.errors import ContextViewError
from keyline_agent.context.state import HistoryEpisode
from keyline_agent.models.protocol import ModelMessage


@dataclass(slots=True)
class ManagedContextState:
    message_count: int = 0
    recent_start: int = 0
    messages: dict[int, ModelMessage] = field(default_factory=dict)
    episodes: tuple[HistoryEpisode, ...] = ()
    stale_mask_through: tuple[int, int] | None = None
    max_tool_result_tokens: int = 16_000
    reduced_message_count: int = 0
    checkpoint_revision: int = 0
    reduction_dirty: bool = False
    token_measurement: ContextTokenMeasurement | None = None


def compacted_end(state: ManagedContextState) -> int:
    return state.episodes[-1].source_range.end_sequence + 1 if state.episodes else 0


def validate_managed_state(state: ManagedContextState) -> None:
    if not 0 <= state.recent_start <= state.message_count:
        raise ContextViewError("Context state has an invalid protected recent boundary")
    start = compacted_end(state)
    if start > state.recent_start:
        raise ContextViewError("Compacted history overlaps protected recent context")
    if len(state.messages) != state.message_count - start:
        raise ContextViewError("Context raw tail must cover every uncompacted source sequence")
    for sequence in range(start, state.message_count):
        if sequence not in state.messages:
            raise ContextViewError(
                "Context raw tail must cover every uncompacted source sequence"
            )
