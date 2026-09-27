"""Immutable model-visible context view types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from handshake.context.budget import ContextTokenMeasurement
from handshake.models.protocol import ModelMessage


@dataclass(frozen=True, slots=True)
class ContextWorkItem:
    work_id: str
    description: str
    status: str


@dataclass(frozen=True, slots=True)
class ContextDependency:
    dependency_id: str
    work_id: str
    tool_call_id: str
    kind: str
    prompt: str
    status: str


@dataclass(frozen=True, slots=True)
class ExactContext:
    """Authoritative current state, preserved verbatim rather than summarized.

    Constraints must be supplied explicitly; they are not inferred from messages.
    Dependency answers remain in the transcript until modeled explicitly.
    """

    goal: str
    workspace: str
    plan: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    work_items: tuple[ContextWorkItem, ...] = ()
    dependencies: tuple[ContextDependency, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceRange:
    """Inclusive, zero-based range in the durable Session message sequence."""

    start_sequence: int
    end_sequence: int


@dataclass(frozen=True, slots=True)
class SequencedMessage:
    """A model-visible protocol message paired with its durable source sequence."""

    sequence: int
    message: ModelMessage


@dataclass(frozen=True, slots=True)
class RawHistorySegment:
    """A continuous, semantically uncompressed older history interval."""

    messages: tuple[SequencedMessage, ...]


@dataclass(frozen=True, slots=True)
class HistoryEpisode:
    """Semantic representation of a continuous source interval, eligible for roll-up."""

    source_range: SourceRange
    title: str
    summary: str


HistorySegment: TypeAlias = RawHistorySegment | HistoryEpisode


@dataclass(frozen=True, slots=True)
class RecentContext:
    """Protected history exposed without semantic summarization."""

    messages: tuple[SequencedMessage, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextView:
    """The complete boundary between durable history and model-visible input."""

    exact: ExactContext
    history: tuple[HistorySegment, ...] = ()
    protected_recent: RecentContext = field(default_factory=RecentContext)
    source_message_count: int = 0


@dataclass(frozen=True, slots=True)
class ContextCheckpoint:
    """Persisted recipe for rebuilding a bounded model-visible context."""

    source_message_count: int = 0
    recent_start: int = 0
    episodes: tuple[HistoryEpisode, ...] = ()
    stale_mask_through: tuple[int, int] | None = None
    max_tool_result_tokens: int = 16_000
    reduced_message_count: int = 0
    revision: int = 0
    token_measurement: ContextTokenMeasurement | None = None
