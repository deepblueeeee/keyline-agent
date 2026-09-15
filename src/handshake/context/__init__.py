"""Model-visible context management."""

from handshake.context.errors import ContextViewError
from handshake.context.manager import AgentContextManager
from handshake.context.state import (
    ContextDependency,
    ContextView,
    ContextWorkItem,
    ExactContext,
    HistoryEpisode,
    HistorySegment,
    RawHistorySegment,
    RecentContext,
    SequencedMessage,
    SourceRange,
)

__all__ = [
    "AgentContextManager",
    "ContextDependency",
    "ContextView",
    "ContextViewError",
    "ContextWorkItem",
    "ExactContext",
    "HistoryEpisode",
    "HistorySegment",
    "RawHistorySegment",
    "RecentContext",
    "SequencedMessage",
    "SourceRange",
]
