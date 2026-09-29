"""Model-visible context management."""

from handshake.context.budget import (
    ContextBudgetPolicy,
    TokenEstimator,
    Utf8TokenEstimator,
    estimate_model_context,
)
from handshake.context.condenser import HistoryCondenser, LLMHistoryCondenser
from handshake.context.errors import ContextPressureError, ContextViewError
from handshake.context.manager import ActiveContextManagers, AgentContextManager
from handshake.context.state import (
    ContextCheckpoint,
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
    "ActiveContextManagers",
    "AgentContextManager",
    "ContextBudgetPolicy",
    "ContextCheckpoint",
    "ContextDependency",
    "ContextPressureError",
    "ContextView",
    "ContextViewError",
    "ContextWorkItem",
    "ExactContext",
    "HistoryEpisode",
    "HistoryCondenser",
    "LLMHistoryCondenser",
    "HistorySegment",
    "RawHistorySegment",
    "RecentContext",
    "SequencedMessage",
    "SourceRange",
    "TokenEstimator",
    "Utf8TokenEstimator",
    "estimate_model_context",
]
