"""Model-visible context management."""

from keyline_agent.context.budget import (
    ContextBudgetPolicy,
    TokenEstimator,
    Utf8TokenEstimator,
    estimate_model_context,
)
from keyline_agent.context.condenser import HistoryCondenser, LLMHistoryCondenser
from keyline_agent.context.errors import ContextPressureError, ContextViewError
from keyline_agent.context.manager import ActiveContextManagers, AgentContextManager
from keyline_agent.context.state import (
    ContextCheckpoint,
    ContextDependency,
    ContextView,
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
