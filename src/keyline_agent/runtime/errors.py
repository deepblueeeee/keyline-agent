"""Explicit domain errors for the Keyline runtime."""

from keyline_agent.errors import KeylineError


class InvalidStateTransitionError(KeylineError):
    """Raised when a run or tool execution attempts an invalid transition."""


class ToolExecutionError(KeylineError):
    """Raised for invalid tool arguments or workspace access."""


class AgentLoopError(KeylineError):
    """Raised when the runtime cannot safely continue a run."""


class ToolOutcomeUnknownError(AgentLoopError):
    """A possibly side-effecting tool was interrupted and cannot be retried safely."""


class UserCancelledError(KeylineError):
    """Raised when the user interrupts an approval prompt."""


class WaitResolutionError(KeylineError):
    """Raised when text input cannot satisfy the requested wait condition."""


class SecretInputError(KeylineError):
    """Raised when Secret input cannot satisfy its requested dependency."""
