"""Explicit domain errors for the Handshake runtime."""

from handshake.errors import HandshakeError


class InvalidStateTransitionError(HandshakeError):
    """Raised when a run or tool execution attempts an invalid transition."""


class ToolExecutionError(HandshakeError):
    """Raised for invalid tool arguments or workspace access."""


class AgentLoopError(HandshakeError):
    """Raised when the runtime cannot safely continue a run."""


class ToolOutcomeUnknownError(AgentLoopError):
    """A possibly side-effecting tool was interrupted and cannot be retried safely."""


class UserCancelledError(HandshakeError):
    """Raised when the user interrupts an approval prompt."""


class WaitResolutionError(HandshakeError):
    """Raised when text input cannot satisfy the requested wait condition."""


class SecretInputError(HandshakeError):
    """Raised when Secret input cannot satisfy its requested dependency."""
