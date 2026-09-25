"""Explicit domain errors for the Handshake runtime."""

from handshake.errors import HandshakeError as HandshakeError
from handshake.models.errors import ModelProtocolError as ModelProtocolError
from handshake.models.errors import ModelProviderError as ModelProviderError


class InvalidStateTransitionError(HandshakeError):
    """Raised when a run or tool execution attempts an invalid transition."""


class ToolExecutionError(HandshakeError):
    """Raised for invalid tool arguments or workspace access."""


class StepLimitExceededError(HandshakeError):
    """Raised when the agent exceeds its configured model-turn budget."""


class AgentLoopError(HandshakeError):
    """Raised when the runtime cannot safely continue a run."""


class UserCancelledError(HandshakeError):
    """Raised when the user interrupts an approval prompt."""


class WaitResolutionError(HandshakeError):
    """Raised when text input cannot satisfy the requested wait condition."""


class SecretInputError(HandshakeError):
    """Raised when Secret input cannot satisfy its requested dependency."""


class SecretStoreError(HandshakeError):
    """Raised when an ephemeral Secret capability cannot perform its operation."""
