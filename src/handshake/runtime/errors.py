"""Explicit domain errors for the Handshake runtime."""


class HandshakeError(Exception):
    """Base class for errors safe to present to a user."""


class InvalidStateTransitionError(HandshakeError):
    """Raised when a run or tool execution attempts an invalid transition."""


class ModelProviderError(HandshakeError):
    """Raised when a model provider cannot produce a usable turn."""


class ModelProtocolError(HandshakeError):
    """Raised when a model response violates the expected protocol."""


class ToolExecutionError(HandshakeError):
    """Raised for invalid tool arguments or workspace access."""


class StepLimitExceededError(HandshakeError):
    """Raised when the agent exceeds its configured model-turn budget."""


class AgentLoopError(HandshakeError):
    """Raised when the runtime cannot safely continue a run."""


class UserCancelledError(HandshakeError):
    """Raised when the user interrupts an approval prompt."""


class StorageError(HandshakeError):
    """Raised when durable session state cannot be read or written safely."""


class RunNotFoundError(StorageError):
    """Raised when a requested durable run does not exist."""


class SessionNotFoundError(StorageError):
    """Raised when a requested durable session does not exist."""


class ConcurrentRunUpdateError(StorageError):
    """Raised when a run snapshot was based on an outdated stored version."""


class ConcurrentSessionUpdateError(StorageError):
    """Raised when session conversation was based on an outdated version."""


class WaitResolutionError(HandshakeError):
    """Raised when text input cannot satisfy the requested wait condition."""
