"""Explicit domain errors for the Handshake runtime."""


class HandshakeError(Exception):
    """Base class for errors safe to present to a user."""


class InvalidStateTransitionError(HandshakeError):
    """Raised when a task or tool run attempts an invalid transition."""


class ModelProviderError(HandshakeError):
    """Raised when a model provider cannot produce a usable turn."""


class ModelProtocolError(HandshakeError):
    """Raised when a model response violates the expected protocol."""


class ToolExecutionError(HandshakeError):
    """Raised for invalid tool arguments or workspace access."""


class StepLimitExceededError(HandshakeError):
    """Raised when the agent exceeds its configured model-turn budget."""


class AgentLoopError(HandshakeError):
    """Raised when the runtime cannot safely continue a task."""


class UserCancelledError(HandshakeError):
    """Raised when the user interrupts an approval prompt."""


class StorageError(HandshakeError):
    """Raised when durable task state cannot be read or written safely."""


class TaskNotFoundError(StorageError):
    """Raised when a requested durable task does not exist."""


class WaitResolutionError(HandshakeError):
    """Raised when text input cannot satisfy the requested wait condition."""
