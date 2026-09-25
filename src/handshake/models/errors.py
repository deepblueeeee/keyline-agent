"""Errors owned by the model layer."""

from handshake.errors import HandshakeError


class ModelProviderError(HandshakeError):
    """Raised when a model provider cannot produce a usable turn."""


class ModelProtocolError(HandshakeError):
    """Raised when a model response violates Anthropic Messages."""
