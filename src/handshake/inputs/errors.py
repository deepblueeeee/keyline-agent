"""Errors owned by the input layer."""

from handshake.errors import HandshakeError


class SecretStoreError(HandshakeError):
    """Raised when an ephemeral Secret capability cannot perform its operation."""
