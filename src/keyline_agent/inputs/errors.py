"""Errors owned by the input layer."""

from keyline_agent.errors import KeylineError


class SecretStoreError(KeylineError):
    """Raised when an ephemeral Secret capability cannot perform its operation."""
