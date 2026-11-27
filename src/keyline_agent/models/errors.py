"""Errors owned by the model layer."""

from keyline_agent.errors import KeylineError


class ModelProviderError(KeylineError):
    """Raised when a model provider cannot produce a usable turn."""


class ModelProtocolError(KeylineError):
    """Raised when a model response violates Anthropic Messages."""
