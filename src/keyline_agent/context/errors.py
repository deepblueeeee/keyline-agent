"""Errors raised while constructing a model-visible context view."""


class ContextViewError(Exception):
    """Raised when a ContextView does not map its source history exactly once."""


class ContextPressureError(Exception):
    """Raised when the model input remains above its hard context limit."""
