"""Stable references for model-readable artifacts stored outside the workspace."""

from keyline_agent.artifacts.local import (
    ArtifactReferenceError,
    StoredTextArtifact,
    open_text_artifact,
    store_text_artifact,
)

__all__ = [
    "ArtifactReferenceError",
    "StoredTextArtifact",
    "open_text_artifact",
    "store_text_artifact",
]
