"""Stable references for model-readable artifacts stored outside the workspace."""

from handshake.artifacts.local import (
    ArtifactReferenceError,
    StoredTextArtifact,
    resolve_artifact_reference,
    store_text_artifact,
)

__all__ = [
    "ArtifactReferenceError",
    "StoredTextArtifact",
    "resolve_artifact_reference",
    "store_text_artifact",
]
