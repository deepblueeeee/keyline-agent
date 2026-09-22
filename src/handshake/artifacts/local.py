"""Local storage behind opaque, model-facing artifact references."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from handshake.config import artifact_directory

_REFERENCE_PREFIX = "@handshake/artifacts/"
_ARTIFACT_ID = re.compile(r"[0-9a-f]{64}")


class ArtifactReferenceError(ValueError):
    """Raised when a Handshake artifact reference is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class StoredTextArtifact:
    reference: str
    byte_count: int
    character_count: int
    line_count: int


def store_text_artifact(key: str, content: str) -> StoredTextArtifact:
    """Atomically store text under a deterministic opaque artifact reference."""
    artifact_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
    root = artifact_directory().expanduser().resolve()
    directory = root / artifact_id[:2]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ArtifactReferenceError("Artifact directory escapes its configured root")

    path = directory / f"{artifact_id}.txt"
    temporary = directory / f".{path.name}.tmp"
    encoded = content.encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return StoredTextArtifact(
        reference=f"{_REFERENCE_PREFIX}{artifact_id}",
        byte_count=len(encoded),
        character_count=len(content),
        line_count=len(content.splitlines()),
    )


def _artifact_path(reference: str) -> Path | None:
    if reference in {_REFERENCE_PREFIX.removesuffix("/"), _REFERENCE_PREFIX}:
        raise ArtifactReferenceError("Artifact references must identify one artifact")
    if not reference.startswith(_REFERENCE_PREFIX):
        return None
    artifact_id = reference.removeprefix(_REFERENCE_PREFIX)
    if _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise ArtifactReferenceError("Invalid Handshake artifact reference")

    root = artifact_directory().expanduser().resolve()
    resolved = (root / artifact_id[:2] / f"{artifact_id}.txt").resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ArtifactReferenceError("Artifact reference escapes its configured root")
    return resolved


def open_text_artifact(reference: str) -> TextIO | None:
    """Open an artifact text stream; ordinary workspace paths return None."""
    path = _artifact_path(reference)
    if path is None:
        return None
    try:
        return path.open(encoding="utf-8", newline="")
    except FileNotFoundError:
        raise ArtifactReferenceError(f"Artifact does not exist: {reference}") from None
