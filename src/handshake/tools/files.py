"""Workspace-confined file and search tools."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from handshake.models.base import ToolDefinition
from handshake.runtime.errors import ToolExecutionError
from handshake.runtime.state import ToolResult
from handshake.tools.registry import ToolBinding

_IGNORED_DIRECTORIES = {
    ".git",
    ".handshake",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f"{name} must be a non-empty string")
    return value


def _optional_int(
    arguments: Mapping[str, Any], name: str, default: int, *, minimum: int
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ToolExecutionError(f"{name} must be an integer >= {minimum}")
    return value


def resolve_workspace_path(workspace: Path, raw_path: str) -> Path:
    """Resolve a model path while rejecting every workspace escape route."""
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ToolExecutionError("Path must be relative and remain inside the workspace")
    root = workspace.resolve()
    resolved = (root / candidate).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ToolExecutionError("Resolved path escapes the workspace")
    return resolved


def read_file(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    raw_path = _required_string(arguments, "path")
    path = resolve_workspace_path(workspace, raw_path)
    if not path.is_file():
        raise ToolExecutionError(f"File does not exist: {raw_path}")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = _optional_int(arguments, "start_line", 1, minimum=1)
    end = _optional_int(arguments, "end_line", len(lines) or 1, minimum=1)
    if end < start:
        raise ToolExecutionError("end_line must be greater than or equal to start_line")
    return ToolResult(ok=True, stdout="".join(lines[start - 1 : end]))


def _iter_search_files(root: Path, workspace: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    workspace_root = workspace.resolve()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = [
            name for name in directory_names if name not in _IGNORED_DIRECTORIES
        ]
        base = Path(directory)
        for file_name in sorted(file_names):
            candidate = base / file_name
            try:
                if candidate.resolve().is_relative_to(workspace_root):
                    yield candidate
            except OSError:
                continue


def search_files(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    query = _required_string(arguments, "query")
    raw_path = arguments.get("path", ".")
    if not isinstance(raw_path, str) or not raw_path:
        raise ToolExecutionError("path must be a non-empty string")
    max_results = _optional_int(arguments, "max_results", 100, minimum=1)
    if max_results > 1000:
        raise ToolExecutionError("max_results must be <= 1000")
    root = resolve_workspace_path(workspace, raw_path)
    if not root.exists():
        raise ToolExecutionError(f"Search path does not exist: {raw_path}")

    matches: list[str] = []
    workspace_root = workspace.resolve()
    for path in _iter_search_files(root, workspace):
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.resolve().relative_to(workspace_root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(f"{relative}:{line_number}:{line[:500]}")
                if len(matches) >= max_results:
                    return ToolResult(ok=True, stdout="\n".join(matches))
    return ToolResult(ok=True, stdout="\n".join(matches))


def write_file(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    raw_path = _required_string(arguments, "path")
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ToolExecutionError("content must be a string")
    path = resolve_workspace_path(workspace, raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ToolResult(ok=True, stdout=f"Wrote {len(content.encode('utf-8'))} bytes to {raw_path}")


def file_tool_bindings() -> tuple[ToolBinding, ...]:
    """Return file tool definitions already paired with their handlers."""
    return (
        ToolBinding(
            definition=ToolDefinition(
                name="read_file",
                description=(
                    "Read a UTF-8 text file from the workspace, optionally by 1-based line range."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=read_file,
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="search_files",
                description=(
                    "Search workspace text files for a literal string and return path:line matches."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string"},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            handler=search_files,
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="write_file",
                description="Create or fully replace a UTF-8 text file inside the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            handler=write_file,
        ),
    )
