"""Workspace-confined file discovery, reading, search, and editing tools."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from handshake.models.base import ToolDefinition
from handshake.runtime.errors import ToolExecutionError
from handshake.runtime.state import ToolResult
from handshake.tools.registry import ToolBinding

_DEFAULT_READ_LINES = 200
_MAX_READ_LINES = 1000
_MAX_READ_CHARACTERS = 50_000
_DEFAULT_LIST_RESULTS = 200
_DEFAULT_SEARCH_RESULTS = 50
_DEFAULT_GLOB_PATTERN = "**/*"
_MAX_RESULTS = 1000
_MAX_MATCH_CHARACTERS = 500

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


def _optional_string(arguments: Mapping[str, Any], name: str, default: str) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f"{name} must be a non-empty string")
    return value


def _optional_int(arguments: Mapping[str, Any], name: str, default: int, *, minimum: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ToolExecutionError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_bool(arguments: Mapping[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise ToolExecutionError(f"{name} must be a boolean")
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


def _bounded_result(text: str, notice: str) -> str:
    if len(text) <= _MAX_READ_CHARACTERS:
        return f"{text}{notice}"
    clipped = text[:_MAX_READ_CHARACTERS]
    return (
        f"{clipped}\n\n[read_file: output truncated at "
        f"{_MAX_READ_CHARACTERS} characters; request a narrower line range]"
    )


def read_file(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    """Read a bounded, 1-based line range instead of returning a whole large file."""
    raw_path = _required_string(arguments, "path")
    path = resolve_workspace_path(workspace, raw_path)
    if not path.is_file():
        raise ToolExecutionError(f"File does not exist: {raw_path}")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(lines)
    start = _optional_int(arguments, "start_line", 1, minimum=1)
    raw_end = arguments.get("end_line")
    if raw_end is None:
        end = start + _DEFAULT_READ_LINES - 1
    else:
        end = _optional_int(arguments, "end_line", start, minimum=1)
    if end < start:
        raise ToolExecutionError("end_line must be greater than or equal to start_line")
    if end - start + 1 > _MAX_READ_LINES:
        raise ToolExecutionError(f"read_file accepts at most {_MAX_READ_LINES} lines per call")

    selected = "".join(lines[start - 1 : end])
    actual_end = min(end, total_lines)
    if start > total_lines and (total_lines > 0 or start > 1):
        notice = (
            f"\n[read_file: start_line {start} is past the end of {raw_path} ({total_lines} lines)]"
        )
    elif start > 1 or actual_end < total_lines:
        next_hint = (
            f"; continue with start_line={actual_end + 1}" if actual_end < total_lines else ""
        )
        notice = f"\n[read_file: showing lines {start}-{actual_end} of {total_lines}{next_hint}]"
    else:
        notice = ""
    return ToolResult(ok=True, stdout=_bounded_result(selected, notice))


def _iter_workspace_files(root: Path, workspace: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    workspace_root = workspace.resolve()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name for name in directory_names if name not in _IGNORED_DIRECTORIES
        )
        base = Path(directory)
        for file_name in sorted(file_names):
            candidate = base / file_name
            try:
                if candidate.resolve().is_relative_to(workspace_root):
                    yield candidate
            except OSError:
                continue


def _matches_glob(path: Path, root: Path, pattern: str) -> bool:
    """Match a root-relative glob where only a complete ``**`` segment recurses."""
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ToolExecutionError("pattern must be relative and must not contain '..'")

    match_root = root if root.is_dir() else root.parent
    path_parts = path.relative_to(match_root).parts
    pattern_parts = pattern_path.parts

    @cache
    def match_from(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        segment = pattern_parts[pattern_index]
        if segment == "**":
            return match_from(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and match_from(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], segment)
            and match_from(pattern_index + 1, path_index + 1)
        )

    return match_from(0, 0)


def list_files(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    """List a bounded set of workspace files, optionally filtered by glob."""
    raw_path = _optional_string(arguments, "path", ".")
    pattern = _optional_string(arguments, "pattern", _DEFAULT_GLOB_PATTERN)
    max_results = _optional_int(arguments, "max_results", _DEFAULT_LIST_RESULTS, minimum=1)
    if max_results > _MAX_RESULTS:
        raise ToolExecutionError(f"max_results must be <= {_MAX_RESULTS}")
    root = resolve_workspace_path(workspace, raw_path)
    if not root.exists():
        raise ToolExecutionError(f"List path does not exist: {raw_path}")

    workspace_root = workspace.resolve()
    matches: list[str] = []
    truncated = False
    for path in _iter_workspace_files(root, workspace):
        if not _matches_glob(path, root, pattern):
            continue
        if len(matches) == max_results:
            truncated = True
            break
        matches.append(path.relative_to(workspace_root).as_posix())
    if truncated:
        matches.append(f"[list_files: stopped after {max_results} files; narrow path or pattern]")
    return ToolResult(ok=True, stdout="\n".join(matches))


def search_files(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    """Search bounded text-file line matches for a literal string."""
    query = _required_string(arguments, "query")
    raw_path = _optional_string(arguments, "path", ".")
    pattern = _optional_string(arguments, "pattern", _DEFAULT_GLOB_PATTERN)
    case_sensitive = _optional_bool(arguments, "case_sensitive", True)
    max_results = _optional_int(arguments, "max_results", _DEFAULT_SEARCH_RESULTS, minimum=1)
    if max_results > _MAX_RESULTS:
        raise ToolExecutionError(f"max_results must be <= {_MAX_RESULTS}")
    root = resolve_workspace_path(workspace, raw_path)
    if not root.exists():
        raise ToolExecutionError(f"Search path does not exist: {raw_path}")

    matches: list[str] = []
    workspace_root = workspace.resolve()
    compared_query = query if case_sensitive else query.casefold()
    truncated = False
    for path in _iter_workspace_files(root, workspace):
        if not _matches_glob(path, root, pattern):
            continue
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(workspace_root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            compared_line = line if case_sensitive else line.casefold()
            if compared_query not in compared_line:
                continue
            if len(matches) == max_results:
                truncated = True
                break
            matches.append(f"{relative}:{line_number}:{line[:_MAX_MATCH_CHARACTERS]}")
        if truncated:
            break
    if truncated:
        matches.append(
            f"[search_files: stopped after {max_results} matches; narrow path, pattern, or query]"
        )
    return ToolResult(ok=True, stdout="\n".join(matches))


def edit_file(workspace: Path, arguments: Mapping[str, Any]) -> ToolResult:
    """Apply a guarded, exact text replacement to an existing UTF-8 file."""
    raw_path = _required_string(arguments, "path")
    old_text = _required_string(arguments, "old_text")
    new_text = arguments.get("new_text")
    if not isinstance(new_text, str):
        raise ToolExecutionError("new_text must be a string")
    if old_text == new_text:
        raise ToolExecutionError("old_text and new_text must differ")
    expected = _optional_int(arguments, "expected_replacements", 1, minimum=1)
    if expected > _MAX_RESULTS:
        raise ToolExecutionError(f"expected_replacements must be <= {_MAX_RESULTS}")

    path = resolve_workspace_path(workspace, raw_path)
    if not path.is_file():
        raise ToolExecutionError(f"File does not exist: {raw_path}")
    content = path.read_text(encoding="utf-8")
    actual = content.count(old_text)
    if actual != expected:
        raise ToolExecutionError(
            f"Expected {expected} exact occurrence(s) of old_text in {raw_path}, "
            f"found {actual}; file was not changed"
        )

    updated = content.replace(old_text, new_text)
    path.write_text(updated, encoding="utf-8")
    return ToolResult(
        ok=True,
        stdout=f"Applied {actual} replacement(s) to {raw_path}",
    )


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
                name="list_files",
                description=(
                    "List workspace files recursively by default with bounded output. An explicit "
                    "glob is relative to path: '*.py' is shallow; '**/*.py' is recursive."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_RESULTS,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            handler=list_files,
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="read_file",
                description=(
                    f"Read a UTF-8 workspace file by 1-based line range. Without end_line, "
                    f"returns at most {_DEFAULT_READ_LINES} lines and provides a continuation hint."
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
                    "Search UTF-8 workspace files recursively by default for a literal string and "
                    "return bounded path:line matches. An explicit glob is relative to path: "
                    "'*.py' is shallow; '**/*.py' is recursive."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                        "case_sensitive": {"type": "boolean"},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_RESULTS,
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
                name="edit_file",
                description=(
                    "Apply a targeted patch to an existing UTF-8 file by replacing an exact "
                    "old_text block. Defaults to requiring exactly one match and makes no change "
                    "if the expected count differs."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string", "minLength": 1},
                        "new_text": {"type": "string"},
                        "expected_replacements": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": _MAX_RESULTS,
                        },
                    },
                    "required": ["path", "old_text", "new_text"],
                    "additionalProperties": False,
                },
            ),
            handler=edit_file,
        ),
        ToolBinding(
            definition=ToolDefinition(
                name="write_file",
                description=(
                    "Create or fully replace a UTF-8 workspace file. Prefer edit_file for "
                    "targeted changes to an existing file."
                ),
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
