from pathlib import Path

import pytest

from handshake.tools.files import (
    file_tool_bindings,
    read_file,
    search_files,
    write_file,
)
from handshake.tools.registry import ToolRegistry
from handshake.tools.shell import process_tool_bindings


def test_file_module_owns_definition_handler_bindings() -> None:
    bindings = file_tool_bindings()

    assert [binding.definition.name for binding in bindings] == [
        "read_file",
        "search_files",
        "write_file",
    ]
    assert [binding.handler for binding in bindings] == [
        read_file,
        search_files,
        write_file,
    ]


def test_shell_module_binds_both_tools_to_injected_approval(tmp_path: Path) -> None:
    approvals: list[tuple[Path, tuple[str, ...]]] = []

    def approve(cwd: Path, argv: tuple[str, ...]) -> bool:
        approvals.append((cwd, argv))
        return False

    bindings = process_tool_bindings(approve)
    result = bindings[0].handler(tmp_path, {"argv": ["not-executed"]})

    assert [binding.definition.name for binding in bindings] == ["run_shell", "run_tests"]
    assert bindings[0].handler is bindings[1].handler
    assert not result.ok
    assert approvals == [(tmp_path.resolve(), ("not-executed",))]


def test_registry_rejects_duplicate_binding_name() -> None:
    binding = file_tool_bindings()[0]
    registry = ToolRegistry()
    registry.register(binding)

    with pytest.raises(ValueError, match="read_file"):
        registry.register(binding)
