from pathlib import Path

from handshake.tools.files import read_file, search_files, write_file


def test_read_search_and_write_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("first\nneedle here\nlast\n", encoding="utf-8")

    read_result = read_file(
        tmp_path, {"path": "src/example.py", "start_line": 2, "end_line": 2}
    )
    search_result = search_files(tmp_path, {"query": "needle", "path": "src"})
    write_result = write_file(
        tmp_path, {"path": "new/nested.txt", "content": "written"}
    )

    assert read_result.ok and read_result.stdout == "needle here\n"
    assert search_result.ok and search_result.stdout == "src/example.py:2:needle here"
    assert write_result.ok
    assert (tmp_path / "new" / "nested.txt").read_text(encoding="utf-8") == "written"


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    from handshake.tools import create_default_registry

    registry = create_default_registry(lambda _cwd, _argv: True)
    result = registry.execute("read_file", tmp_path, {"path": "../outside.txt"})

    assert not result.ok
    assert "inside the workspace" in result.stderr


def test_symlink_escape_is_rejected_by_registry(tmp_path: Path) -> None:
    from handshake.tools import create_default_registry

    outside = tmp_path.parent / "outside-handshake-test.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    registry = create_default_registry(lambda _cwd, _argv: True)

    result = registry.execute("read_file", tmp_path, {"path": "link.txt"})

    assert not result.ok
    assert "escapes" in result.stderr
    outside.unlink()


def test_search_skips_binary_files_and_caches(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"needle\x00data")
    cache = tmp_path / ".git"
    cache.mkdir()
    (cache / "ignored.txt").write_text("needle", encoding="utf-8")

    result = search_files(tmp_path, {"query": "needle"})

    assert result.ok
    assert result.stdout == ""
