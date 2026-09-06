from pathlib import Path

from handshake.tools.files import edit_file, list_files, read_file, search_files, write_file


def test_read_search_and_write_file(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("first\nneedle here\nlast\n", encoding="utf-8")

    read_result = read_file(tmp_path, {"path": "src/example.py", "start_line": 2, "end_line": 2})
    search_result = search_files(tmp_path, {"query": "needle", "path": "src"})
    write_result = write_file(tmp_path, {"path": "new/nested.txt", "content": "written"})

    assert read_result.ok
    assert read_result.stdout == (
        "needle here\n\n[read_file: showing lines 2-2 of 3; continue with start_line=3]"
    )
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


def test_read_file_defaults_to_bounded_page_with_continuation(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("".join(f"line {number}\n" for number in range(1, 206)), encoding="utf-8")

    first = read_file(tmp_path, {"path": "large.txt"})
    second = read_file(tmp_path, {"path": "large.txt", "start_line": 201})

    assert first.ok
    assert "line 1\n" in first.stdout
    assert "line 200\n" in first.stdout
    assert "line 201\n" not in first.stdout
    assert "showing lines 1-200 of 205; continue with start_line=201" in first.stdout
    assert second.ok
    assert second.stdout == (
        "line 201\nline 202\nline 203\nline 204\nline 205\n"
        "\n[read_file: showing lines 201-205 of 205]"
    )


def test_read_file_rejects_unbounded_explicit_range(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("content", encoding="utf-8")

    from handshake.tools import create_default_registry

    result = create_default_registry(lambda _cwd, _argv: True).execute(
        "read_file", tmp_path, {"path": "large.txt", "start_line": 1, "end_line": 1001}
    )

    assert not result.ok
    assert "at most 1000 lines" in result.stderr


def test_read_file_bounds_a_single_very_long_line(tmp_path: Path) -> None:
    (tmp_path / "minified.js").write_text("x" * 60_000, encoding="utf-8")

    result = read_file(tmp_path, {"path": "minified.js"})

    assert result.ok
    assert len(result.stdout) < 51_000
    assert "output truncated at 50000 characters" in result.stdout


def test_read_file_reports_a_page_past_an_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    result = read_file(tmp_path, {"path": "empty.txt", "start_line": 2})

    assert result.ok
    assert "start_line 2 is past the end of empty.txt (0 lines)" in result.stdout


def test_list_files_filters_with_glob_and_bounds_results(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "notes.txt").write_text("", encoding="utf-8")

    result = list_files(tmp_path, {"path": "src", "pattern": "*.py", "max_results": 1})

    assert result.ok
    assert result.stdout.splitlines() == [
        "src/a.py",
        "[list_files: stopped after 1 files; narrow path or pattern]",
    ]


def test_glob_only_recurses_with_explicit_globstar(tmp_path: Path) -> None:
    source = tmp_path / "src"
    nested = source / "package"
    nested.mkdir(parents=True)
    (source / "top.py").write_text("needle\n", encoding="utf-8")
    (nested / "nested.py").write_text("needle\n", encoding="utf-8")

    default_list = list_files(tmp_path, {"path": "src"})
    shallow_list = list_files(tmp_path, {"path": "src", "pattern": "*.py"})
    recursive_list = list_files(tmp_path, {"path": "src", "pattern": "**/*.py"})
    default_search = search_files(tmp_path, {"path": "src", "query": "needle"})
    shallow_search = search_files(
        tmp_path, {"path": "src", "pattern": "*.py", "query": "needle"}
    )
    recursive_search = search_files(
        tmp_path, {"path": "src", "pattern": "**/*.py", "query": "needle"}
    )

    assert default_list.stdout.splitlines() == ["src/top.py", "src/package/nested.py"]
    assert shallow_list.stdout.splitlines() == ["src/top.py"]
    assert recursive_list.stdout.splitlines() == ["src/top.py", "src/package/nested.py"]
    assert default_search.stdout.splitlines() == [
        "src/top.py:1:needle",
        "src/package/nested.py:1:needle",
    ]
    assert shallow_search.stdout.splitlines() == ["src/top.py:1:needle"]
    assert recursive_search.stdout.splitlines() == [
        "src/top.py:1:needle",
        "src/package/nested.py:1:needle",
    ]


def test_search_files_supports_glob_and_case_insensitive_matching(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("Needle here\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle there\n", encoding="utf-8")

    result = search_files(
        tmp_path,
        {"query": "needle", "pattern": "*.py", "case_sensitive": False},
    )

    assert result.ok
    assert result.stdout == "code.py:1:Needle here"


def test_edit_file_applies_unique_targeted_replacement(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    path.write_text("before\nold block\nafter\n", encoding="utf-8")

    result = edit_file(
        tmp_path,
        {"path": "example.py", "old_text": "old block", "new_text": "new block"},
    )

    assert result.ok
    assert result.stdout == "Applied 1 replacement(s) to example.py"
    assert path.read_text(encoding="utf-8") == "before\nnew block\nafter\n"


def test_edit_file_does_not_change_ambiguous_match(tmp_path: Path) -> None:
    path = tmp_path / "example.py"
    original = "same\nsame\n"
    path.write_text(original, encoding="utf-8")

    from handshake.tools import create_default_registry

    result = create_default_registry(lambda _cwd, _argv: True).execute(
        "edit_file",
        tmp_path,
        {"path": "example.py", "old_text": "same", "new_text": "changed"},
    )

    assert not result.ok
    assert "found 2; file was not changed" in result.stderr
    assert path.read_text(encoding="utf-8") == original
