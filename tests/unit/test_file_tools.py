from pathlib import Path

import pytest

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


def test_file_tools_accept_absolute_workspace_paths(tmp_path: Path) -> None:
    path = tmp_path / "src" / "example.txt"
    assert write_file(tmp_path, {"path": str(path), "content": "needle\n"}).ok
    assert read_file(tmp_path, {"path": str(path)}).stdout == "needle\n"
    assert list_files(tmp_path, {"path": str(path.parent)}).stdout == "src/example.txt"
    assert search_files(
        tmp_path, {"path": str(path.parent), "query": "needle"}
    ).stdout == "src/example.txt:1:needle"
    assert edit_file(
        tmp_path, {"path": str(path), "old_text": "needle", "new_text": "edited"}
    ).ok
    assert path.read_text(encoding="utf-8") == "edited\n"


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_path_traversal_is_rejected(tmp_path: Path, absolute: bool) -> None:
    from handshake.tools import create_default_registry

    registry = create_default_registry(lambda _cwd, _argv: True)
    relative = "src/../example.txt"
    path = str(tmp_path / relative) if absolute else relative
    result = registry.execute("read_file", tmp_path, {"path": path})

    assert not result.ok
    assert "must not contain '..'" in result.stderr


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_workspace_tools_cannot_read_or_change_user_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    private_home = tmp_path / ".handshake"
    private_home.mkdir()
    config = private_home / "config.toml"
    config.write_text("[approval]\nmode = 'always_ask'\n", encoding="utf-8")
    from handshake.tools import create_default_registry

    registry = create_default_registry(lambda _cwd, _argv: True)
    path = str(config) if absolute else ".handshake/config.toml"
    read = registry.execute("read_file", tmp_path, {"path": path})
    write = registry.execute("write_file", tmp_path, {"path": path, "content": "changed"})

    assert not read.ok
    assert not write.ok
    assert read.stderr == write.stderr
    assert "User configuration and credentials" in read.stderr
    assert config.read_text(encoding="utf-8") == "[approval]\nmode = 'always_ask'\n"


def test_paths_outside_the_workspace_are_rejected(tmp_path: Path) -> None:
    from handshake.tools import create_default_registry

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "workspace-sibling.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    link.symlink_to(outside)
    registry = create_default_registry(lambda _cwd, _argv: True)

    for path in (link.name, str(link), str(outside)):
        result = registry.execute("read_file", workspace, {"path": path})
        assert not result.ok
        assert "escapes the workspace" in result.stderr


def test_search_skips_binary_files_and_caches(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"needle\x00data")
    cache = tmp_path / ".git"
    cache.mkdir()
    (cache / "ignored.txt").write_text("needle", encoding="utf-8")

    result = search_files(tmp_path, {"query": "needle"})

    assert result.ok
    assert "0 UTF-8 files searched" in result.stdout
    assert "binary.dat" not in result.stdout and "ignored.txt" not in result.stdout


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
    assert len(result.stdout) < 41_000
    assert "line selection exceeded 40000 UTF-8 bytes" in result.stdout
    assert "continue exact content with start_character=40001" in result.stdout

    remainder = read_file(
        tmp_path,
        {"path": "minified.js", "start_character": 40_001, "end_character": 50_000},
    )

    assert remainder.stdout.startswith("x" * 10_000)
    assert "showing characters 40001-50000 of 60000" in remainder.stdout
    assert "continue with start_character=50001" in remainder.stdout


def test_read_file_character_and_line_ranges_are_mutually_exclusive(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("abcdef\n", encoding="utf-8")

    from handshake.tools import create_default_registry

    registry = create_default_registry(lambda _cwd, _argv: True)
    mixed = registry.execute(
        "read_file",
        tmp_path,
        {"path": "value.txt", "start_line": 1, "start_character": 1},
    )
    characters = registry.execute(
        "read_file",
        tmp_path,
        {"path": "value.txt", "start_character": 2, "end_character": 4},
    )

    assert not mixed.ok
    assert "mutually exclusive" in mixed.stderr
    assert characters.ok
    assert characters.stdout == (
        "bcd\n[read_file: showing characters 2-4 of 7; continue with start_character=5]"
    )


def test_read_file_accepts_artifact_references_but_not_internal_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "handshake-home"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))

    from handshake.artifacts import open_text_artifact, store_text_artifact
    from handshake.tools import create_default_registry

    artifact = store_text_artifact("test-result", "recover me")
    (output,) = (home / "artifacts").glob("*/*.txt")
    internal = home / "handshake.db"
    internal.write_text("do not expose", encoding="utf-8")

    registry = create_default_registry(lambda _cwd, _argv: True)
    allowed = registry.execute("read_file", workspace, {"path": artifact.reference})
    absolute_rejected = registry.execute("read_file", workspace, {"path": str(output)})
    rejected = registry.execute("read_file", workspace, {"path": str(internal)})
    traversal_rejected = registry.execute(
        "read_file", workspace, {"path": "@handshake/artifacts/../handshake.db"}
    )
    write_rejected = registry.execute(
        "write_file", workspace, {"path": str(output), "content": "changed"}
    )

    assert allowed.ok and allowed.stdout == "recover me"
    assert not absolute_rejected.ok
    assert not rejected.ok
    assert "escapes the workspace" in rejected.stderr
    assert not traversal_rejected.ok
    assert "Invalid Handshake artifact reference" in traversal_rejected.stderr
    assert not write_rejected.ok
    artifact_stream = open_text_artifact(artifact.reference)
    assert artifact_stream is not None
    with artifact_stream:
        assert artifact_stream.read() == "recover me"


def test_artifact_ranges_preserve_crlf_and_cr_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "handshake-home"))

    from handshake.artifacts import store_text_artifact

    content = "first\r\nsecond\rthird\n"
    artifact = store_text_artifact("mixed-newlines", content)

    characters = read_file(
        tmp_path,
        {
            "path": artifact.reference,
            "start_character": 1,
            "end_character": len(content),
        },
    )
    lines = read_file(
        tmp_path,
        {"path": artifact.reference, "start_line": 1, "end_line": 2},
    )

    assert artifact.character_count == len(content)
    assert artifact.line_count == 3
    assert characters.stdout == content
    assert lines.stdout.startswith("first\r\nsecond\r")
    assert "showing lines 1-2 of 3; continue with start_line=3" in lines.stdout


def test_read_file_reports_a_page_past_an_empty_file(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    result = read_file(tmp_path, {"path": "empty.txt", "start_line": 2})

    assert result.ok
    assert "start_line 2 is past the end of empty.txt (0 lines)" in result.stdout


@pytest.mark.parametrize(
    "reference",
    (
        "@handshake/artifacts",
        "@handshake/artifacts/",
        "@handshake/artifacts/not-an-artifact-id",
        f"@handshake/artifacts/{'a' * 64}/child",
    ),
)
def test_artifact_reader_requires_one_opaque_artifact_id(
    reference: str,
) -> None:
    from handshake.artifacts import ArtifactReferenceError, open_text_artifact

    with pytest.raises(ArtifactReferenceError, match="identify one artifact|Invalid"):
        open_text_artifact(reference)


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


def test_search_and_list_glob_semantics(tmp_path: Path) -> None:
    source = tmp_path / "src"
    nested = source / "package"
    deep = nested / "deep"
    deep.mkdir(parents=True)
    (source / "top.py").write_text("needle\n", encoding="utf-8")
    (nested / "test_nested.py").write_text("needle\n", encoding="utf-8")
    (deep / "test_deep.py").write_text("needle\n", encoding="utf-8")

    default_list = list_files(tmp_path, {"path": "src"})
    shallow_list = list_files(tmp_path, {"path": "src", "pattern": "*.py"})
    recursive_list = list_files(tmp_path, {"path": "src", "pattern": "**/*.py"})
    paths = ["src/top.py", "src/package/test_nested.py", "src/package/deep/test_deep.py"]
    assert default_list.stdout.splitlines() == paths
    assert shallow_list.stdout.splitlines() == ["src/top.py"]
    assert recursive_list.stdout.splitlines() == paths

    arguments = {"path": "src", "query": "needle"}
    matches = [f"{path}:1:needle" for path in paths]
    assert search_files(tmp_path, arguments).stdout.splitlines() == matches
    for pattern, expected in (
        ("*.py", matches),
        ("test_*.py", matches[1:]),
        ("./*.py", matches[:1]),
        ("package/*.py", matches[1:2]),
        ("package/**/*.py", matches[1:]),
    ):
        result = search_files(tmp_path, {**arguments, "pattern": pattern})
        assert result.ok and result.stdout.splitlines() == expected


def test_search_files_supports_glob_and_case_insensitive_matching(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("Needle here\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("needle there\n", encoding="utf-8")

    result = search_files(
        tmp_path,
        {"query": "needle", "pattern": "*.py", "case_sensitive": False},
    )

    assert result.ok
    assert result.stdout == "code.py:1:Needle here"


def test_search_without_matches_reports_files_searched(tmp_path: Path) -> None:
    source = tmp_path / "src"
    nested = source / "package"
    nested.mkdir(parents=True)
    (source / "top.py").write_text("other text\n", encoding="utf-8")
    (nested / "example.py").write_text("nested text\n", encoding="utf-8")

    arguments = {"path": "src", "pattern": "*.py", "query": "needle"}
    result = search_files(tmp_path, arguments)

    assert result.ok and "no literal matches" in result.stdout
    assert "path='src'" in result.stdout and "pattern='*.py'" in result.stdout
    assert "2 UTF-8 files searched" in result.stdout
    assert "Try a shorter literal query" in result.stdout
    assert "**/*.py" not in result.stdout


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
    original = "header\n  same same\n" + "same\n" * 5
    path.write_text(original, encoding="utf-8")

    from handshake.tools import create_default_registry

    result = create_default_registry(lambda _cwd, _argv: True).execute(
        "edit_file",
        tmp_path,
        {"path": "example.py", "old_text": "same", "new_text": "changed"},
    )

    assert not result.ok
    assert "found 7; file was not changed" in result.stderr
    assert "2:3, 2:8, 3:1, 4:1, 5:1" in result.stderr
    assert "(+2 more)" in result.stderr
    assert "surrounding text in old_text" in result.stderr
    assert path.read_text(encoding="utf-8") == original
