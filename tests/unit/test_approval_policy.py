from __future__ import annotations

from pathlib import Path

import pytest

from keyline_agent.config.approval import ApprovalMode, requires_approval


@pytest.mark.parametrize(
    "argv",
    [
        ("pwd",),
        ("ls", "-la", "."),
        ("git", "status", "--short"),
        ("git", "diff", "--no-ext-diff"),
        ("rg", "-n", "needle", "src"),
        ("ruff", "check", "src"),
    ],
)
def test_dangerous_only_skips_prompt_for_recognized_commands(
    tmp_path: Path, argv: tuple[str, ...]
) -> None:
    assert not requires_approval(ApprovalMode.DANGEROUS_ONLY, tmp_path, argv)
    assert requires_approval(ApprovalMode.ALWAYS_ASK, tmp_path, argv)


@pytest.mark.parametrize(
    "argv",
    [
        ("rm", "-rf", "."),
        ("git", "reset", "--hard"),
        ("git", "diff", "--output=changed.txt"),
        ("sh", "-c", "rm -rf ."),
        ("python", "-c", "print('execute')"),
        ("pytest", "--basetemp", "../outside"),
        ("ruff", "check", "--fix", "src"),
        ("pytest", "tests/unit", "-q"),
        ("python", "-m", "pytest", "tests/unit", "-q"),
        ("mypy", "src"),
        ("rg", "needle", "--pre=script"),
        ("rg", "needle", "../outside"),
        ("unknown-tool",),
        ("./pytest", "-q"),
    ],
)
def test_dangerous_only_prompts_for_dangerous_or_unknown_commands(
    tmp_path: Path, argv: tuple[str, ...]
) -> None:
    assert requires_approval(ApprovalMode.DANGEROUS_ONLY, tmp_path, argv)
    assert not requires_approval(ApprovalMode.ALLOW_ALL, tmp_path, argv)


def test_workspace_path_symlink_does_not_bypass_classification(tmp_path: Path) -> None:
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)

    assert requires_approval(
        ApprovalMode.DANGEROUS_ONLY,
        tmp_path,
        ("rg", "needle", "outside"),
    )


def test_secret_command_is_sensitive_in_middle_mode(tmp_path: Path) -> None:
    argv = ("rg", "needle", "src")

    assert requires_approval(
        ApprovalMode.DANGEROUS_ONLY, tmp_path, argv, with_secret=True
    )
    assert not requires_approval(
        ApprovalMode.ALLOW_ALL, tmp_path, argv, with_secret=True
    )


def test_middle_mode_does_not_autoapprove_home_directory_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert requires_approval(ApprovalMode.DANGEROUS_ONLY, tmp_path, ("rg", "needle"))
