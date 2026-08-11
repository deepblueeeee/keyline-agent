from pathlib import Path

from handshake.cli import main


def test_demo_runs_without_api_key(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = main(["demo"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Task completed" in captured.out
    assert "verified the fixture tests pass" in captured.out


def test_real_run_requires_api_key(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    exit_code = main(["run", str(tmp_path), "inspect the repository"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "DEEPSEEK_API_KEY is required" in captured.err
