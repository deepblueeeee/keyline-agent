import json
import sys
from pathlib import Path

from keyline_agent.inputs import EphemeralSecretStore
from keyline_agent.runtime.errors import UserCancelledError
from keyline_agent.tools.shell import process_tool_handler


def test_process_success_and_nonzero_exit(tmp_path: Path) -> None:
    handler = process_tool_handler(lambda _cwd, _argv: True)

    success = handler(
        tmp_path, {"argv": [sys.executable, "-c", "print('ok')"]}
    )
    failure = handler(
        tmp_path,
        {"argv": [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"]},
    )

    assert success.ok and success.stdout == "ok\n" and success.exit_code == 0
    assert not failure.ok and failure.stdout == "bad\n" and failure.exit_code == 7


def test_process_denial_does_not_execute(tmp_path: Path) -> None:
    handler = process_tool_handler(lambda _cwd, _argv: False)

    result = handler(
        tmp_path,
        {"argv": [sys.executable, "-c", "open('created', 'w').write('no')"]},
    )

    assert not result.ok
    assert "denied" in result.stderr
    assert not (tmp_path / "created").exists()


def test_process_timeout_is_failure(tmp_path: Path) -> None:
    handler = process_tool_handler(lambda _cwd, _argv: True)

    result = handler(
        tmp_path,
        {
            "argv": [sys.executable, "-c", "import time; time.sleep(0.2)"],
            "timeout_seconds": 0.02,
        },
    )

    assert not result.ok
    assert "timed out" in result.stderr
    assert result.exit_code is None


def test_process_does_not_inherit_provider_key(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DEEPSEEK_API_KEY", "canary-super-secret")
    handler = process_tool_handler(lambda _cwd, _argv: True)

    result = handler(
        tmp_path,
        {
            "argv": [
                sys.executable,
                "-c",
                "import json, os; print(json.dumps(os.environ.get('DEEPSEEK_API_KEY')))",
            ]
        },
    )

    assert result.ok
    assert json.loads(result.stdout) is None
    assert "canary-super-secret" not in result.stdout


def test_approval_cancellation_propagates(tmp_path: Path) -> None:
    def cancel(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        raise UserCancelledError("cancelled")

    handler = process_tool_handler(cancel)

    try:
        handler(tmp_path, {"argv": [sys.executable, "-V"]})
    except UserCancelledError as exc:
        assert str(exc) == "cancelled"
    else:  # pragma: no cover
        raise AssertionError("Expected cancellation")


def test_normal_shell_can_observe_handle_but_cannot_obtain_secret(
    tmp_path: Path,
) -> None:
    canary = "normal-shell-must-not-see-this"
    handle = "secret_public_opaque_handle"
    store = EphemeralSecretStore()
    store.writer().put(handle, canary)
    handler = process_tool_handler(lambda _cwd, _argv: True)

    result = handler(
        tmp_path,
        {
            "argv": [
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "print(sys.argv[1]); "
                    "print(os.environ.get('TEST_API_KEY'))"
                ),
                handle,
            ]
        },
    )

    assert result.ok
    assert handle in result.stdout
    assert "None" in result.stdout
    assert canary not in result.stdout


def test_cancelled_approval_cannot_start_command(tmp_path: Path) -> None:
    from threading import Event

    import pytest

    from keyline_agent.inputs.cancellation import cancellation_scope

    cancelled = Event()

    def late_approval(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        cancelled.set()
        return True

    handler = process_tool_handler(late_approval)
    with cancellation_scope(cancelled), pytest.raises(UserCancelledError):
        handler(tmp_path, {
            "argv": [sys.executable, "-c", "open('started', 'w').write('started')"],
        })
    assert not (tmp_path / "started").exists()
