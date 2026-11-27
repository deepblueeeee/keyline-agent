import sys
from pathlib import Path

from keyline_agent.inputs import EphemeralSecretStore
from keyline_agent.runtime.state import (
    Dependency,
    DependencyKind,
    DependencyStatus,
    Run,
    SecretHandle,
    SecretHandleStatus,
)
from keyline_agent.tools.secret_command import SecretCommandRunner


def _available_secret(
    raw_value: str,
) -> tuple[Run, SecretHandle, object, SecretCommandRunner]:
    run = Run(goal="integration test", session_id="session", agent_spec_id="test")
    dependency = Dependency(
        tool_call_id="ask-secret",
        prompt="Provide test API key",
        kind=DependencyKind.SECRET,
    )
    run.add_dependency(dependency)
    handle = SecretHandle(
        dependency_id=dependency.dependency_id,
        purpose="authenticate the integration test",
        consumer="run_command_with_secret",
        environment_variable="TEST_API_KEY",
    )
    run.add_secret_handle(dependency, handle)
    dependency.transition(DependencyStatus.SATISFIED)
    store = EphemeralSecretStore()
    writer = store.writer()
    consumer = store.consumer()
    writer.put(handle.secret_handle, raw_value)
    handle.transition(SecretHandleStatus.AVAILABLE)
    return run, handle, consumer, SecretCommandRunner(
        consumer, lambda _cwd, _argv: True
    )


def test_secret_command_runner_injects_only_target_and_redacts_output(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    canary = "secure-runner-canary"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "provider-key-must-not-pass")
    run, handle, consumer, runner = _available_secret(canary)

    result = runner.execute(
        run,
        tmp_path,
        {
            "secret_handle": handle.secret_handle,
            "argv": [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "print(os.environ['TEST_API_KEY']); "
                    "print(os.environ.get('DEEPSEEK_API_KEY'))"
                ),
            ],
        },
    )

    assert result.ok
    assert canary not in result.stdout
    assert "[REDACTED]" in result.stdout
    assert "None" in result.stdout
    assert handle.status is SecretHandleStatus.CONSUMED
    assert not consumer.contains(handle.secret_handle)  # type: ignore[attr-defined]


def test_secret_command_approval_happens_before_consumption(tmp_path: Path) -> None:
    class TrackingConsumer:
        def __init__(self) -> None:
            self.consumed = False

        def contains(self, _secret_handle: str) -> bool:
            return True

        def consume(self, _secret_handle: str) -> str:
            self.consumed = True
            return "raw-secret"

    run, handle, _consumer, _runner = _available_secret("unused")
    tracking = TrackingConsumer()

    def deny(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        assert not tracking.consumed
        return False

    result = SecretCommandRunner(tracking, deny).execute(
        run,
        tmp_path,
        {"secret_handle": handle.secret_handle, "argv": [sys.executable, "-V"]},
    )

    assert not result.ok
    assert not tracking.consumed
    assert handle.status is SecretHandleStatus.AVAILABLE


def test_secret_command_fails_closed_for_wrong_or_consumed_handle(tmp_path: Path) -> None:
    canary = "one-time-secret"
    run, handle, _consumer, runner = _available_secret(canary)
    handle.consumer = "another_consumer"

    wrong_consumer = runner.execute(
        run,
        tmp_path,
        {"secret_handle": handle.secret_handle, "argv": [sys.executable, "-V"]},
    )
    assert not wrong_consumer.ok
    assert "wrong consumer" in wrong_consumer.stderr
    assert handle.status is SecretHandleStatus.AVAILABLE

    handle.consumer = "run_command_with_secret"
    first = runner.execute(
        run,
        tmp_path,
        {"secret_handle": handle.secret_handle, "argv": [sys.executable, "-V"]},
    )
    second = runner.execute(
        run,
        tmp_path,
        {"secret_handle": handle.secret_handle, "argv": [sys.executable, "-V"]},
    )

    assert first.ok
    assert not second.ok
    assert "consumed" in second.stderr
    assert canary not in repr(runner)


def test_secret_command_timeout_consumes_and_redacts(tmp_path: Path) -> None:
    canary = "timeout-canary"
    run, handle, consumer, runner = _available_secret(canary)

    result = runner.execute(
        run,
        tmp_path,
        {
            "secret_handle": handle.secret_handle,
            "argv": [
                sys.executable,
                "-c",
                "import os,time; print(os.environ['TEST_API_KEY'], flush=True); time.sleep(1)",
            ],
            "timeout_seconds": 0.05,
        },
    )

    assert not result.ok
    assert "timed out" in result.stderr
    assert canary not in result.stdout
    assert handle.status is SecretHandleStatus.CONSUMED
    assert not consumer.contains(handle.secret_handle)  # type: ignore[attr-defined]


def test_secret_command_rejects_lost_handle_without_consuming(tmp_path: Path) -> None:
    run, handle, consumer, runner = _available_secret("lost-secret")
    handle.transition(SecretHandleStatus.LOST)

    result = runner.execute(
        run,
        tmp_path,
        {"secret_handle": handle.secret_handle, "argv": [sys.executable, "-V"]},
    )

    assert not result.ok
    assert "lost" in result.stderr
    assert consumer.contains(handle.secret_handle)  # type: ignore[attr-defined]


def test_cancelled_approval_does_not_consume_secret_or_start_process(tmp_path: Path) -> None:
    from threading import Event

    import pytest

    from keyline_agent.inputs.cancellation import cancellation_scope
    from keyline_agent.runtime.errors import UserCancelledError

    run, handle, consumer, _runner = _available_secret("secret-cancellation-canary")
    cancelled = Event()

    def late_approval(_cwd: Path, _argv: tuple[str, ...]) -> bool:
        cancelled.set()
        return True

    runner = SecretCommandRunner(consumer, late_approval)
    with cancellation_scope(cancelled), pytest.raises(UserCancelledError):
        runner.execute(run, tmp_path, {
            "secret_handle": handle.secret_handle,
            "argv": [sys.executable, "-c", "open('started', 'w').write('started')"],
        })
    assert handle.status is SecretHandleStatus.AVAILABLE
    assert consumer.contains(handle.secret_handle)  # type: ignore[attr-defined]
    assert not (tmp_path / "started").exists()
