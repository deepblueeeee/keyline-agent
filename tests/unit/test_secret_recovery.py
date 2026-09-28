from pathlib import Path

from handshake.inputs import EphemeralSecretStore
from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.loop import prepare_run_for_resume, start_run
from handshake.runtime.secret_dependencies import (
    resolve_secret_dependency,
)
from handshake.runtime.state import (
    DependencyStatus,
    Run,
    RunStatus,
    SecretHandleStatus,
    Session,
)
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry
from handshake.tools.secret_command import SecretCommandRunner


def _secret_wait_turn() -> ModelTurn:
    call = ToolCall(
        "ask-secret",
        "wait_for_secret",
        {
            "prompt": "Enter the integration key",
            "purpose": "run the integration test",
            "environment_variable": "TEST_API_KEY",
        },
    )
    return ModelTurn(
        content=(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            },
        ),
        tool_calls=(call,),
        stop_reason="tool_use",
    )


def test_available_secret_is_reissued_after_process_restart(tmp_path: Path) -> None:
    canary = "restart-canary-value"
    state_store = SQLiteStateStore(tmp_path / "state.db")
    first_secret_store = EphemeralSecretStore()
    session = Session(workspace=tmp_path)
    run = Run(goal="integration", session_id=session.session_id, agent_spec_id="test")
    state_store.create_session(session)
    start_run(
        session,
        run,
        ScriptedModel([_secret_wait_turn()]),
        create_default_registry(lambda _cwd, _argv: True),
        store=state_store,
    )
    requested_snapshot = state_store.load_run(run.run_id)
    assert requested_snapshot.status is RunStatus.WAITING
    assert requested_snapshot.dependencies[0].status is DependencyStatus.OPEN
    assert requested_snapshot.secret_handles[0].status is SecretHandleStatus.REQUESTED
    original_dependency = run.dependencies[0]
    original_handle = run.secret_handles[0]
    continuation = resolve_secret_dependency(
        session,
        run,
        original_dependency.dependency_id,
        canary,
        first_secret_store.writer(),
    )
    state_store.checkpoint_run(session, run, (continuation,))

    resumed_session = state_store.load_session(session.session_id)
    resumed_run = state_store.load_run(run.run_id)
    empty_process_store = EphemeralSecretStore()
    runner = SecretCommandRunner(
        empty_process_store.consumer(), lambda _cwd, _argv: True
    )

    assert prepare_run_for_resume(
        resumed_session,
        resumed_run,
        runner,
        store=state_store,
    )

    assert resumed_run.status is RunStatus.WAITING
    assert resumed_run.dependencies[0].status is DependencyStatus.LOST
    assert resumed_run.secret_handles[0].status is SecretHandleStatus.LOST
    assert resumed_run.dependencies[1].status is DependencyStatus.OPEN
    assert resumed_run.secret_handles[1].status is SecretHandleStatus.REQUESTED
    assert resumed_run.secret_handles[1].secret_handle != original_handle.secret_handle
    assert (
        resumed_run.secret_handles[1].dependency_id
        == resumed_run.dependencies[1].dependency_id
    )
    assert canary.encode() not in state_store.database_path.read_bytes()

    loaded_again = state_store.load_run(run.run_id)
    assert loaded_again.secret_handles[0].status is SecretHandleStatus.LOST
    assert loaded_again.secret_handles[1].status is SecretHandleStatus.REQUESTED
