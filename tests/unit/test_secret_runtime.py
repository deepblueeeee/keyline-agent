import sys
from pathlib import Path

from handshake.inputs import EphemeralSecretStore
from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.loop import continue_run, start_run
from handshake.runtime.secret_dependencies import resolve_secret_dependency
from handshake.runtime.state import (
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    SecretHandleStatus,
    Session,
    WorkItemStatus,
)
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry
from handshake.tools.secret_command import SecretCommandRunner


def _tool_turn(*calls: ToolCall) -> ModelTurn:
    return ModelTurn(
        content=tuple(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments,
            }
            for call in calls
        ),
        tool_calls=calls,
        stop_reason="tool_use",
    )


def _final_turn() -> ModelTurn:
    return ModelTurn(
        content=({"type": "text", "text": "secure integration test passed"},),
        stop_reason="end_turn",
    )


def test_secret_dependency_and_command_complete_without_model_leak(
    tmp_path: Path,
) -> None:
    canary = "runtime-canary-secret"
    state_store = SQLiteStateStore(tmp_path / "state.db")
    secret_store = EphemeralSecretStore()
    writer = secret_store.writer()
    runner = SecretCommandRunner(
        secret_store.consumer(), lambda _cwd, _argv: True
    )
    session = Session(workspace=tmp_path)
    run = Run(goal="run authenticated integration test", session_id=session.session_id)
    state_store.create_session(session)
    wait_call = ToolCall(
        "ask-secret",
        "wait_for_secret",
        {
            "prompt": "Enter the integration test API key",
            "purpose": "authenticate the local integration test",
            "environment_variable": "TEST_API_KEY",
        },
    )
    first_model = ScriptedModel([_tool_turn(wait_call)])

    start_run(
        session,
        run,
        first_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=state_store,
        secret_command_runner=runner,
    )

    dependency = run.dependencies[0]
    handle = run.secret_handles[0]
    assert run.status is RunStatus.WAITING
    assert run.work_items[0].status is WorkItemStatus.READY
    assert dependency.kind is DependencyKind.SECRET
    assert dependency.status is DependencyStatus.OPEN
    assert handle.status is SecretHandleStatus.REQUESTED
    assert handle.secret_handle.startswith("secret_")
    assert len(handle.secret_handle) > 30
    immediate = session.messages[-1].content[0]
    assert immediate["tool_use_id"] == wait_call.id
    assert handle.secret_handle in immediate["content"]
    assert handle.consumer in immediate["content"]
    assert handle.environment_variable in immediate["content"]
    wait_definition = next(
        tool for tool in first_model.tool_sets[0] if tool.name == "wait_for_secret"
    )
    assert wait_definition.input_schema["required"] == [
        "prompt",
        "purpose",
        "environment_variable",
    ]

    resolve_secret_dependency(
        session, run, dependency.dependency_id, canary, writer
    )
    assert canary not in repr(session.messages)
    assert canary.encode() not in state_store.database_path.read_bytes()

    secure_call = ToolCall(
        "secure-test",
        "run_command_with_secret",
        {
            "secret_handle": handle.secret_handle,
            "argv": [
                sys.executable,
                "-c",
                "import os; print(os.environ['TEST_API_KEY'])",
            ],
        },
    )
    second_model = ScriptedModel([_tool_turn(secure_call), _final_turn()])
    continue_run(
        session,
        run,
        second_model,
        create_default_registry(lambda _cwd, _argv: True),
        store=state_store,
        secret_command_runner=runner,
    )

    assert run.status is RunStatus.COMPLETED
    assert handle.status is SecretHandleStatus.CONSUMED
    assert run.tool_runs_for(run.work_items[0])[-1].arguments == secure_call.arguments
    assert canary not in repr(second_model.contexts)
    assert canary not in repr(session.messages)
    assert canary not in repr(run.tool_runs)
    assert canary.encode() not in state_store.database_path.read_bytes()
    secure_result = second_model.contexts[-1].messages[-1].content[0]
    assert "[REDACTED]" in secure_result["content"]
    assert canary not in secure_result["content"]


def test_secret_wait_mixed_with_other_tool_has_no_side_effect(tmp_path: Path) -> None:
    state_store = SQLiteStateStore(tmp_path / "state.db")
    session = Session(workspace=tmp_path)
    run = Run(goal="safe request", session_id=session.session_id)
    state_store.create_session(session)
    model = ScriptedModel(
        [
            _tool_turn(
                ToolCall(
                    "ask",
                    "wait_for_secret",
                    {
                        "prompt": "Key?",
                        "purpose": "test",
                        "environment_variable": "TEST_API_KEY",
                    },
                ),
                ToolCall(
                    "write",
                    "write_file",
                    {"path": "should-not-exist", "content": "bad"},
                ),
            ),
            _final_turn(),
        ]
    )

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
        store=state_store,
    )

    assert run.status is RunStatus.COMPLETED
    assert run.dependencies == []
    assert run.secret_handles == []
    assert not (tmp_path / "should-not-exist").exists()


def test_secret_command_without_capability_fails_without_shell_fallback(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [
            _tool_turn(
                ToolCall(
                    "secure",
                    "run_command_with_secret",
                    {"secret_handle": "secret_public", "argv": ["touch", "unsafe"]},
                )
            ),
            _final_turn(),
        ]
    )
    session = Session(workspace=tmp_path)
    run = Run(goal="no fallback", session_id=session.session_id)

    start_run(
        session,
        run,
        model,
        create_default_registry(lambda _cwd, _argv: True),
    )

    result = run.tool_runs[-1].result
    assert result is not None and not result.ok
    assert "no shell fallback" in result.stderr
    assert not (tmp_path / "unsafe").exists()
