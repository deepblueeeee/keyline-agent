import shutil
import sys
from pathlib import Path

from handshake.models.protocol import ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.loop import start_run
from handshake.runtime.state import Run, RunStatus, Session
from handshake.tools import create_default_registry
from tests.history_support import fresh_store


def _tool_turn(call: ToolCall) -> ModelTurn:
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


def test_scripted_model_completes_repository_edit(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[2] / "examples" / "minimal_task"
    workspace = tmp_path / "workspace"
    shutil.copytree(fixture, workspace)
    model = ScriptedModel(
        [
            _tool_turn(
                ToolCall(
                    "list",
                    "list_files",
                    {"pattern": "*.py"},
                )
            ),
            _tool_turn(ToolCall("read", "read_file", {"path": "calculator.py"})),
            _tool_turn(
                ToolCall(
                    "search",
                    "search_files",
                    {"query": "NotImplementedError", "path": "."},
                )
            ),
            _tool_turn(
                ToolCall("plan", "update_plan", {"steps": ["Edit", "Test"]})
            ),
            _tool_turn(
                ToolCall(
                    "edit",
                    "edit_file",
                    {
                        "path": "calculator.py",
                        "old_text": "    raise NotImplementedError",
                        "new_text": "    return left + right",
                    },
                )
            ),
            _tool_turn(
                ToolCall(
                    "test",
                    "run_tests",
                    {"argv": [sys.executable, "-m", "pytest", "-q"]},
                )
            ),
            ModelTurn(
                content=({"type": "text", "text": "Implemented and tested add."},),
                stop_reason="end_turn",
            ),
        ]
    )
    approvals: list[tuple[Path, tuple[str, ...]]] = []

    def approve(cwd: Path, argv: tuple[str, ...]) -> bool:
        approvals.append((cwd, argv))
        return True

    session = Session(workspace=workspace)
    run = Run(goal="Implement add", session_id=session.session_id)
    start_run(
        session,
        run,
        model,
        create_default_registry(approve),
        store=fresh_store(tmp_path, session),
    )

    assert run.status is RunStatus.COMPLETED
    assert "return left + right" in (workspace / "calculator.py").read_text()
    assert run.tool_runs[-1].result is not None
    assert run.tool_runs[-1].result.ok
    assert len(approvals) == 1
    assert [tool_run.tool_name for tool_run in run.tool_runs] == [
        "list_files",
        "read_file",
        "search_files",
        "edit_file",
        "run_tests",
    ]
    assert model.contexts[3].plan == ()
    assert model.contexts[4].plan == ("Edit", "Test")
    assert run.plan == ["Edit", "Test"]
