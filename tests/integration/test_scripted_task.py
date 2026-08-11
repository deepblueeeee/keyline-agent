import shutil
import sys
from pathlib import Path

from handshake.models.scripted import ScriptedModel
from handshake.runtime.loop import run_agent
from handshake.runtime.state import ModelTurn, Task, TaskStatus, ToolCall
from handshake.tools import create_default_registry


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
    replacement = "def add(left: int, right: int) -> int:\n    return left + right\n"
    model = ScriptedModel(
        [
            _tool_turn(
                ToolCall("plan", "update_plan", {"steps": ["Inspect", "Edit", "Test"]})
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
                ToolCall(
                    "write", "write_file", {"path": "calculator.py", "content": replacement}
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

    task = Task(goal="Implement add", workspace=workspace)
    run_agent(task, model, create_default_registry(approve))

    assert task.status is TaskStatus.COMPLETED
    assert "return left + right" in (workspace / "calculator.py").read_text()
    assert task.tool_runs[-1].result is not None
    assert task.tool_runs[-1].result.ok
    assert len(approvals) == 1
