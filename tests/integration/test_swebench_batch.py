import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("docker")
pytest.importorskip("swebench")

from evals import swebench_run  # noqa: E402
from evals.tasks import Task, load_tasks, select_tasks  # noqa: E402
from handshake.models.protocol import ModelTurn, ToolCall  # noqa: E402
from handshake.models.scripted import ScriptedModel  # noqa: E402
from handshake.storage.sqlite import SQLiteStateStore  # noqa: E402


def test_dataset_selection() -> None:
    tasks = load_tasks()
    assert len(tasks) >= 3
    for suite in ("regression", "capability"):
        selected = select_tasks(tasks, instance_ids=None, suite=suite)
        assert selected and all(task.suite == suite for task in selected)
    ids = [tasks[0].instance_id, tasks[-1].instance_id]
    assert select_tasks(tasks, instance_ids=ids, suite=None) == (tasks[0], tasks[-1])
    with pytest.raises(ValueError, match="Unknown instance IDs"):
        select_tasks(tasks, instance_ids=["unknown"], suite=None)


def test_batch_continues_and_preserves_trial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = load_tasks()
    budget = swebench_run.Budget(requests=4, steps=3, seconds=20)
    paths: list[Path] = []
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")

    def trial(task: Task, directory: Path, trial_budget: swebench_run.Budget) -> dict[str, Any]:
        assert trial_budget is budget
        paths.append(directory)
        if len(paths) == 1:
            raise RuntimeError("preparation failed")
        directory.mkdir(parents=True)
        resolved = len(paths) == 2
        result = {
            "status": "scored",
            "resolved": resolved,
            "official": {"resolved": resolved},
            "agent": {"status": "failed", "usage": {"output_tokens": 7}},
            "elapsed_seconds": 1,
        }
        (directory / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(swebench_run, "run_trial", trial)
    directory = tmp_path / "batch"
    result = swebench_run.run_batch(tasks, directory, trials=2, budget=budget)
    count = 2 * len(tasks)
    assert result["planned_trials"] == result["recorded_trials"] == count
    assert result["task_count"] == len(tasks)
    assert len(set(paths)) == count
    assert result["counts"] == {"resolved": 1, "unresolved": count - 2, "unscored": 1}
    assert result["usage"] == {"output_tokens": (count - 1) * 7}
    assert [entry["trial"] for entry in result["trials"]] == [1, 2] * len(tasks)
    assert result["trials"][0]["status"] == "error"
    for entry in result["trials"][1:]:
        raw = json.loads((Path(entry["artifact_dir"]) / "result.json").read_text())
        assert {key: entry[key] for key in raw} == raw
    assert json.loads((directory / "summary.json").read_text()) == result
    experiment = json.loads((directory / "experiment.json").read_text())
    assert experiment["git_commit"]
    assert experiment["started_at"].endswith("+00:00")
    assert experiment["model"]["name"] == "fixture-model"
    assert experiment["model"]["max_tokens"] == swebench_run.MAX_OUTPUT_TOKENS
    assert experiment["tasks"] == [task.as_dict() for task in tasks]
    assert experiment["trials_per_task"] == 2
    assert experiment["budget"] == {"requests": 4, "steps": 3, "seconds": 20}
    with pytest.raises(FileExistsError):
        swebench_run.run_batch(tasks, directory, trials=2, budget=budget)
    assert json.loads((directory / "experiment.json").read_text()) == experiment


def test_interrupted_batch_saves_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trial = Mock(side_effect=[{"status": "empty_patch", "resolved": None}, KeyboardInterrupt])
    monkeypatch.setattr(swebench_run, "run_trial", trial)
    directory = tmp_path / "batch"
    result = swebench_run.run_batch(load_tasks(), directory, trials=2, budget=swebench_run.Budget())
    assert trial.call_count == 2
    assert result["interrupted"]
    assert [entry["status"] for entry in result["trials"]] == ["empty_patch", "interrupted"]
    assert result["counts"]["unscored"] == 2
    assert (directory / "summary.json").is_file()


@pytest.mark.skipif(
    os.environ.get("HANDSHAKE_EVAL_DOCKER_TEST") != "1", reason="requires Docker image"
)
def test_batch_cli_starts_each_trial_with_clean_workspace_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance_id = "sympy__sympy-20590"
    calls = (
        ToolCall("read", "read_file", {"path": "eval-trial-marker.txt"}),
        ToolCall("write", "write_file", {"path": "eval-trial-marker.txt", "content": "marker\n"}),
    )
    turns = [
        ModelTurn(
            ({"type": "tool_use", "name": call.name, "id": call.id, "input": call.arguments},),
            (call,),
            "tool_use",
        )
        for call in calls
    ]
    turns.append(ModelTurn(({"type": "text", "text": "done"},), stop_reason="end_turn"))
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setattr(swebench_run, "read_deepseek_api_key", lambda: "fixture")
    monkeypatch.setattr(swebench_run, "DeepSeekAnthropicModel", lambda **_: ScriptedModel(turns))
    official = {"resolved": False, "patch_successfully_applied": True, "tests_status": {}}
    monkeypatch.setattr(swebench_run, "evaluate", Mock(return_value=(tmp_path, official, None)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "swebench_run",
            "--live",
            "--instance-id",
            instance_id,
            "--trials",
            "2",
            "--max-requests",
            "3",
            "--max-steps",
            "3",
            "--agent-seconds",
            "30",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert swebench_run.main() == 0
    (summary,) = tmp_path.glob("*/summary.json")
    entries = json.loads(summary.read_text())["trials"]
    assert len(entries) == 2
    sessions = set()
    for entry in entries:
        directory = Path(entry["artifact_dir"])
        (run,) = SQLiteStateStore(directory / "state.db").list_runs()
        assert not run.tool_runs[0].result.ok
        assert run.tool_runs[1].result.ok
        assert not (directory / "workspace").exists()
        assert "+marker\n" in (directory / "candidate.patch").read_text()
        assert entry["agent"]["status"] == "completed"
        sessions.add(run.session_id)
    assert len(sessions) == 2
