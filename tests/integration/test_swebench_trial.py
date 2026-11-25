import asyncio
import json
import os
import shutil
import subprocess
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("docker")
pytest.importorskip("swebench")

import docker  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402
from swebench.harness.utils import load_swebench_dataset  # noqa: E402

from evals import swebench, swebench_run, swebench_workspace  # noqa: E402
from evals.swebench import dataset_snapshot  # noqa: E402
from evals.swebench_workspace import DockerWorkspace  # noqa: E402
from evals.tasks import load_tasks  # noqa: E402
from handshake.models.errors import ModelProviderError  # noqa: E402
from handshake.models.protocol import ModelContext, ModelTurn, ModelUsage, ToolCall  # noqa: E402
from handshake.models.scripted import ScriptedModel  # noqa: E402
from handshake.storage.sqlite import SQLiteStateStore  # noqa: E402

INSTANCE_ID = "sympy__sympy-20590"
TASK = next(task for task in load_tasks() if task.instance_id == INSTANCE_ID)
BUDGET = swebench_run.Budget()


@pytest.fixture
def trial_workspace(monkeypatch: pytest.MonkeyPatch) -> Mock:
    workspace = Mock()
    client = Mock()
    client.containers.get.side_effect = docker.errors.NotFound("already removed")
    monkeypatch.setattr(docker, "from_env", Mock(return_value=client))
    monkeypatch.setattr(swebench_run, "DockerWorkspace", Mock(return_value=workspace))
    monkeypatch.setattr(swebench_run, "dataset_snapshot", lambda _task: "dataset")
    monkeypatch.setattr(swebench, "dataset_snapshot", lambda _task: "dataset")
    monkeypatch.setattr(
        swebench_run,
        "load_swebench_dataset",
        Mock(return_value=[{"base_commit": "base", "problem_statement": "fixture"}]),
    )
    monkeypatch.setattr(swebench_run, "read_deepseek_api_key", lambda: "test-key")
    monkeypatch.setattr(
        swebench_run, "make_test_spec", Mock(return_value=Mock(instance_image_key="image"))
    )
    monkeypatch.setattr(swebench_run, "DeepSeekAnthropicModel", Mock(return_value=object()))
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    async def run_agent(*args: object) -> dict[str, str]:
        return {"status": "completed"}

    monkeypatch.setattr(swebench_run, "run_agent", run_agent)
    return workspace


@pytest.mark.parametrize(
    ("original", "changed", "binary"),
    [
        (b"first\r\nsecond\r\n", b"first\r\nchanged\r\n", False),
        (b"caf\xe9\nold\n", b"caf\xe9\nnew\n", True),
    ],
)
def test_exported_patch_preserves_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_workspace: Mock,
    original: bytes,
    changed: bytes,
    binary: bool,
) -> None:
    checkout = tmp_path / "workspace"
    checkout.mkdir()

    def git(*args: str, cwd: Path = checkout, data: bytes | None = None) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=data,
            capture_output=True,
            check=True,
        ).stdout

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    (checkout / "example.txt").write_bytes(original)
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.org", "commit", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    scorer = tmp_path / "scorer"
    shutil.copytree(checkout, scorer)
    shutil.copytree(checkout / ".git", tmp_path / "baseline.git")

    attributes = tmp_path / "baseline.git" / "info" / "attributes"
    attributes.write_bytes(b"example.txt diff\n")
    (checkout / "example.txt").write_bytes(changed)
    expected = git("diff", "--binary")
    patch = DockerWorkspace(Mock(), checkout, base, image="fixture-image").export_patch()
    assert attributes.read_bytes() == b"example.txt diff\n"
    run = subprocess.run

    def official_cli(command: list[str], *, cwd: Path, **kwargs: object) -> None:
        prediction = json.loads((cwd / "predictions.jsonl").read_text(encoding="utf-8"))
        data = prediction["model_patch"].encode("utf-8")
        if binary:
            assert b"GIT binary patch" in data
        else:
            assert data == expected
        run(["git", "apply", "-"], cwd=scorer, input=data, check=True)
        assert (scorer / "example.txt").read_bytes() == changed

    monkeypatch.setattr(subprocess, "run", official_cli)
    swebench.evaluate(TASK, patch, output_dir=tmp_path / "results")


@pytest.mark.parametrize(("http_status", "expected_attempts"), [(500, 3), (401, 1)])
def test_pinned_image_pull_retries_transient_errors_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, http_status: int, expected_attempts: int
) -> None:
    pause = Mock()
    monkeypatch.setattr(swebench_workspace, "sleep", pause)
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("not cached")
    error = docker.errors.APIError("pull failed", response=Mock(status_code=http_status))
    client.images.pull.side_effect = error
    workspace = DockerWorkspace(client, tmp_path, "base", image="fixture:latest@sha256:pin")

    with pytest.raises(docker.errors.APIError) as caught:
        workspace._ensure_image()

    assert caught.value is error
    assert client.images.pull.call_count == expected_attempts
    client.images.pull.assert_called_with("fixture@sha256:pin", platform="linux/amd64")
    assert pause.call_count == expected_attempts - 1


def test_empty_patch_is_not_an_official_unresolved_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_workspace: Mock,
) -> None:
    trial_workspace.export_patch.return_value = ""
    evaluate = Mock()
    monkeypatch.setattr(swebench_run, "evaluate", evaluate)
    directory = tmp_path / "trial"
    result = swebench_run.run_trial(TASK, directory, BUDGET)

    evaluate.assert_not_called()
    assert result["agent"]["status"] == "completed"
    assert result["status"] == "empty_patch"
    assert result["resolved"] is None
    assert result["official_scoring_attempted"] is False
    assert result["official"] is None
    assert json.loads((directory / "result.json").read_text()) == result


def test_report_without_test_scores_does_not_count_as_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_workspace: Mock,
) -> None:
    trial_workspace.export_patch.return_value = "candidate patch"
    official = {"resolved": False, "patch_successfully_applied": False}
    monkeypatch.setattr(swebench_run, "evaluate", Mock(return_value=(tmp_path, official, None)))
    result = swebench_run.run_trial(TASK, tmp_path / "trial", BUDGET)
    assert result["status"] == "scoring_failed"
    assert result["official"] == official
    assert result["resolved"] is None


@pytest.mark.parametrize(("fallback_fails", "cli_fails"), [(False, False), (True, True)])
def test_scorer_cleanup_failure_keeps_official_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_workspace: Mock,
    fallback_fails: bool,
    cli_fails: bool,
) -> None:
    trial_workspace.export_patch.return_value = "candidate patch"
    get = docker.from_env.return_value.containers.get
    get.side_effect = None
    container = get.return_value
    if fallback_fails:
        container.remove.side_effect = docker.errors.APIError("cleanup failed")
    official = {"resolved": True, "patch_successfully_applied": True, "tests_status": {}}

    def official_cli(command: list[str], *, cwd: Path, **kwargs: object) -> None:
        run_id = command[command.index("--run_id") + 1]
        report = cwd / "logs" / "run_evaluation" / run_id / "candidate" / INSTANCE_ID
        report.mkdir(parents=True)
        (report / "report.json").write_text(json.dumps({INSTANCE_ID: official}))
        if cli_fails:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", official_cli)
    directory = tmp_path / "trial"
    result = swebench_run.run_trial(TASK, directory, BUDGET)

    assert result["status"] == "scoring_cleanup_failed"
    assert result["official_scoring_attempted"] is True
    assert result["official"] == official
    assert result["resolved"] is True
    scoring_dir = Path(result["scoring_dir"])
    get.assert_called_once_with(f"sweb.eval.{INSTANCE_ID}.{scoring_dir.name}")
    container.remove.assert_called_once_with(force=True)
    assert ("APIError" in result["scoring_cleanup_error"]) is fallback_fails
    assert json.loads((directory / "result.json").read_text()) == result


@pytest.mark.parametrize("previous", [None, "original-state"])
@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_trial_restores_state_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trial_workspace: Mock,
    previous: str | None,
    error: type[BaseException] | None,
) -> None:
    if previous is None:
        monkeypatch.delenv("HANDSHAKE_HOME", raising=False)
    else:
        monkeypatch.setenv("HANDSHAKE_HOME", previous)
    trial_workspace.export_patch.return_value = ""
    directory = tmp_path / "trial"

    def prepare() -> None:
        assert os.environ.get("HANDSHAKE_HOME") == str(directory)
        if error is not None:
            raise error

    trial_workspace.prepare.side_effect = prepare
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            swebench_run.run_trial(TASK, directory, BUDGET)
    else:
        result = swebench_run.run_trial(TASK, directory, BUDGET)
        assert result["status"] == ("error" if error else "empty_patch")
    assert os.environ.get("HANDSHAKE_HOME") == previous


def test_requests_are_capped_and_usage_is_written(tmp_path: Path) -> None:
    turn = ModelTurn(({"type": "text", "text": "done"},), usage=ModelUsage(11, 2, 3, 4))
    path = tmp_path / "usage.jsonl"
    model = swebench_run.BudgetedModel(ScriptedModel([turn]), path, request_limit=1)

    async def run() -> None:
        context = ModelContext("system", "goal", "workspace", (), ())
        assert await model.complete(context, ()) == turn
        with pytest.raises(ModelProviderError, match="request budget exhausted"):
            await model.complete(context, ())

    asyncio.run(run())
    assert json.loads(path.read_text())["usage"] == {
        "input_tokens": 11,
        "output_tokens": 2,
        "cache_write_tokens": 3,
        "cache_read_tokens": 4,
    }


def test_container_cleanup_failure_prevents_export(tmp_path: Path) -> None:
    client = Mock()
    container = client.containers.create.return_value
    container.attrs = {"HostConfig": {"NetworkMode": "bridge"}, "Mounts": []}
    container.remove.side_effect = RuntimeError("failed to remove container")
    workspace = DockerWorkspace(client, tmp_path / "workspace", "base", image="fixture-image")
    with (
        pytest.raises(RuntimeError, match="failed to remove"),
        workspace._container(("true",), bind=False),
    ):
        pytest.fail("unsafe container should never start")
    container.start.assert_not_called()
    with pytest.raises(RuntimeError, match="still in use"):
        workspace.export_patch()


def test_special_files_cannot_be_exported(tmp_path: Path) -> None:
    checkout = tmp_path / "workspace"
    checkout.mkdir()
    os.mkfifo(checkout / "pipe")
    workspace = DockerWorkspace(Mock(), checkout, "base", image="fixture-image")
    with pytest.raises(RuntimeError, match="Unsupported workspace file: pipe"):
        workspace.export_patch()


@pytest.mark.skipif(
    os.environ.get("HANDSHAKE_EVAL_DOCKER_TEST") != "1", reason="requires Docker image"
)
def test_runtime_and_docker_share_checkout_and_export_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (task,) = load_swebench_dataset(dataset_snapshot(TASK), "test", [INSTANCE_ID])
    spec = make_test_spec(task, namespace="swebench", instance_image_tag=TASK.image_tag)
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path))
    command = """import errno, os, socket, subprocess, sympy
from pathlib import Path
assert sympy.__file__.startswith('/testbed/')
credentials = ('DEEPSEEK_API_KEY', 'DOCKER_HOST', 'HTTP_PROXY', 'HTTPS_PROXY')
assert not any(x in os.environ for x in credentials)
assert not Path('/var/run/docker.sock').exists()
assert not Path('/eval.sh').exists()
assert subprocess.check_output(['git','rev-list','--all','--count'], text=True).strip() == '1'
s = socket.socket(); s.settimeout(1)
assert s.connect_ex(('192.0.2.1',80)) == errno.ENETUNREACH
p = Path('task1b-check.txt'); assert p.read_text() == 'host file\\n'; p.write_text('docker file\\n')
Path('AUTHORS').unlink(); Path('README.md').write_text('modified by test\\n')
"""
    calls = (
        ToolCall("write", "write_file", {"path": "task1b-check.txt", "content": "host file\n"}),
        ToolCall("shell", "run_shell", {"argv": ["python", "-c", command]}),
        ToolCall("read", "read_file", {"path": "task1b-check.txt"}),
    )
    turns = [
        ModelTurn(
            (
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                },
            ),
            (call,),
            "tool_use",
        )
        for call in calls
    ]
    turns.append(ModelTurn(({"type": "text", "text": "done"},), stop_reason="end_turn"))

    with closing(docker.from_env()) as client:
        workspace = DockerWorkspace(
            client, tmp_path / "workspace", task["base_commit"], image=spec.instance_image_key
        )
        workspace.prepare()
        assert workspace.export_patch() == ""
        stats = asyncio.run(
            swebench_run.run_agent(
                workspace,
                "Exercise the shared tools",
                "scripted",
                ScriptedModel(turns),
                tmp_path,
                BUDGET,
            )
        )
        assert stats["status"] == "completed"
        run = SQLiteStateStore(tmp_path / "state.db").load_run(stats["run_id"])
        assert all(tool.result and tool.result.ok for tool in run.tool_runs), run.tool_runs
        assert "docker file" in run.tool_runs[-1].result.stdout
        patch = workspace.export_patch()
        assert "+++ b/task1b-check.txt" in patch and "+++ b/README.md" in patch
        assert "--- a/AUTHORS" in patch
        timeout = workspace.execute(
            workspace.path, {"argv": ["sleep", "10"], "timeout_seconds": 0.2}
        )
        assert not timeout.ok and "timed out" in timeout.stderr
        assert not workspace.execute(workspace.path, {"argv": ["no-command-task1b-test"]}).ok
        assert workspace.export_patch() == patch
