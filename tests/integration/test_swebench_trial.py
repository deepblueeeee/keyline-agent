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

from evals import _swebench_tools, swebench, swebench_run, swebench_workspace  # noqa: E402
from evals.swebench import dataset_snapshot  # noqa: E402
from evals.swebench_workspace import DockerWorkspace  # noqa: E402
from evals.tasks import load_tasks  # noqa: E402
from keyline_agent.artifacts import store_text_artifact  # noqa: E402
from keyline_agent.models.errors import ModelProviderError  # noqa: E402
from keyline_agent.models.protocol import (  # noqa: E402
    ModelContext,
    ModelTurn,
    ModelUsage,
    ToolCall,
)
from keyline_agent.models.scripted import ScriptedModel  # noqa: E402
from keyline_agent.runtime.recovery import resume_run, update_paused_step_limit  # noqa: E402
from keyline_agent.storage.sqlite import SQLiteStateStore  # noqa: E402

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
    baseline = tmp_path / "baseline.git"
    shutil.copytree(checkout, scorer)
    shutil.copytree(checkout / ".git", baseline)

    attributes = baseline / "info" / "attributes"
    attributes.write_bytes(b"example.txt diff\n")
    (checkout / "example.txt").write_bytes(changed)
    expected = git("diff", "--binary")
    patch = _swebench_tools.export_patch(checkout, baseline, base).decode("utf-8")
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
    monkeypatch: pytest.MonkeyPatch, http_status: int, expected_attempts: int
) -> None:
    pause = Mock()
    monkeypatch.setattr(swebench_workspace, "sleep", pause)
    client = Mock()
    client.images.get.side_effect = docker.errors.ImageNotFound("not cached")
    error = docker.errors.APIError("pull failed", response=Mock(status_code=http_status))
    client.images.pull.side_effect = error
    workspace = DockerWorkspace(client, "base", image="fixture:latest@sha256:pin")

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
    trial_workspace.close.assert_called_once_with()
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


def test_paused_trial_records_checkpoint_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trial_workspace: Mock,
) -> None:
    async def pause_agent(*args: object) -> dict[str, str]:
        return {"status": "paused"}

    monkeypatch.setattr(swebench_run, "run_agent", pause_agent)
    trial_workspace.export_patch.return_value = ""
    trial_workspace.checkpoint.return_value = "checkpoint-image"
    directory = tmp_path / "trial"

    result = swebench_run.run_trial(TASK, directory, BUDGET)

    assert result["workspace_checkpoint"] == "checkpoint-image"
    assert json.loads((directory / "result.json").read_text()) == result
    trial_workspace.checkpoint.assert_called_once_with()
    trial_workspace.close.assert_called_once_with()


def test_trial_preserves_patch_if_container_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trial_workspace: Mock,
) -> None:
    trial_workspace.export_patch.return_value = "candidate patch"
    trial_workspace.close.side_effect = RuntimeError("container removal failed")
    evaluate = Mock()
    monkeypatch.setattr(swebench_run, "evaluate", evaluate)
    directory = tmp_path / "trial"

    result = swebench_run.run_trial(TASK, directory, BUDGET)

    assert result["status"] == "error"
    assert result["failed_stage"] == "workspace_cleanup"
    assert not result["official_scoring_attempted"]
    assert (directory / "candidate.patch").read_text() == "candidate patch"
    evaluate.assert_not_called()


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
        monkeypatch.delenv("KEYLINE_HOME", raising=False)
    else:
        monkeypatch.setenv("KEYLINE_HOME", previous)
    trial_workspace.export_patch.return_value = ""
    directory = tmp_path / "trial"

    def prepare() -> None:
        assert os.environ.get("KEYLINE_HOME") == str(directory)
        if error is not None:
            raise error

    trial_workspace.prepare.side_effect = prepare
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            swebench_run.run_trial(TASK, directory, BUDGET)
    else:
        result = swebench_run.run_trial(TASK, directory, BUDGET)
        assert result["status"] == ("error" if error else "empty_patch")
    trial_workspace.close.assert_called_once_with()
    assert os.environ.get("KEYLINE_HOME") == previous


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


def test_eval_command_tools_advertise_and_enforce_the_same_timeout() -> None:
    workspace = DockerWorkspace(Mock(), "base", image="fixture-image")
    registry, _ = swebench_run.agent_setup(workspace, "scripted")
    definitions = {definition.name: definition for definition in registry.definitions()}

    for name in ("run_shell", "run_tests"):
        schema = definitions[name].input_schema["properties"]["timeout_seconds"]
        assert schema["maximum"] == 60
        result = registry.execute(name, workspace.path, {"argv": ["true"], "timeout_seconds": 61})
        assert not result.ok and "at most 60" in result.stderr


def test_timed_out_command_returns_output_already_written() -> None:
    result = _swebench_tools.run_command(
        ["sh", "-c", "printf 'started\\n'; printf 'warning\\n' >&2; sleep 3"], 0.5,
    )

    assert not result.ok and result.exit_code is None
    assert result.stdout == "started\n"
    assert result.stderr == "warning\nCommand timed out after 0.5s"


def test_unsafe_container_is_removed_before_any_tool_executes() -> None:
    client = Mock()
    container = client.containers.create.return_value
    container.attrs = {"HostConfig": {"NetworkMode": "bridge"}, "Mounts": []}
    workspace = DockerWorkspace(client, "base", image="fixture-image")
    with pytest.raises(RuntimeError, match="isolation verification failed"):
        workspace.prepare()
    container.start.assert_not_called()
    container.remove.assert_called_once_with(force=True, v=True)


@pytest.mark.skipif(
    os.environ.get("KEYLINE_EVAL_DOCKER_TEST") != "1", reason="requires Docker image"
)
def test_runtime_and_docker_share_checkout_and_export_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (task,) = load_swebench_dataset(dataset_snapshot(TASK), "test", [INSTANCE_ID])
    spec = make_test_spec(task, namespace="swebench", instance_image_tag=TASK.image_tag)
    monkeypatch.setenv("KEYLINE_HOME", str(tmp_path))
    command = """import errno, os, socket, subprocess, sympy
from pathlib import Path
assert sympy.__file__.startswith('/testbed/')
credentials = ('DEEPSEEK_API_KEY', 'DOCKER_HOST', 'HTTP_PROXY', 'HTTPS_PROXY')
assert not any(x in os.environ for x in credentials)
assert os.geteuid() != 0
assert not Path('/var/run/docker.sock').exists()
assert not Path('/eval.sh').exists()
assert not os.access('/.keyline-agent-eval/tools.pyz', os.W_OK)
assert not os.access('/.keyline-agent-eval/baseline.git/config', os.W_OK)
assert subprocess.check_output(['git','rev-list','--all','--count'], text=True).strip() == '1'
s = socket.socket(); s.settimeout(1)
assert s.connect_ex(('192.0.2.1',80)) == errno.ENETUNREACH
p = Path('task1b-check.txt'); assert p.read_text() == 'file tool\\n'; p.write_text('shell tool\\n')
Path('AUTHORS').unlink(); Path('README.md').write_text('modified by test\\n')
Path('/tmp/task1b-marker').write_text('container-local state')
"""
    calls = (
        ToolCall(
            "write", "write_file", {"path": "/testbed/task1b-check.txt", "content": "file tool\n"}
        ),
        ToolCall("shell", "run_shell", {"argv": ["python", "-c", command]}),
        ToolCall("read", "read_file", {"path": "/testbed/task1b-check.txt"}),
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

    with closing(docker.from_env(timeout=120)) as client:
        checkpoint = None
        try:
            with closing(DockerWorkspace(
                client, task["base_commit"], image=spec.instance_image_key
            )) as workspace:
                workspace.prepare()
                container_id = workspace.container.id
                assert workspace.export_patch() == ""
                stats = asyncio.run(swebench_run.run_agent(
                    workspace, "Exercise the shared tools", "scripted",
                    ScriptedModel(turns[:2]), tmp_path, swebench_run.Budget(steps=2),
                ))
                assert stats["status"] == "paused"
                marker = workspace.execute(workspace.path, {"argv": ["cat", "/tmp/task1b-marker"]})
                assert marker.stdout == "container-local state"
                artifact = store_text_artifact("test-trial", "host artifact")
                assert workspace.execute_file(
                    "read_file", workspace.path, {"path": artifact.reference}
                ).stdout == "host artifact"
                timeout = workspace.execute(
                    workspace.path,
                    {
                        "argv": ["sh", "-c", "(sleep 1; touch /tmp/late-output) & wait"],
                        "timeout_seconds": 0.5,
                    },
                )
                assert not timeout.ok and timeout.exit_code is None
                assert "timed out" in timeout.stderr
                assert workspace.execute(
                    workspace.path, {"argv": ["sh", "-c", "sleep 1; test ! -e /tmp/late-output"]}
                ).ok
                registry, _ = swebench_run.agent_setup(workspace, "scripted")
                pipeline = "(printf 'test failed\\n'; exit 7) | tail -n 1"
                for shell, flag in (("bash", "-lc"), ("/bin/sh", "-c")):
                    test_result = registry.execute(
                        "run_tests", workspace.path, {"argv": [shell, flag, pipeline]}
                    )
                    assert not test_result.ok and test_result.exit_code == 7
                    assert test_result.stdout == "test failed\n"
                normal_exit = workspace.execute(workspace.path, {"argv": ["sh", "-c", "exit 124"]})
                assert normal_exit.exit_code == 124 and not normal_exit.stderr
                assert not workspace.execute(
                    workspace.path, {"argv": ["no-command-task1b-test"]}
                ).ok
                patch = workspace.export_patch()
                assert "+++ b/task1b-check.txt" in patch and "+++ b/README.md" in patch
                assert "--- a/AUTHORS" in patch
                checkpoint = workspace.checkpoint()
            with pytest.raises(docker.errors.NotFound):
                client.containers.get(container_id)

            fork_db = tmp_path / "fork.db"
            shutil.copyfile(tmp_path / "state.db", fork_db)
            store = SQLiteStateStore(fork_db)
            run = store.load_run(stats["run_id"])
            session = store.load_session(run.session_id)
            assert session.workspace == Path("/testbed") == workspace.path
            update_paused_step_limit(session, run, 4, store=store)
            with closing(DockerWorkspace(
                client, task["base_commit"], image=spec.instance_image_key
            )) as restored:
                restored.prepare(checkpoint=checkpoint)
                restored_id = restored.container.id
                registry, agent_spec = swebench_run.agent_setup(restored, "scripted")
                asyncio.run(resume_run(
                    session, run, ScriptedModel(turns[2:]), registry, store=store,
                    agent_spec=agent_spec,
                ))
                resumed = store.load_run(run.run_id)
                assert resumed.status == "completed"
                assert all(tool.result and tool.result.ok for tool in resumed.tool_runs)
                assert "shell tool" in resumed.tool_runs[-1].result.stdout
                marker = restored.execute(restored.path, {"argv": ["cat", "/tmp/task1b-marker"]})
                assert marker.stdout == "container-local state"
                assert restored.export_patch() == patch
                with pytest.raises(RuntimeError, match="Unsupported workspace file: ./pipe"):
                    restored.execute(restored.path, {"argv": ["mkfifo", "pipe"]})
                with pytest.raises(RuntimeError, match="Unsupported workspace file: ./pipe"):
                    restored.export_patch()
            with pytest.raises(docker.errors.NotFound):
                client.containers.get(restored_id)
        finally:
            if checkpoint:
                client.images.remove(checkpoint)
