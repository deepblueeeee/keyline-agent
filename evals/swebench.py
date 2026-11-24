"""Run the official SWE-bench CLI with pinned inputs."""

import json
import subprocess
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .tasks import Task


def dataset_snapshot(task: Task) -> str:
    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            task.source,
            repo_type="dataset",
            revision=task.revision,
            allow_patterns=["README.md", "data/*.parquet"],
            token=False,
        )
    )


def evaluate(
    task: Task, git_patch: str | None, *, output_dir: Path
) -> tuple[Path, dict[str, Any] | None, str | None]:
    """Return the run directory, official report, and any scoring cleanup error.

    Pass None to use the CLI's gold prediction. The CLI's summary and instance
    log remain available for runs that did not produce an instance report.
    """
    import docker
    from docker.errors import NotFound

    if version("swebench") != "4.1.0":
        raise RuntimeError("Expected swebench==4.1.0")
    dataset = dataset_snapshot(task)
    instance_id = task.instance_id
    run_id = uuid4().hex
    directory = output_dir.resolve() / run_id
    directory.mkdir(mode=0o700, parents=True)
    model = predictions = "gold"
    if git_patch is not None:
        model, predictions = "candidate", "predictions.jsonl"
        payload = {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": git_patch,
        }
        (directory / predictions).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    launcher = Path(__file__).with_name("_swebench_docker.py").resolve()
    command = [sys.executable, "-I", str(launcher)]
    command += ["--dataset_name", dataset, "--instance_ids", instance_id]
    command += ["--predictions_path", predictions, "--run_id", run_id, "--max_workers", "1"]
    command += ["--instance_image_tag", task.image_tag, "--cache_level", "instance"]
    cli_error = None
    cleanup_error = None
    container_name = f"sweb.eval.{instance_id.lower()}.{run_id}"
    with (directory / "evaluator.log").open("wb") as log:
        try:
            subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            cli_error = exc
        finally:
            # The pinned CLI writes its report before cleanup and may log removal errors.
            try:
                with closing(docker.from_env(timeout=15)) as client:
                    try:
                        container = client.containers.get(container_name)
                    except NotFound:
                        pass
                    else:
                        cleanup_error = f"Official evaluator left container {container_name}"
                        try:
                            container.remove(force=True)
                        except NotFound:
                            cleanup_error += "; already removed"
                        else:
                            cleanup_error += "; removed by trial cleanup"
            except Exception as exc:
                cleanup_error = (
                    f"Could not clean up or verify {container_name} ({type(exc).__name__})"
                )

    if cli_error is not None and cleanup_error is None:
        raise cli_error

    report = directory / "logs" / "run_evaluation" / run_id / model / instance_id / "report.json"
    if not report.is_file():
        return directory, None, cleanup_error
    official = dict(json.loads(report.read_text(encoding="utf-8"))[instance_id])
    return directory, official, cleanup_error
