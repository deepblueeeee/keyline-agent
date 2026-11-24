"""Run the official SWE-bench CLI with pinned inputs."""

import json
import subprocess
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

INSTANCE_ID = "sympy__sympy-20590"
DATASET_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"
IMAGE_TAG = "latest@sha256:bcdff110a35d19a6838fde5f678d1b5103610234bd034e5703df46878aad3b63"


def dataset_snapshot() -> str:
    from huggingface_hub import snapshot_download

    return str(
        snapshot_download(
            "princeton-nlp/SWE-bench_Lite",
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=["README.md", "data/*.parquet"],
            token=False,
        )
    )


def evaluate(
    git_patch: str | None, *, output_dir: Path
) -> tuple[Path, dict[str, Any] | None, str | None]:
    """Return the run directory, official report, and any scoring cleanup error.

    Pass None to use the CLI's gold prediction. The CLI's summary and instance
    log remain available for runs that did not produce an instance report.
    """
    import docker
    from docker.errors import NotFound

    if version("swebench") != "4.1.0":
        raise RuntimeError("Expected swebench==4.1.0")
    dataset = dataset_snapshot()
    run_id = uuid4().hex
    directory = output_dir.resolve() / run_id
    directory.mkdir(mode=0o700, parents=True)
    model = predictions = "gold"
    if git_patch is not None:
        model, predictions = "candidate", "predictions.jsonl"
        payload = {
            "instance_id": INSTANCE_ID,
            "model_name_or_path": model,
            "model_patch": git_patch,
        }
        (directory / predictions).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    launcher = Path(__file__).with_name("_swebench_docker.py").resolve()
    command = [sys.executable, "-I", str(launcher)]
    command += ["--dataset_name", dataset, "--instance_ids", INSTANCE_ID]
    command += ["--predictions_path", predictions, "--run_id", run_id, "--max_workers", "1"]
    command += ["--instance_image_tag", IMAGE_TAG, "--cache_level", "instance"]
    cli_error = None
    cleanup_error = None
    container_name = f"sweb.eval.{INSTANCE_ID.lower()}.{run_id}"
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

    report = directory / "logs" / "run_evaluation" / run_id / model / INSTANCE_ID / "report.json"
    if not report.is_file():
        return directory, None, cleanup_error
    official = dict(json.loads(report.read_text(encoding="utf-8"))[INSTANCE_ID])
    return directory, official, cleanup_error
