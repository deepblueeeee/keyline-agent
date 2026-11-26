"""Run a serial batch of Handshake SWE-bench trials."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import traceback
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial
from itertools import product
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import docker
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset

from handshake.config.provider import DeepSeekConfig, read_deepseek_api_key
from handshake.context.budget import ContextBudgetPolicy
from handshake.context.errors import ContextPressureError
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter, ToolDefinition, close_model
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.errors import ModelProviderError
from handshake.models.protocol import ModelContext, ModelTurn
from handshake.runtime.agents import AgentSpec, main_agent_spec
from handshake.runtime.loop import start_run
from handshake.runtime.progress import current_run
from handshake.runtime.state import RunStatus, Session
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools.files import file_tool_bindings
from handshake.tools.registry import ToolRegistry
from handshake.tools.shell import process_tool_bindings

from .swebench import dataset_snapshot, evaluate
from .swebench_workspace import DockerWorkspace
from .tasks import DEFAULT_DATASET, Task, load_tasks, select_tasks
from .tool_observations import visible_context

MAX_OUTPUT_TOKENS = 8192
MODEL_TIMEOUT_SECONDS = 45
MODEL_MAX_RETRIES = 0


@dataclass(frozen=True, slots=True)
class Budget:
    requests: int = 20
    steps: int = 16
    seconds: int = 480


class BudgetedModel:
    def __init__(
        self, model: ModelAdapter, ledger: Path, request_limit: int, *,
        run_id: str | None = None, model_name: str | None = None,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.request_limit = request_limit
        self.run_id = run_id or uuid4().hex
        self.model_name = model_name
        self.requests = 0
        self.responses_with_usage = 0
        self.usage: Counter[str] = Counter()
        self.recorded_tool_definitions = False

    async def complete(self, context: ModelContext, tools: tuple[ToolDefinition, ...]) -> ModelTurn:
        if self.requests >= self.request_limit:
            raise ModelProviderError("Evaluation model request budget exhausted")
        self.requests += 1
        purpose = "compaction" if not context.goal and not tools else "agent"
        record: dict[str, Any] = {
            "run_id": self.run_id, "model": self.model_name,
            "request": self.requests, "purpose": purpose, "usage": None,
        }
        started = monotonic()
        if purpose == "compaction":
            record["compaction"] = {
                "input": json.loads(context.messages[0].content[0]["text"]),
                "output": None,
            }
        try:
            turn = await self.model.complete(context, tools)
            if "compaction" in record and not turn.tool_calls and turn.stop_reason != "max_tokens":
                record["compaction"]["output"] = turn.text
            if (run := current_run()) is not None and context.goal and turn.tool_calls:
                # The eval runner uses a single agent with a fixed tool registry.
                if not self.recorded_tool_definitions:
                    record["tool_definitions"] = [tool.as_anthropic() for tool in tools]
                    self.recorded_tool_definitions = True
                record["tool_decision"] = {
                    "turn": run.model_turns + 1,
                    "calls": [asdict(call) for call in turn.tool_calls],
                    **visible_context(context),
                }
            if turn.usage is not None:
                counts = asdict(turn.usage)
                self.usage.update(counts)
                self.responses_with_usage += 1
                record["usage"] = counts
            return turn
        except (Exception, asyncio.CancelledError) as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            if "compaction" in record:
                record["compaction"]["seconds"] = round(monotonic() - started, 3)
            with self.ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")


def agent_setup(workspace: DockerWorkspace, model_name: str) -> tuple[ToolRegistry, AgentSpec]:
    registry = ToolRegistry()
    for binding in file_tool_bindings():
        registry.register(replace(
            binding, handler=partial(workspace.execute_file, binding.definition.name)
        ))
    for binding in process_tool_bindings(lambda _path, _argv: False):
        registry.register(replace(binding, handler=workspace.execute, prepare=None))
    spec = main_agent_spec(
        model_name,
        {"update_plan", *(definition.name for definition in registry.definitions())},
        ContextBudgetPolicy(reserved_output_tokens=MAX_OUTPUT_TOKENS),
    )
    spec = replace(
        spec,
        instructions=spec.instructions
        + (
            "\nShell and test commands run in Linux at /testbed. The workspace is the same "
            "checkout used by the file tools. Its Python environment is already on PATH."
        ),
    )
    return registry, spec


async def run_agent(
    workspace: DockerWorkspace,
    goal: str,
    model_name: str,
    model: ModelAdapter,
    directory: Path,
    budget: Budget,
) -> dict[str, Any]:
    registry, spec = agent_setup(workspace, model_name)
    store = SQLiteStateStore(directory / "state.db")
    session = Session(workspace.path)
    store.create_session(session)
    run = session.context_scope.new_run(
        goal, sequence=1, model_name=model_name, max_steps=budget.steps, agent_spec_id=spec.spec_id
    )
    observed = BudgetedModel(
        model, directory / "usage.jsonl", budget.requests, run_id=run.run_id, model_name=model_name,
    )
    started = monotonic()
    with (directory / "progress.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            async with asyncio.timeout(budget.seconds):
                await start_run(
                    session,
                    run,
                    observed,
                    registry,
                    store=store,
                    agent_spec=spec,
                    emit=lambda text: print(text, file=log),
                )
        except TimeoutError:
            run.error = f"Agent exceeded its {budget.seconds}s evaluation budget"
            if run.status is RunStatus.RUNNING:
                run.transition(RunStatus.CANCELLED)
                store.checkpoint_run(session, run)
        except (HandshakeError, ContextPressureError) as exc:
            print(str(exc), file=log)

    run = store.load_run(run.run_id)
    calls = Counter(
        block["name"]
        for item in store.read_messages(
            run.context_scope_id, 0, store.message_count(run.context_scope_id)
        )
        for block in item.message.content
        if block["type"] == "tool_use"
    )
    return {
        "run_id": run.run_id,
        "status": run.status,
        "error": run.error,
        "summary": run.final_summary,
        "model": model_name,
        "model_turns": run.model_turns,
        "model_requests": observed.requests,
        "responses_with_usage": observed.responses_with_usage,
        "usage": dict(observed.usage) if observed.responses_with_usage else None,
        "tool_calls": dict(calls),
        "elapsed_seconds": round(monotonic() - started, 3),
    }


def run_trial(task: Task, directory: Path, budget: Budget) -> dict[str, Any]:
    directory.mkdir(mode=0o700, parents=True)
    previous_home = os.environ.get("HANDSHAKE_HOME")
    started = monotonic()
    report: dict[str, Any] = {
        "instance_id": task.instance_id,
        "task": task.as_dict(),
        "budget": asdict(budget),
        "status": "prepare",
        "agent": None,
        "official_scoring_attempted": False,
        "official": None,
        "resolved": None,
    }
    try:
        os.environ["HANDSHAKE_HOME"] = str(directory)
        config = DeepSeekConfig.from_environment()
        if config.base_url.rstrip("/") != DeepSeekConfig().base_url:
            raise ValueError("This live evaluation requires the approved api.deepseek.com endpoint")
        (instance,) = load_swebench_dataset(dataset_snapshot(task), "test", [task.instance_id])
        spec = make_test_spec(instance, namespace="swebench", instance_image_tag=task.image_tag)
        report["base_commit"] = instance["base_commit"]
        report["image"] = spec.instance_image_key
        with (
            closing(docker.from_env(timeout=120)) as client,
            closing(DockerWorkspace(
                client, instance["base_commit"], image=spec.instance_image_key
            )) as workspace,
        ):
            workspace.prepare()
            report["status"] = "agent"
            model = DeepSeekAnthropicModel(
                api_key=read_deepseek_api_key(),
                model=config.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                timeout_seconds=MODEL_TIMEOUT_SECONDS,
                max_retries=MODEL_MAX_RETRIES,
            )

            async def run() -> dict[str, Any]:
                try:
                    return await run_agent(
                        workspace,
                        instance["problem_statement"],
                        config.model,
                        model,
                        directory,
                        budget,
                    )
                finally:
                    await close_model(model)

            report["agent"] = asyncio.run(run())
            report["status"] = "patch_export"
            patch = workspace.export_patch()
            (directory / "candidate.patch").write_bytes(patch.encode("utf-8"))
            if report["agent"]["status"] == RunStatus.PAUSED:
                report["workspace_checkpoint"] = workspace.checkpoint()
            report["status"] = "workspace_cleanup"
        if patch:
            report.update(status="scoring", official_scoring_attempted=True)
            scoring_dir, official, cleanup_error = evaluate(
                task, patch, output_dir=directory / "scoring"
            )
            resolved = None
            if (
                official
                and official.get("patch_successfully_applied")
                and "tests_status" in official
            ):
                resolved = official["resolved"]
            report.update(
                status="scored" if resolved is not None else "scoring_failed",
                scoring_dir=str(scoring_dir),
                resolved=resolved,
                official=official,
            )
            if cleanup_error is not None:
                report.update(status="scoring_cleanup_failed", scoring_cleanup_error=cleanup_error)
        else:
            report["status"] = "empty_patch"
    except Exception as exc:
        (directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        report.update(failed_stage=report["status"], status="error", error=type(exc).__name__)
    finally:
        if previous_home is None:
            os.environ.pop("HANDSHAKE_HOME", None)
        else:
            os.environ["HANDSHAKE_HOME"] = previous_home
    report["elapsed_seconds"] = round(monotonic() - started, 3)
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_batch(
    tasks: tuple[Task, ...], directory: Path, trials: int, budget: Budget
) -> dict[str, Any]:
    directory.mkdir(mode=0o700, parents=True)
    started = monotonic()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    experiment = {
        "experiment_id": directory.name,
        "git_commit": commit.stdout.strip() or None,
        "started_at": datetime.now(UTC).isoformat(),
        "model": {
            "name": DeepSeekConfig.from_environment().model,
            "base_url": DeepSeekConfig().base_url,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "timeout_seconds": MODEL_TIMEOUT_SECONDS,
            "max_retries": MODEL_MAX_RETRIES,
        },
        "tasks": [task.as_dict() for task in tasks],
        "trials_per_task": trials,
        "budget": asdict(budget),
    }
    (directory / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    results: list[dict[str, Any]] = []
    interrupted = False
    try:
        for task, number in product(tasks, range(1, trials + 1)):
            artifact_dir = directory / task.instance_id / f"trial-{number:03d}"
            entry: dict[str, Any] = {
                "instance_id": task.instance_id,
                "suite": task.suite,
                "trial": number,
                "artifact_dir": str(artifact_dir),
                "agent": None,
                "official": None,
                "resolved": None,
            }
            results.append(entry)
            print(f"{task.instance_id} trial {number}/{trials}: {artifact_dir}", flush=True)
            try:
                entry.update(run_trial(task, artifact_dir, budget))
            except Exception as exc:
                entry.update(status="error", error=type(exc).__name__)
            except KeyboardInterrupt:
                entry["status"] = "interrupted"
                raise
    except KeyboardInterrupt:
        interrupted = True

    counts = Counter(resolved=0, unresolved=0, unscored=0)
    usage: Counter[str] = Counter()
    outcomes = {True: "resolved", False: "unresolved", None: "unscored"}
    for entry in results:
        counts[outcomes[entry["resolved"]]] += 1
        usage.update((entry.get("agent") or {}).get("usage") or {})
    summary = {
        "task_count": len(tasks),
        "planned_trials": len(tasks) * trials,
        "recorded_trials": len(results),
        "trials_per_task": trials,
        "interrupted": interrupted,
        "counts": dict(counts),
        "tasks": [task.as_dict() for task in tasks],
        "budget": asdict(budget),
        "trials": results,
        "usage": dict(usage) if usage else None,
        "elapsed_seconds": round(monotonic() - started, 3),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--instance-id", action="append", help="Select a task; repeat for multiple"
    )
    selection.add_argument("--suite", choices=("regression", "capability"))
    parser.add_argument("--trials", type=int, default=1)
    defaults = Budget()
    parser.add_argument("--max-requests", type=int, default=defaults.requests)
    parser.add_argument("--max-steps", type=int, default=defaults.steps)
    parser.add_argument("--agent-seconds", type=int, default=defaults.seconds)
    parser.add_argument(
        "--live", action="store_true", required=True, help="Authorize real API requests"
    )
    args = parser.parse_args()
    budget = Budget(args.max_requests, args.max_steps, args.agent_seconds)
    if min(args.trials, budget.requests, budget.steps, budget.seconds) < 1:
        parser.error("Trials and execution budget limits must be positive")
    try:
        tasks = select_tasks(
            load_tasks(args.dataset), instance_ids=args.instance_id, suite=args.suite
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    directory = args.output_dir.resolve() / uuid4().hex
    summary = run_batch(tasks, directory, args.trials, budget)
    print(f"Summary: {directory / 'summary.json'}")
    if summary["interrupted"]:
        return 130
    return (
        2
        if any(entry["status"] not in {"scored", "empty_patch"} for entry in summary["trials"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
