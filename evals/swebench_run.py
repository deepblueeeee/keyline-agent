"""Run one live Handshake trial and score its patch with the official evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
from collections import Counter
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import docker
from swebench.harness.utils import load_swebench_dataset

from handshake.config.provider import DeepSeekConfig, read_deepseek_api_key
from handshake.context.budget import ContextBudgetPolicy
from handshake.context.errors import ContextPressureError
from handshake.errors import HandshakeError
from handshake.models.base import ModelAdapter, ToolDefinition, close_model
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.models.errors import ModelProviderError
from handshake.models.protocol import ModelContext, ModelTurn
from handshake.runtime.agents import main_agent_spec
from handshake.runtime.loop import start_run
from handshake.runtime.state import RunStatus, Session
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools.files import file_tool_bindings
from handshake.tools.registry import ToolRegistry
from handshake.tools.shell import process_tool_bindings

from .swebench import INSTANCE_ID, dataset_snapshot, evaluate
from .swebench_workspace import DockerWorkspace

MAX_REQUESTS = 20
MAX_STEPS = 16
AGENT_SECONDS = 480
MAX_OUTPUT_TOKENS = 8192


class BudgetedModel:
    def __init__(self, model: ModelAdapter, ledger: Path) -> None:
        self.model = model
        self.ledger = ledger
        self.requests = 0
        self.responses_with_usage = 0
        self.usage: Counter[str] = Counter()

    async def complete(self, context: ModelContext, tools: tuple[ToolDefinition, ...]) -> ModelTurn:
        if self.requests >= MAX_REQUESTS:
            raise ModelProviderError("Evaluation model request budget exhausted")
        self.requests += 1
        record: dict[str, Any] = {"request": self.requests, "usage": None}
        try:
            turn = await self.model.complete(context, tools)
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
            with self.ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")


async def run_agent(
    workspace: DockerWorkspace, goal: str, model_name: str, model: ModelAdapter, directory: Path
) -> dict[str, Any]:
    registry = ToolRegistry()
    for binding in file_tool_bindings():
        registry.register(binding)
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
    store = SQLiteStateStore(directory / "state.db")
    session = Session(workspace.path)
    store.create_session(session)
    run = session.context_scope.new_run(
        goal, sequence=1, model_name=model_name, max_steps=MAX_STEPS, agent_spec_id=spec.spec_id
    )
    observed = BudgetedModel(model, directory / "usage.jsonl")
    started = monotonic()
    with (directory / "progress.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            async with asyncio.timeout(AGENT_SECONDS):
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
            run.error = f"Agent exceeded its {AGENT_SECONDS}s evaluation budget"
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


def run_trial(directory: Path) -> dict[str, Any]:
    directory.mkdir(mode=0o700, parents=True)
    previous_home = os.environ.get("HANDSHAKE_HOME")
    started = monotonic()
    report: dict[str, Any] = {
        "instance_id": INSTANCE_ID,
        "status": "prepare",
        "official_scoring_attempted": False,
        "official": None,
        "resolved": None,
    }
    try:
        os.environ["HANDSHAKE_HOME"] = str(directory)
        config = DeepSeekConfig.from_environment()
        if config.base_url.rstrip("/") != DeepSeekConfig().base_url:
            raise ValueError("This live evaluation requires the approved api.deepseek.com endpoint")
        (task,) = load_swebench_dataset(dataset_snapshot(), "test", [INSTANCE_ID])
        report["base_commit"] = task["base_commit"]
        with closing(docker.from_env()) as client:
            workspace = DockerWorkspace(client, directory / "workspace", task["base_commit"])
            workspace.prepare()
            report["status"] = "agent"
            model = DeepSeekAnthropicModel(
                api_key=read_deepseek_api_key(),
                model=config.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                timeout_seconds=45,
                max_retries=0,
            )

            async def run() -> dict[str, Any]:
                try:
                    return await run_agent(
                        workspace, task["problem_statement"], config.model, model, directory
                    )
                finally:
                    await close_model(model)

            report["agent"] = asyncio.run(run())
            report["status"] = "patch_export"
            patch = workspace.export_patch()
            (directory / "candidate.patch").write_bytes(patch.encode("utf-8"))
        if patch:
            report.update(status="scoring", official_scoring_attempted=True)
            scoring_dir, official, cleanup_error = evaluate(patch, output_dir=directory / "scoring")
            report.update(
                status="scored" if official is not None else "scoring_failed",
                scoring_dir=str(scoring_dir),
                resolved=official["resolved"] if official is not None else None,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--live", action="store_true", required=True, help="Authorize real API requests"
    )
    args = parser.parse_args()
    directory = args.output_dir.resolve() / uuid4().hex
    print(f"Trial directory: {directory}", flush=True)
    result = run_trial(directory)
    print(f"Outcome: {result['status']}, resolved: {result['resolved']}")
    return 0 if result["status"] in {"scored", "empty_patch"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
