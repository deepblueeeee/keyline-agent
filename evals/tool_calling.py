"""Evaluate tool decisions recorded by the SWE-bench runner."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from jsonschema import Draft202012Validator

from handshake.config.provider import DeepSeekConfig, read_deepseek_api_key
from handshake.models.base import ModelAdapter, close_model
from handshake.models.deepseek_anthropic import DeepSeekAnthropicModel
from handshake.storage.sqlite import SQLiteStateStore

from .judge import check_judge_input, judge_json
from .tool_observations import EXCERPT_CHARS, VISIBLE_BLOCKS

RUBRIC = Path(__file__).with_name("tool_calling_prompt.txt")
BATCH_SIZE = 6
MAX_JUDGE_OUTPUT_TOKENS = 8192


def prepare(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    database = directory / "state.db"
    if not database.is_file():
        raise ValueError("Trial has no state.db")
    (run,) = SQLiteStateStore(database).list_runs()
    executions = {(tool.model_turn, tool.tool_call_id): tool for tool in run.tool_runs}
    definitions: list[dict[str, Any]] = []
    validators: dict[str, Draft202012Validator] = {}
    batches, calls, items = [], [], []
    prior_calls: dict[tuple[str, str], int] = {}
    seen_ids: Counter[str] = Counter()
    for line in (directory / "usage.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "tool_definitions" in record:
            definitions = record["tool_definitions"]
            validators = {
                tool["name"]: Draft202012Validator(tool["input_schema"]) for tool in definitions
            }
        if not (decision := record.get("tool_decision")):
            continue
        turn = decision["turn"]
        start = len(items)
        visible_results = {
            block["tool_use_id"] for block in decision["visible"] if block["type"] == "tool_result"
        }
        for choice in decision["calls"]:
            call_id, name, arguments = choice["id"], choice["name"], choice["arguments"]
            row = {
                "id": f"{turn}:{call_id}",
                "call_id": call_id,
                "turn": turn,
                "request": record["request"],
                "tool": name,
            }
            assessment = {**row, "arguments": arguments}
            tool_run = executions.get((turn, call_id))
            schema = validators.get(name)
            errors = [e.message for e in schema.iter_errors(arguments)] if schema else []
            row.update(
                status=tool_run.status if tool_run else "unrecorded",
                exit_code=tool_run.result.exit_code if tool_run and tool_run.result else None,
                parameter_legality="invalid" if errors else "valid" if schema else "unavailable",
                schema_errors=errors,
            )

            key = name, json.dumps(arguments, sort_keys=True)
            if (index := prior_calls.get(key)) is not None:
                prior = calls[index]
                same_turn = prior["turn"] == turn
                visible = seen_ids[prior["call_id"]] == 1 and prior["call_id"] in visible_results
                row["repeat_of"] = prior["id"]
                assessment["comparison"] = {
                    "previous": prior["id"],
                    "same_turn": same_turn,
                    "earlier_result_visible": False if same_turn else True if visible else None,
                    "intervening": [
                        {k: item[k] for k in ("id", "tool", "turn", "arguments")}
                        for item in items[index + 1 :]
                    ],
                }
            prior_calls[key] = len(calls)
            seen_ids[call_id] += 1
            calls.append(row)
            items.append(assessment)
        # Batch within one turn so a later turn's context cannot reveal earlier outcomes.
        for offset in range(start, len(items), BATCH_SIZE):
            batches.append({
                "tools": definitions,
                "context": {k: decision[k] for k in ("system_prompt", "visible", "omitted_blocks")},
                "calls": items[offset : offset + BATCH_SIZE],
            })

    if not calls and executions:
        raise ValueError("Trial lacks tool decision evidence. Rerun it with the current runner.")
    return calls, {"trial_id": run.run_id, "batches": batches}


def _valid_grade(grade: Any, dimension: str) -> bool:
    if not isinstance(grade, dict) or not isinstance(grade.get("reason"), str):
        return False
    evidence = grade.get("evidence")
    value = grade.get("value" if dimension == "redundant" else "score")
    if dimension == "redundant":
        valid = type(value) is bool
    else:
        valid = type(value) is int and 1 <= value <= 5
    return bool(
        valid and grade["reason"].strip() and isinstance(evidence, list) and evidence
        and all(isinstance(item, str) and item.strip() for item in evidence)
    )


async def score_batches(model: ModelAdapter, payload: dict[str, Any]) -> dict[str, Any]:
    grades, errors = {}, []
    usage: Counter[str] = Counter()
    requests = with_usage = 0
    rubric = RUBRIC.read_text(encoding="utf-8")
    for batch in payload["batches"]:
        expected = {call["id"]: call for call in batch["calls"]}
        requests += 1
        data, counts, error = await judge_json(model, rubric, batch)
        if counts is not None:
            usage.update(counts)
            with_usage += 1
        rows = data.get("grades") if data else None
        if not isinstance(rows, list):
            errors.append({"ids": list(expected), "error": error or "missing_grades"})
            continue
        ids = Counter(row.get("id") for row in rows
                      if isinstance(row, dict) and isinstance(row.get("id"), str))
        for row in rows:
            call_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(call_id, str) or call_id not in expected or ids[call_id] != 1:
                errors.append({"ids": list(expected), "error": "unexpected_or_duplicate_id"})
                continue
            dimensions = ["selection", "parameters"]
            if "comparison" in expected[call_id]:
                dimensions.append("redundant")
            for dimension in dimensions:
                if (grade := row.get(dimension)) is None:
                    continue
                if _valid_grade(grade, dimension):
                    grades.setdefault(call_id, {})[dimension] = grade
                else:
                    errors.append({"ids": [call_id], "error": f"invalid_{dimension}"})
    return {
        "grades": grades,
        "requests": requests,
        "responses_with_usage": with_usage,
        "usage": dict(usage) if with_usage else None,
        "errors": errors,
    }


def summarize(calls: list[dict[str, Any]], trial_id: str, judge: dict[str, Any]) -> dict[str, Any]:
    grades = judge["grades"]
    total = len(calls)

    def dimension(name: str) -> dict[str, Any]:
        scores = [grade[name]["score"] for grade in grades.values() if name in grade]
        return {
            "mean": mean(scores) if scores else None,
            "scored": len(scores),
            "total": total,
            "coverage": len(scores) / total if total else None,
        }

    candidates = [call["id"] for call in calls if "repeat_of" in call]
    compared = [grades[i]["redundant"] for i in candidates if "redundant" in grades.get(i, {})]
    redundant = sum(grade["value"] for grade in compared)
    execution = Counter(call["status"] for call in calls)
    completed = execution["succeeded"] + execution["failed"]
    return {
        "trial_id": trial_id,
        "status": "evaluated" if total else "not_applicable",
        "judge": {key: value for key, value in judge.items() if key != "grades"},
        "calls": [{**call, "scores": grades.get(call["id"], {})} for call in calls],
        "tool_selection": dimension("selection"),
        "parameter_accuracy": {
            "schema_only": dict(Counter(call["parameter_legality"] for call in calls)),
            "semantic": dimension("parameters"),
        },
        "execution_failure_rate": {
            "source": "persisted_tool_runs",
            "total_calls": total,
            "completed": completed,
            "failed": execution["failed"],
            "rate": execution["failed"] / completed if completed else None,
        },
        "redundancy": {
            "strategy": "nearest prior exact match of tool name and arguments across the trial",
            "total_calls": total,
            "candidate_ids": candidates,
            "judged_candidates": len(compared),
            "candidate_coverage": len(compared) / len(candidates) if candidates else None,
            "redundant": redundant,
            "rate_among_judged_candidates": redundant / len(compared) if compared else None,
        },
        "context_budget": {
            "visible_blocks": VISIBLE_BLOCKS,
            "excerpt_chars": EXCERPT_CHARS,
            "batch_size": BATCH_SIZE,
            "max_output_tokens": MAX_JUDGE_OUTPUT_TOKENS,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--live", action="store_true", required=True, help="Authorize Judge requests"
    )
    args = parser.parse_args()
    output = args.output or args.trial / "tool_calling.json"
    if output.exists() or output.is_symlink():
        parser.error(f"Output already exists: {output}. Choose a new --output path.")
    calls, payload = prepare(args.trial)
    config = DeepSeekConfig.from_environment(model_override=args.judge_model)
    if config.base_url.rstrip("/") != DeepSeekConfig().base_url:
        parser.error("This live evaluation requires the approved api.deepseek.com endpoint")

    async def score() -> dict[str, Any]:
        if not calls:
            return {"grades": {}, "requests": 0, "responses_with_usage": 0, "usage": None,
                    "errors": []}
        key = read_deepseek_api_key()
        check_judge_input(json.dumps(payload), key)
        model = DeepSeekAnthropicModel(
            api_key=key, model=config.model, max_tokens=MAX_JUDGE_OUTPUT_TOKENS,
            max_retries=0, timeout_seconds=60,
        )
        try:
            return await score_batches(model, payload)
        finally:
            await close_model(model)

    judge = asyncio.run(score())
    judge["model"] = config.model if calls else None
    judge["rubric_digest"] = sha256(RUBRIC.read_bytes()).hexdigest()
    result = summarize(calls, payload["trial_id"], judge)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
