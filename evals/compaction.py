"""Generate or rescore semantic compaction artifacts without replaying an Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from keyline_agent.config.provider import DeepSeekConfig, read_deepseek_api_key
from keyline_agent.context.budget import Utf8TokenEstimator, compact_json
from keyline_agent.context.condenser import LLMHistoryCondenser
from keyline_agent.context.reduction import same_tool_round
from keyline_agent.context.state import SequencedMessage
from keyline_agent.models.base import ModelAdapter, close_model
from keyline_agent.models.deepseek_anthropic import DeepSeekAnthropicModel
from keyline_agent.models.errors import ModelProtocolError, ModelProviderError
from keyline_agent.models.protocol import ModelMessage

from .judge import check_judge_input, judge_json

SAMPLES = Path(__file__).with_name("compaction_samples.json")
RUBRIC = Path(__file__).with_name("compaction_prompt.txt")
RECALL = {"preserved": 1, "partial": 0.5, "missing": 0, "incorrect": 0}


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def generate(
    sample: dict[str, Any], model: ModelAdapter, artifact: Path, *, model_name: str | None = None,
) -> dict[str, Any]:
    from .swebench_run import BudgetedModel

    history = tuple(
        SequencedMessage(i, ModelMessage(message["role"], tuple(message["content"])))
        for i, message in enumerate(sample["messages"])
    )
    artifact.touch(mode=0o600, exist_ok=False)
    condenser = LLMHistoryCondenser(BudgetedModel(model, artifact, 1, model_name=model_name))
    with suppress(ModelProtocolError, ModelProviderError):
        # The original request, result, and usage remain available in the ledger.
        await condenser.compact_raw((), history[: sample["target_count"]])
    (record,) = read_records(artifact)
    record.update(
        sample=sample["id"],
        facts=sample["facts"],
        protected=sample["messages"][sample["target_count"] :],
    )
    artifact.write_text(compact_json(record) + "\n", encoding="utf-8")
    return record


def covered_input(source: dict[str, Any], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the claimed span before comparing the replaced input and output."""
    target = source.get("target")
    if target is None:
        original = source["episodes"]
        bounds = [
            original[0]["source_range"]["start_sequence"],
            original[-1]["source_range"]["end_sequence"],
        ]
        if (
            len(episodes) != 1
            or [episodes[0][f"{edge}_sequence"] for edge in ("start", "end")] != bounds
        ):
            raise ValueError("Rollup output must cover its entire input")
        return original
    end = target[0]["sequence"] - 1
    for episode in episodes:
        start, stop = episode["start_sequence"], episode["end_sequence"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not end + 1 == start <= stop <= target[-1]["sequence"]
        ):
            raise ValueError("Compaction output does not cover a contiguous target prefix")
        end = stop
    replaced = [row for row in target if row["sequence"] <= end]
    endpoints = {episode["end_sequence"] for episode in episodes}
    for previous, following in zip(replaced, target[1:], strict=False):
        if previous["sequence"] in endpoints:
            pair = [
                SequencedMessage(
                    row["sequence"],
                    ModelMessage(row["message"]["role"], tuple(row["message"]["content"])),
                )
                for row in (previous, following)
            ]
            if same_tool_round(*pair):
                raise ValueError("Compaction output splits a tool round")
    return replaced


def _tokens(value: Any) -> int:
    return Utf8TokenEstimator().estimate_text(compact_json(value)) if value else 0


async def score_record(
    record: dict[str, Any], model: ModelAdapter | None, key: str = ""
) -> dict[str, Any]:
    event = record["compaction"]
    result: dict[str, Any] = {
        "request": record["request"],
        "sample": record.get("sample"),
        "status": "unscored",
        "compaction": {"requests": 1, "usage": record["usage"], "seconds": event["seconds"]},
        "judge": {"requests": 0, "usage": None},
        "recall": None,
        "factual_consistency": None,
        "compression": None,
    }
    try:
        episodes = json.loads(event["output"])["episodes"]
        if not isinstance(episodes, list):
            raise ValueError("Expected an array of episodes")
        if episodes == []:
            return {**result, "status": "declined"}
        for row in episodes:
            if (
                not isinstance(row["title"], str)
                or not isinstance(row["summary"], str)
                or not row["title"].strip()
                or not row["summary"].strip()
            ):
                raise ValueError("Invalid compaction episode")
        original = covered_input(event["input"], episodes)
    except (KeyError, IndexError, TypeError, ValueError):
        return {**result, "status": "invalid_or_failed", "error": record.get("error")}

    before, after = _tokens(original), _tokens(episodes)
    remaining = event["input"].get("target", [])[len(original) :]
    result["compression"] = {
        "scope": "replaced payload only",
        "token_counter": "ceil(utf8_bytes/3)",
        "original_tokens": before,
        "compacted_tokens": after,
        "ratio": before / after,
        "reduction_rate": 1 - after / before,
        "uncompacted_target_tokens": _tokens(remaining),
        "protected_tokens": _tokens(record["protected"]) if "protected" in record else None,
        "complete_context_tokens": None,
    }
    result["source_ranges"] = [[row["start_sequence"], row["end_sequence"]] for row in episodes]
    end = episodes[-1]["end_sequence"]
    facts = [dict(fact, status=None) for fact in record.get("facts", [])]
    summary = "\n".join(f"{row['title']}\n{row['summary']}" for row in episodes)
    pending = []
    for fact in facts:
        if fact["sequence"] > end:
            fact["status"] = "retained_outside_summary"
        elif literal := fact.get("literal"):
            fact.update(status="preserved" if literal in summary else "missing", method="literal")
        else:
            pending.append(fact)

    if model is not None:
        payload = {"original": original, "compacted": episodes, "key_items": pending}
        if "target" in event["input"] and event["input"]["episodes"]:
            payload["background"] = event["input"]["episodes"]
        check_judge_input(json.dumps(payload, ensure_ascii=False), key)
        response, usage, error = await judge_json(model, RUBRIC.read_text(), payload)
        response = response or {}
        result["judge"] = {"requests": 1, "usage": usage, "error": error}
        grades = response.get("recall")
        for fact in pending:
            grade = grades.get(fact["id"]) if isinstance(grades, dict) else None
            if (
                isinstance(grade, dict)
                and grade.get("status") in list(RECALL)
                and isinstance(grade.get("evidence"), str)
                and grade["evidence"].strip()
            ):
                fact.update(status=grade["status"], evidence=grade["evidence"], method="judge")
        consistency = response.get("consistency")
        if (
            isinstance(consistency, dict)
            and type(consistency.get("score")) is int
            and 0 <= consistency["score"] <= 2
            and isinstance(consistency.get("reason"), str)
            and consistency["reason"].strip()
            and isinstance(consistency.get("errors"), list)
            and all(isinstance(item, str) and item.strip() for item in consistency["errors"])
            and (consistency["score"] == 2) == (not consistency["errors"])
        ):
            result["factual_consistency"] = consistency
    eligible = [fact for fact in facts if fact["status"] != "retained_outside_summary"]
    scored = [RECALL[fact["status"]] for fact in eligible if fact["status"] in RECALL]
    result["recall"] = {
        "items": facts,
        "eligible": len(eligible),
        "scored": len(scored),
        "coverage": len(scored) / len(eligible) if eligible else None,
        "mean": mean(scored) if scored else None,
        "complete_rate_among_scored": scored.count(1) / len(scored) if scored else None,
        "incorrect_ids": [fact["id"] for fact in eligible if fact["status"] == "incorrect"],
    }
    if result["factual_consistency"] is not None and len(scored) == len(eligible):
        result["status"] = "scored"
    elif scored or result["factual_consistency"] is not None:
        result["status"] = "partially_scored"
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    generating = args.sample is not None
    key = read_deepseek_api_key() if generating else ""

    def connect(name: str) -> DeepSeekAnthropicModel:
        return DeepSeekAnthropicModel(
            api_key=key, model=name, max_tokens=2048, max_retries=0, timeout_seconds=60
        )

    if generating:
        samples = json.loads(SAMPLES.read_text())
        sample = next(row for row in samples if row["id"] == args.sample)
        check_judge_input(json.dumps(sample, ensure_ascii=False), key)
        artifact = args.output.with_suffix(".capture.jsonl")
        compactor = connect(args.compaction_model)
        try:
            records = [
                await generate(sample, compactor, artifact, model_name=args.compaction_model)
            ]
        finally:
            await close_model(compactor)
    else:
        artifact = args.saved or args.trial / "usage.jsonl"
        records = read_records(artifact) if artifact.is_file() or args.saved else []

    captures = [record for record in records if "compaction" in record]
    agent = [record for record in records if record.get("purpose") == "agent"]
    usage: Counter[str] = Counter()
    for record in agent:
        usage.update(record["usage"] or {})
    if captures and args.judge_model and not key:
        key = read_deepseek_api_key()
    judge = connect(args.judge_model) if captures and args.judge_model else None
    try:
        scored = [await score_record(record, judge, key) for record in captures]
    finally:
        if judge:
            await close_model(judge)
    return {
        "artifact": str(artifact.resolve()),
        "run_id": records[0].get("run_id") if records else None,
        "rubric": RUBRIC.name,
        "rubric_digest": sha256(RUBRIC.read_bytes()).hexdigest(),
        "status": "evaluated" if captures else "not_applicable",
        "samples": scored,
        "agent": {"requests": len(agent), "usage": dict(usage) or None}
        if generating or agent or captures
        else None,
        "compaction_model": records[0].get("model") if records else None,
        "judge_model": args.judge_model,
        "scope": "condenser inputs and proposals; runtime acceptance is not measured",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", choices=[row["id"] for row in json.loads(SAMPLES.read_text())])
    source.add_argument("--saved", type=Path)
    source.add_argument("--trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compaction-model", default=DeepSeekConfig().model)
    parser.add_argument("--judge-model", help="Score reviewed public or synthetic artifacts")
    parser.add_argument(
        "--live", action="store_true", help="Authorize model requests with the selected data"
    )
    args = parser.parse_args()
    if (args.sample or args.judge_model) and not args.live:
        parser.error("Model requests require --live")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Output already exists: {args.output}. Choose a new --output path.")
    result = asyncio.run(run(args))
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
