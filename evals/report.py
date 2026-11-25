"""Generate and compare reports from saved evaluation artifacts."""

import argparse
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from handshake.models.protocol import ModelUsage

from .compaction import RECALL, read_records
from .cost import PRICE_DATE, estimate_deepseek_cost

L1 = "L1 — Component Evaluation"
L2 = "L2 — Regression Evaluation"
L3 = "L3 — Capability Evaluation"
EFFICIENCY = "Cost & Efficiency"
TOKENS = tuple(ModelUsage.__dataclass_fields__)
Row = dict[str, Any]


def read(path: Path) -> Row:
    return json.loads(path.read_text(encoding="utf-8"))


def total(values: list[Any]) -> Any:
    return sum(values) if values and None not in values else None


def account(rows: list[Row]) -> Row:
    count = total([r.get("requests") for r in rows])
    measured = total([r.get("measured") for r in rows])
    costs = [r.get("cost") for r in rows]
    known_costs = [cost for cost in costs if cost is not None]
    tokens: Counter[str] = Counter()
    for row in rows:
        tokens.update(row.get("usage") or {})
    complete = count is not None and count == measured
    input_count = sum(tokens[k] for k in TOKENS if k != "output_tokens")
    return {
        **{key: tokens[key] if complete else None for key in TOKENS},
        "requests": count,
        "with_usage": measured,
        "cache_hit_rate": tokens["cache_read_tokens"] / input_count if complete and input_count
        else None,
        "cost_usd": float(sum(costs)) if complete and None not in costs else None,
        "known_cost_usd": float(sum(known_costs)) if known_costs else None,
        "request_seconds": total([r.get("seconds") for r in rows]),
        "priced": sum(r["measured"] or 0 for r in rows if r.get("cost") is not None),
        "timed": sum(r["requests"] or 0 for r in rows if r.get("seconds") is not None),
    }


def trial_rows(batch: Row, path: Path | None) -> dict[str, Row]:
    tasks = {t["instance_id"]: t for t in batch.get("tasks", [])}
    rows = batch.get("trials", [batch] if "instance_id" in batch else [])
    numbers = range(1, batch.get("trials_per_task", 0) + 1)
    expected = {(key, number): {} for key in tasks for number in numbers}
    expected.update({(r["instance_id"], r.get("trial", 1)): r for r in rows})
    trials = {}
    for (instance, number), row in sorted(expected.items()):
        task = row.get("task") or tasks.get(instance, {})
        agent, official = row.get("agent") or {}, row.get("official") or {}
        directory = row.get("artifact_dir") or (str(path.parent) if row is batch and path else None)
        benchmark = [task.get(k) for k in ("source", "revision", "image_digest")]
        settings = {
            "model": agent.get("model"), "budget": row.get("budget") or batch.get("budget"),
            "benchmark": benchmark if all(benchmark) else None,
            "suite": row.get("suite") or task.get("suite"),
        }
        scored = official.get("patch_successfully_applied") is True and "tests_status" in official
        trials[f"{instance}#{number}"] = {
            "scope": settings, "resolved": official.get("resolved") if scored else None,
            "status": row.get("status"), "agent_status": agent.get("status"), "artifact": directory,
            "agent": agent, "elapsed_seconds": row.get("elapsed_seconds"),
        }
    return trials


def build_report(
    batch_path: Path | None, tool_paths: tuple[Path, ...] = (),
    compaction_paths: tuple[Path, ...] = (), *, peak: bool = True,
) -> Row:
    batch = read(batch_path) if batch_path else {}
    sections: dict[str, Row] = {k: {} for k in (L1, L2, L3, EFFICIENCY)}
    trials = trial_rows(batch, batch_path)
    stages: dict[str, list[Row]] = defaultdict(list)
    runs, compact_scope, judge_scope = {}, {}, {}
    agents = []

    def record_usage(stage: str, row: Row, model: str | None) -> None:
        count = 1 if "purpose" in row else row.get("model_requests", row.get("requests"))
        raw = row.get("usage")
        usage = ModelUsage(**raw) if raw is not None else None
        stages[stage].append({
            "requests": count, "usage": raw,
            "measured": row.get("responses_with_usage", 1 if count == 1 else None)
            if usage else 0 if count is not None else None,
            "cost": 0 if count == 0 else estimate_deepseek_cost(model or "", usage, peak=peak),
            "seconds": 0 if count == 0 else row.get("compaction", row).get("seconds"),
        })

    def add(section: str, name: str, value: Any, scope: Row | None = None,
            scored: int | None = None, count: int | None = None, **details: Any) -> None:
        if name in sections[section]:
            raise ValueError(f"Multiple reports supplied for {name}")
        sections[section][name] = {
            "value": value, "scope": scope or {}, "scored": scored, "total": count,
            "coverage": scored / count if scored is not None and count else None, **details,
        }

    for key, row in trials.items():
        agent = row.pop("agent")
        run_id = agent.get("run_id")
        if run_id in runs:
            continue
        agents.append(agent)
        path = (Path(row["artifact"]) / "usage.jsonl").resolve() if row["artifact"] else None
        captures = read_records(path) if path and path.is_file() else []
        if run_id:
            runs[run_id] = key, captures
        tagged = len(captures) == agent.get("model_requests") and all(
            r.get("purpose") in ("agent", "compaction") for r in captures
        )
        if tagged and captures:
            for capture in captures:
                record_usage(capture["purpose"].capitalize(), capture, agent.get("model"))
        else:
            record_usage("Unsplit" if not tagged else "Agent", agent, agent.get("model"))

    task_scope = {key: row["scope"] for key, row in trials.items()}
    for suite, section in (("regression", L2), ("capability", L3)):
        roster = {key: scope for key, scope in task_scope.items() if scope["suite"] == suite}
        scored = [key for key in roster if type(trials[key]["resolved"]) is bool]
        resolved = sum(trials[key]["resolved"] for key in scored)
        rate = resolved / len(scored) if scored else None
        comparable = all(None not in roster[key].values() for key in scored)
        scope = {"tasks": roster if comparable else None, "official_coverage": scored}
        outcomes = {
            "Resolved": resolved,
            "Unresolved": len(scored) - resolved,
            "Unscored": len(roster) - len(scored),
            "Resolved rate": rate,
        }
        for name, value in outcomes.items():
            denominator = len(scored) if name == "Resolved rate" else len(roster)
            add(section, name, value if roster else None, scope, len(scored), len(roster),
                denominator=denominator)

    seen_tools = set()
    for path in tool_paths:
        report = read(path)
        judge, run_id = report["judge"], report["trial_id"]
        if run_id in seen_tools:
            continue
        seen_tools.add(run_id)
        record_usage("Judge", judge, judge.get("model"))
        label = runs.get(run_id, (run_id, []))[0]
        scope = {"trial": task_scope.get(label, {"same_run": run_id})}
        config = [judge.get("model"), judge.get("rubric_digest"), report.get("context_budget")]
        config = config if all(config) else None
        judge_scope[f"Tool Calling [{label}]"] = {**scope, "judge_config": config}
        params, repeat = report["parameter_accuracy"], report["redundancy"]
        for name, dimension, stats in (
            ("Tool Selection", "selection", report["tool_selection"]),
            ("Parameter Accuracy", "parameters", params["semantic"]),
            ("Redundancy", "redundant", {
                "mean": repeat["rate_among_judged_candidates"],
                "scored": repeat["judged_candidates"],
                "total": len(repeat["candidate_ids"]),
            }),
        ):
            scored, count = stats["scored"], stats["total"]
            ids = [c["id"] for c in report["calls"] if dimension in c.get("scores", {})]
            coverage = [count, "all" if len(ids) == count else [run_id, ids]]
            basis = {**scope, "judge_config": config, "scoring_coverage": coverage}
            if dimension == "redundant":
                basis["strategy"] = repeat["strategy"]
                if scored != count:
                    basis["candidates"] = repeat["candidate_ids"]
            add(L1, f"{name} [{label}]", stats["mean"], basis, scored, count)
        add(L1, f"Parameter Schema [{label}]", params["schema_only"] or None)
        run = report["execution_failure_rate"]
        complete = run["completed"] == run["total_calls"]
        finished = [c["id"] for c in report["calls"] if c["status"] in ("succeeded", "failed")]
        coverage = [run["total_calls"], "all" if complete else [run_id, finished]]
        basis = {**scope, "execution_coverage": coverage, "source": run.get("source")}
        add(L1, f"Execution Failure Rate [{label}]", run["rate"], basis,
            run["completed"], run["total_calls"], denominator=run["completed"],
            failures=run["failed"])

    seen_compactions = set()
    sample_counts: Counter[str] = Counter()
    for path in compaction_paths:
        report = read(path)
        run_id = report.get("run_id")
        if run_id is None and not report["samples"]:
            continue
        if run_id is None:
            raise ValueError("Compaction report lacks a run ID. Regenerate it from a new capture.")
        if run_id in seen_compactions:
            continue
        seen_compactions.add(run_id)
        linked, captures = runs.get(run_id, (None, []))
        model = report.get("compaction_model") or task_scope.get(linked, {}).get("model")
        ledger = Path(report["artifact"])
        if not captures and ledger.is_file():
            captures = read_records(ledger)
        inputs = {r["request"]: r["compaction"]["input"] for r in captures if "compaction" in r}
        if linked is None:
            if captures:
                for capture in captures:
                    record_usage(capture["purpose"].capitalize(), capture, model)
            else:
                record_usage("Agent", report.get("agent") or {}, model)
                for sample in report["samples"]:
                    record_usage("Compaction", sample["compaction"], model)
        for sample in report["samples"]:
            request = sample["request"]
            name = sample.get("sample") or linked or run_id
            label = f"{name}#{request}"
            sample_counts[label] += 1
            if sample_counts[label] > 1:
                label += f" ({sample_counts[label]})"
            record_usage("Judge", sample["judge"], report.get("judge_model"))
            source = json.dumps(inputs[request], sort_keys=True) if request in inputs else None
            basis = {"input": sha256(source.encode()).hexdigest() if source else None,
                     "model": model, "covered_ranges": sample.get("source_ranges")}
            recall = sample.get("recall") or {}
            fact = sample.get("factual_consistency") or {}
            ratio = sample.get("compression") or {}
            config = [report.get("judge_model"), report.get("rubric_digest")]
            config = config if all(config) else None
            facts = [
                [f["id"], sha256(f["fact"].encode()).hexdigest(), f.get("method", "unscored"),
                 f.get("status") in RECALL]
                for f in recall.get("items", [])
                if f.get("status") != "retained_outside_summary"
            ]
            judged = {**basis, "judge_config": config}
            if sample["judge"]["requests"]:
                judge_scope[label] = judged
            recall_scope = judged if any(f[2] == "judge" for f in facts) else basis
            add(L1, f"Key Information Recall [{label}]", recall.get("mean"),
                {**recall_scope, "scoring_coverage": facts},
                recall.get("scored"), recall.get("eligible"))
            add(L1, f"Factual Consistency [{label}]", fact.get("score"), judged, int(bool(fact)), 1)
            token_scope = {**basis, "scope": ratio.get("scope")}
            token_scope["counter"] = ratio.get("token_counter")
            add(L1, f"Compression Ratio [{label}]", ratio.get("ratio"), token_scope,
                int(bool(ratio)), 1, original_tokens=ratio.get("original_tokens"),
                compacted_tokens=ratio.get("compacted_tokens"))
            if not linked:
                compact_scope[label] = basis

    run_scope = {"trials": task_scope, "compactions": compact_scope}
    pricing = {"period": "peak" if peak else "off-peak", "date": PRICE_DATE}
    generation = stages["Agent"] + stages["Compaction"] + stages["Unsplit"]
    stages["Agent run"] = generation
    stages["Overall"] = generation + stages["Judge"]
    zero = {"requests": 0, "measured": 0, "cost": 0, "seconds": 0}
    for stage in ("Agent", "Compaction", "Judge", "Agent run", "Overall"):
        rows = stages[stage]
        if stage in ("Agent", "Compaction") and generation:
            rows = rows + [{}] if stages["Unsplit"] else rows or [zero]
        scope = run_scope if stage != "Judge" else {"judge": judge_scope}
        if stage == "Overall" and judge_scope:
            scope = {**scope, "judge": judge_scope}
        stats = account(rows)
        requests, measured = stats["requests"], stats.pop("with_usage")
        coverage = {
            "requests": requests, "request_seconds": stats.pop("timed"),
            "cost_usd": stats["priced"], "known_cost_usd": stats.pop("priced"),
        }
        for name, value in stats.items():
            if name == "known_cost_usd" and value == stats["cost_usd"]:
                continue
            basis = {**scope, "pricing": pricing} if name.endswith("cost_usd") else scope
            used = coverage.get(name, measured)
            add(EFFICIENCY, f"{stage} · {name}", value, basis, used, requests)
    tool_calls = [sum(a["tool_calls"].values()) if "tool_calls" in a else None for a in agents]
    agent_seconds = total([a.get("elapsed_seconds") for a in agents])
    for name, value in {
        "Tool calls": total(tool_calls),
        "Agent loop wall seconds (sum, includes compaction)": agent_seconds,
        "End-to-end wall seconds": batch.get("elapsed_seconds"),
    }.items():
        add(EFFICIENCY, name, value, {"trials": task_scope})
    inputs = (batch_path, *tool_paths, *compaction_paths)
    return {
        "sections": sections, "trials": trials, "pricing": pricing,
        "artifacts": [str(p.resolve()) for p in inputs if p],
    }


def compare(baseline: Row, current: Row) -> Row:
    result: Row = {"sections": {}, "trials": {}, "regressions": []}
    left_sections = {**baseline["sections"], "Trials": baseline["trials"]}
    right_sections = {**current["sections"], "Trials": current["trials"]}
    for section, left in left_sections.items():
        right = right_sections[section]
        is_trial = section == "Trials"
        changes = result["trials"] if is_trial else result["sections"].setdefault(section, {})
        for key in sorted(left.keys() | right.keys()):
            a, b = left.get(key, {}), right.get(key, {})
            field = "resolved" if is_trial else "value"
            before, after = a.get(field), b.get(field)
            x, y = a.get("scope", {}), b.get("scope", {})
            reasons = [f"{k} differs" for k in sorted(x.keys() | y.keys()) if x.get(k) != y.get(k)]
            reasons += [
                f"{k} unavailable" for k in sorted(x.keys() & y.keys())
                if x[k] is None or y[k] is None
            ]
            types = (bool,) if is_trial else (int, float)
            if not x or not y or type(before) not in types or type(after) not in types:
                reasons.append("comparison data unavailable")
            if key.endswith("known_cost_usd"):
                reasons.append("partial cost only")
            change = changes[key] = {"baseline": before, "current": after, "differences": reasons}
            if not is_trial:
                change["delta"] = round(after - before, 10) if not reasons else None
                change["coverage"] = {
                    label: {k: original.get(k) for k in ("scored", "total", "coverage")}
                    for label, original in (("baseline", a), ("current", b))
                }
            elif not reasons and x["suite"] == "regression" and before is True and after is False:
                result["regressions"].append(key)
    return result


def markdown(report: Row) -> str:
    def cell(value: Any) -> str:
        if type(value) is bool:
            return "resolved" if value else "unresolved"
        if isinstance(value, float):
            return f"{value:.8g}"
        return "N/A" if value is None else str(value).replace("|", "\\|").replace("\n", " ")

    comparing = "regressions" in report
    lines = ["# Baseline comparison" if comparing else "# Evaluation report"]
    if "pricing" in report:
        price = report["pricing"]
        lines.extend(["", f"Estimated API cost in USD · {price['period']} · {price['date']}"])
    for section, rows in {**report["sections"], "Trials": report["trials"]}.items():
        lines.extend(["", f"## {section}", ""])
        if not rows:
            lines.append("N/A")
            continue
        if comparing:
            headers = ("Metric", "Baseline", "Current", "Δ", "Comparability")
            values = ([k, r["baseline"], r["current"], r.get("delta"), ", ".join(r["differences"])]
                      for k, r in rows.items())
        elif section == "Trials":
            headers = ("Trial", "Suite", "Official result", "Runtime status")
            values = (
                [k, r["scope"]["suite"], r["resolved"], r["agent_status"]] for k, r in rows.items()
            )
        else:
            headers = ("Metric", "Result", "Scored", "Total", "Denominator")
            values = (
                [k, r["value"], r["scored"], r["total"], r.get("denominator", r["total"])]
                for k, r in rows.items()
            )
        for row in (headers, ("---",) * len(headers), *values):
            lines.append("| " + " | ".join(map(cell, row)) + " |")
    if comparing:
        losses = ", ".join(report["regressions"]) or "None identified."
        lines.extend(["", "## Regression losses", "", losses])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--batch", type=Path)
    for option in ("--tool-calling", "--compaction"):
        generate.add_argument(option, type=Path, action="append", default=[])
    generate.add_argument("--off-peak", action="store_true")
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("current", type=Path)
    for command in (generate, comparison):
        command.add_argument("--output", type=Path, required=True, help="JSON and Markdown prefix")
    args = parser.parse_args()
    if args.command == "generate":
        if not (args.batch or args.tool_calling or args.compaction):
            parser.error("Generate requires at least one saved artifact")
        report = build_report(
            args.batch, tuple(args.tool_calling), tuple(args.compaction), peak=not args.off_peak
        )
    else:
        report = compare(read(args.baseline), read(args.current))
    json_report = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.with_suffix(".json").write_text(json_report, encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
