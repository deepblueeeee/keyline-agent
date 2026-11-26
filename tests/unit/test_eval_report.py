import json
import sys
from dataclasses import asdict
from pathlib import Path
from shutil import copyfile
from typing import Any

import pytest

from evals.cost import estimate_deepseek_cost
from evals.report import EFFICIENCY, L1, L2, L3, build_report, compare, main, markdown
from handshake.models.protocol import ModelUsage

MODEL = "deepseek-flash"
AGENT = ModelUsage(90, 25, cache_read_tokens=10)
COMPACTION = ModelUsage(20, 5)
JUDGE = ModelUsage(30, 8)
COMBINED = ModelUsage(110, 30, cache_read_tokens=10)


def save(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("existing_suffix", [".json", ".md"])
def test_comparison_does_not_clobber_either_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    existing_suffix: str,
) -> None:
    baseline = save(tmp_path / "baseline.json", {"sections": {}, "trials": {}})
    current = copyfile(baseline, tmp_path / "current.json")
    prefix = tmp_path / "comparison"
    (existing := prefix.with_suffix(existing_suffix)).write_text("existing report\n")
    command = ["report", "compare", str(baseline), str(current), "--output"]
    monkeypatch.setattr(sys, "argv", [*command, str(prefix)])

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "already exists" in capsys.readouterr().err
    assert existing.read_text() == "existing report\n"
    assert list(tmp_path.glob("comparison.*")) == [existing]


def fixtures(
    directory: Path,
    *,
    resolved: bool = True,
    steps: int = 12,
    judge_model: str = MODEL,
    judged: bool = True,
    agent_coverage: int = 2,
) -> tuple[Path, Path, Path]:
    directory.mkdir()
    task = {
        "instance_id": "repo__project-1",
        "suite": "regression",
        "source": "official",
        "revision": "abc",
        "image_digest": "sha256:123",
    }
    capability = {**task, "instance_id": "repo__project-2", "suite": "capability"}
    ledger = directory / "usage.jsonl"
    ledger.write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {
                        "request": 1,
                        "run_id": "run",
                        "purpose": "agent",
                        "usage": asdict(AGENT) if agent_coverage == 2 else None,
                    },
                    {
                        "request": 2,
                        "run_id": "run",
                        "purpose": "compaction",
                        "usage": asdict(COMPACTION),
                        "compaction": {"input": {"target": ["Original history"]}, "seconds": 3.5},
                    },
                ],
            )
        )
    )
    batch = save(
        directory / "summary.json",
        {
            "tasks": [task, capability],
            "trials_per_task": 1,
            "budget": {"steps": steps},
            "elapsed_seconds": 50.0,
            "trials": [
                {
                    "instance_id": task["instance_id"],
                    "artifact_dir": str(directory),
                    "status": "scored",
                    "agent": {
                        "status": "failed",
                        "model": MODEL,
                        "model_requests": 2,
                        "responses_with_usage": agent_coverage,
                        "usage": asdict(COMBINED if agent_coverage == 2 else COMPACTION),
                        "elapsed_seconds": 20.0,
                        "tool_calls": {"read_file": 3},
                        "run_id": "run",
                    },
                    "official": {
                        "patch_successfully_applied": True,
                        "tests_status": {},
                        "resolved": resolved,
                    },
                },
                {
                    "instance_id": capability["instance_id"],
                    "status": "scoring_failed",
                    "resolved": False,
                    "official": {"patch_successfully_applied": False, "resolved": False},
                },
            ],
        },
    )
    compaction = save(
        directory / "compaction.json",
        {
            "artifact": str(ledger),
            "run_id": "run",
            "judge_model": judge_model,
            "rubric_digest": "rules",
            "samples": [
                {
                    "request": 2,
                    "source_ranges": [[1, 5]],
                    "compaction": {"requests": 1, "usage": asdict(COMPACTION), "seconds": 3.5},
                    "judge": {"requests": 1, "usage": asdict(JUDGE)},
                    "recall": {
                        "mean": 1,
                        "eligible": 1,
                        "scored": 1,
                        "items": [
                            {
                                "id": "a",
                                "fact": "Do not add dependencies",
                                "method": "judge",
                                "status": "preserved",
                            }
                        ],
                    },
                    "factual_consistency": {"score": 2},
                    "compression": {
                        "original_tokens": 80,
                        "compacted_tokens": 20,
                        "ratio": 4,
                        "scope": "replaced payload only",
                        "token_counter": "estimated",
                    },
                }
            ],
        },
    )
    tool = save(
        directory / "tools.json",
        {
            "trial_id": "run",
            "context_budget": {"visible_blocks": 6},
            "judge": {
                "model": judge_model,
                "rubric_digest": "tool rules",
                "requests": 1,
                "usage": asdict(JUDGE),
            },
            "calls": [
                {
                    "id": "1",
                    "status": "succeeded",
                    "scores": {"selection": {}, "parameters": {}} if judged else {},
                },
                {"id": "2", "status": "failed", "scores": {"redundant": {}}},
            ],
            "tool_selection": {"mean": 3 if judged else None, "scored": int(judged), "total": 2},
            "parameter_accuracy": {
                "schema_only": {"valid": 2},
                "semantic": {
                    "mean": 4 if judged else None,
                    "scored": int(judged),
                    "total": 2,
                },
            },
            "execution_failure_rate": {
                "rate": 0.5,
                "total_calls": 2,
                "completed": 2,
                "failed": 1,
                "source": "persisted_tool_runs",
            },
            "redundancy": {
                "rate_among_judged_candidates": 0,
                "judged_candidates": 1,
                "candidate_ids": ["2"],
                "strategy": "exact candidates",
            },
        },
    )
    return batch, tool, compaction


def report(paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    batch, tool, compaction = paths
    return build_report(batch, (tool,), (compaction,))


def test_report_summarizes_and_does_not_count_trace_compaction_twice(tmp_path: Path) -> None:
    result = report(fixtures(tmp_path / "run"))
    components = result["sections"][L1]
    assert components["Tool Selection [repo__project-1#1]"]["coverage"] == 0.5
    assert components["Execution Failure Rate [repo__project-1#1]"]["denominator"] == 2
    assert components["Compression Ratio [repo__project-1#1#2]"]["value"] == 4
    assert result["sections"][L2]["Resolved rate"]["value"] == 1
    assert result["sections"][L3]["Unscored"]["value"] == 1
    assert result["sections"][L3]["Unresolved"]["value"] == 0
    assert result["sections"][L3]["Resolved rate"]["value"] is None

    metrics = result["sections"][EFFICIENCY]
    assert metrics["Agent run · requests"]["value"] is None  # One trial has no model record.
    assert metrics["Compaction · known_cost_usd"]["value"] == pytest.approx(
        float(estimate_deepseek_cost(MODEL, COMPACTION, peak=True))
    )
    assert metrics["Overall · known_cost_usd"]["value"] == pytest.approx(
        float(
            estimate_deepseek_cost(MODEL, COMBINED, peak=True)
            + 2 * estimate_deepseek_cost(MODEL, JUDGE, peak=True)
        )
    )
    assert metrics["Overall · cost_usd"]["value"] is None
    assert "N/A" in markdown(result)


def test_complete_trial_reports_efficiency_and_partial_usage_is_unavailable(tmp_path: Path) -> None:
    batch, tool, compaction = fixtures(tmp_path / "run")
    raw = json.loads(batch.read_text())
    raw["tasks"].pop()
    raw["trials"].pop()
    save(batch, raw)
    metrics = report((batch, tool, compaction))["sections"][EFFICIENCY]
    assert metrics["Agent run · requests"]["value"] == 2
    assert metrics["Agent run · cache_hit_rate"]["value"] == pytest.approx(10 / 120)
    assert metrics["Compaction · requests"]["value"] == 1
    assert "Compaction count" not in metrics
    assert metrics["Compaction · request_seconds"]["value"] == 3.5
    assert metrics["Tool calls"]["value"] == 3
    assert metrics["End-to-end wall seconds"]["value"] == 50
    assert metrics["Overall · requests"]["value"] == 4
    raw["trials"][0]["agent"]["responses_with_usage"] = 1
    raw["trials"][0]["agent"]["usage"] = asdict(COMPACTION)
    save(batch, raw)
    assert report((batch, tool, compaction))["sections"][EFFICIENCY][
        "Agent run · output_tokens"
    ]["value"] == 30  # The ledger is the source when available.
    ledger = batch.parent / "usage.jsonl"
    first, second = (json.loads(line) for line in ledger.read_text().splitlines())
    first["usage"] = None
    ledger.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    metrics = report((batch, tool, compaction))["sections"][EFFICIENCY]
    assert metrics["Agent run · output_tokens"]["value"] is None
    assert metrics["Agent run · cost_usd"]["value"] is None
    assert metrics["Agent run · known_cost_usd"]["coverage"] == 0.5


def test_baseline_reports_regressions_and_requires_matching_budget(tmp_path: Path) -> None:
    baseline = report(fixtures(tmp_path / "base"))
    current = report(fixtures(tmp_path / "current", resolved=False))
    delta = compare(baseline, current)
    assert delta["regressions"] == ["repo__project-1#1"]
    assert delta["sections"][L2]["Resolved rate"]["delta"] == -1
    assert "resolved | unresolved" in markdown(delta)

    changed = report(fixtures(tmp_path / "changed", resolved=False, steps=24))
    delta = compare(baseline, changed)
    assert delta["regressions"] == []
    assert "budget differs" in delta["trials"]["repo__project-1#1"]["differences"]
    assert delta["sections"][L2]["Resolved rate"]["delta"] is None


def test_baseline_checks_judge_model_and_scored_items_independently(tmp_path: Path) -> None:
    baseline = report(fixtures(tmp_path / "base"))
    changed = report(fixtures(tmp_path / "changed", judge_model="deepseek-v4-pro", judged=False))
    comps = compare(baseline, changed)["sections"][L1]
    selection = comps["Tool Selection [repo__project-1#1]"]
    assert selection["delta"] is None
    assert "judge_config differs" in selection["differences"]
    assert "scoring_coverage differs" in selection["differences"]
    assert comps["Compression Ratio [repo__project-1#1#2]"]["delta"] == 0
    assert comps["Execution Failure Rate [repo__project-1#1]"]["delta"] == 0


def test_copied_capture_does_not_charge_trial_twice(tmp_path: Path) -> None:
    batch, tool, compaction = fixtures(tmp_path / "run")
    raw = json.loads(batch.read_text())
    raw["tasks"].pop()
    raw["trials"].pop()
    save(batch, raw)
    capture = json.loads(compaction.read_text())
    copy = tmp_path / "copied.jsonl"
    copyfile(capture["artifact"], copy)
    capture.update(artifact=str(copy), agent={"requests": 1, "usage": asdict(AGENT)})
    copied_report = save(tmp_path / "copy.json", capture)

    expected = report((batch, tool, compaction))
    for paths in ((copied_report,), (copied_report, compaction), (compaction, copied_report)):
        actual = build_report(batch, (tool,), paths)
        assert actual["sections"] == expected["sections"]
    metrics = expected["sections"][EFFICIENCY]
    assert metrics["Overall · requests"]["value"] == 4
    assert metrics["Overall · output_tokens"]["value"] == 46


def test_independent_trials_of_same_compaction_sample_are_counted_once(tmp_path: Path) -> None:
    _, _, source = fixtures(tmp_path / "run")
    capture = json.loads(source.read_text())
    record = json.loads(Path(capture["artifact"]).read_text().splitlines()[1])
    first, second = (tmp_path / f"trial-{i}.json" for i in (1, 2))
    for path in (first, second):
        ledger = path.with_suffix(".jsonl")
        run_id = path.stem
        ledger.write_text(json.dumps({**record, "run_id": run_id, "request": 1}) + "\n")
        capture.update(run_id=run_id, artifact=str(ledger), compaction_model=MODEL)
        capture["samples"][0].update(sample="retry-triage", request=1)
        save(path, capture)
    duplicate = tmp_path / "duplicate.json"
    copyfile(first, duplicate)

    result = build_report(None, compaction_paths=(first, duplicate, second))
    metrics = result["sections"][EFFICIENCY]
    assert metrics["Compaction · requests"]["value"] == 2
    assert metrics["Judge · requests"]["value"] == 2
    assert metrics["Overall · requests"]["value"] == 4
    assert sum(k.startswith("Compression Ratio [retry-triage") for k in result["sections"][L1]) == 2


def test_agent_usage_is_counted_when_capture_has_no_compaction(tmp_path: Path) -> None:
    _, _, source = fixtures(tmp_path / "run")
    capture = json.loads(source.read_text())
    ledger = Path(capture["artifact"])
    ledger.write_text(ledger.read_text().splitlines()[0] + "\n")
    capture.update(samples=[], compaction_model=MODEL)
    save(source, capture)

    metrics = build_report(None, compaction_paths=(source,))["sections"][EFFICIENCY]
    assert metrics["Agent run · requests"]["value"] == 1
    assert metrics["Compaction · requests"]["value"] == 0
    assert metrics["Overall · output_tokens"]["value"] == AGENT.output_tokens


def test_literal_recall_ignores_unrelated_judge_config_and_markdown_uses_runtime_status(
    tmp_path: Path,
) -> None:
    batch, _, compaction = fixtures(tmp_path / "run")
    capture = json.loads(compaction.read_text())
    sample = capture["samples"][0]
    sample["recall"]["items"][0]["method"] = "literal"
    sample["factual_consistency"] = None
    sample["judge"] = {"requests": 0, "usage": None}
    capture["judge_model"] = None
    save(compaction, capture)
    baseline = build_report(batch, compaction_paths=(compaction,))
    capture["judge_model"] = "different"
    save(compaction, capture)
    current = build_report(batch, compaction_paths=(compaction,))

    delta = compare(baseline, current)["sections"][L1]
    assert delta["Key Information Recall [repo__project-1#1#2]"]["delta"] == 0
    assert "| repo__project-1#1 | regression | resolved | failed |" in markdown(current)
