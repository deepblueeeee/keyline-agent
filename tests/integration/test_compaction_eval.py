import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("docker")

from evals import compaction  # noqa: E402
from handshake.models.protocol import ModelContext, ModelTurn, ModelUsage  # noqa: E402
from handshake.models.scripted import ScriptedModel  # noqa: E402


def response(value: dict[str, Any]) -> ModelTurn:
    return ModelTurn(({"type": "text", "text": json.dumps(value)},), usage=ModelUsage(20, 5, 1, 2))


def text(role: str, value: str) -> dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": value}]}


@pytest.mark.parametrize(
    ("summary", "constraint", "progress", "accuracy", "errors", "expected"),
    [
        (
            "src/io.py is edited. Keep the API stable, add no dependencies. Tests pending.",
            "preserved",
            "preserved",
            2,
            [],
            1,
        ),
        ("src/io.py is edited. Tests pending.", "missing", "preserved", 2, [], 2 / 3),
        (
            "src/io.py is edited. Keep the API stable. Tests pending.",
            "partial",
            "preserved",
            2,
            [],
            2.5 / 3,
        ),
        (
            "src/io.py is edited. Keep the API stable, add no dependencies. Tests passed.",
            "preserved",
            "incorrect",
            0,
            ["Tests passed contradicts tests pending."],
            2 / 3,
        ),
        (
            "src/io.py is edited. Keep the API stable, add no dependencies. "
            "Tests pending. Released v3.",
            "preserved",
            "preserved",
            0,
            ["No release appears in the history."],
            1,
        ),
    ],
)
def test_compaction_recall_and_consistency_are_independent(
    tmp_path: Path,
    summary: str,
    constraint: str,
    progress: str,
    accuracy: int,
    errors: list[str],
    expected: float,
) -> None:
    sample = {
        "id": "fixture",
        "target_count": 3,
        "messages": [
            text("user", "Update src/io.py, keep the API stable and add no dependencies."),
            text("assistant", "src/io.py is edited. Tests still need to run."),
            text("user", "Please run the unit tests."),
            text("user", "NEXT: keep changes local to the file."),
        ],
        "facts": [
            {
                "id": "path",
                "sequence": 0,
                "fact": "Exact path is src/io.py",
                "literal": "src/io.py",
            },
            {
                "id": "constraint",
                "sequence": 0,
                "fact": "Keep the API stable and add no dependencies.",
            },
            {"id": "progress", "sequence": 1, "fact": "Source was edited and tests are pending."},
            {"id": "tail", "sequence": 2, "fact": "Run the unit tests."},
            {"id": "recent", "sequence": 3, "fact": "Keep changes local to the file."},
        ],
    }
    compactor = ScriptedModel(
        [
            response(
                {
                    "episodes": [
                        {
                            "start_sequence": 0,
                            "end_sequence": 1,
                            "title": "I/O change",
                            "summary": summary,
                        },
                    ]
                }
            )
        ]
    )
    artifact = tmp_path / "capture.jsonl"
    raw = asyncio.run(compaction.generate(sample, compactor, artifact))
    assert compactor.tool_sets == [()]
    assert "NEXT" not in compactor.contexts[0].messages[0].content[0]["text"]
    assert raw["purpose"] == "compaction"
    assert compaction.read_records(artifact) == [raw]

    judged = {
        "recall": {
            "constraint": {
                "status": constraint,
                "evidence": "Compare the user constraint and summary.",
            },
            "progress": {"status": progress, "evidence": "Compare the recorded test state."},
        },
        "consistency": {
            "score": accuracy,
            "reason": "Compared against the source span.",
            "errors": errors,
        },
    }
    judge = ScriptedModel([response(judged)])
    report = asyncio.run(compaction.score_record(raw, judge))
    assert judge.tool_sets == [()]
    payload = json.loads(judge.contexts[0].messages[0].content[0]["text"])
    assert [key["id"] for key in payload["key_items"]] == ["constraint", "progress"]
    assert [item["sequence"] for item in payload["original"]] == [0, 1]
    assert report["recall"]["mean"] == pytest.approx(expected)
    assert report["recall"]["coverage"] == 1
    assert report["recall"]["incorrect_ids"] == (["progress"] if progress == "incorrect" else [])
    assert [item["status"] for item in report["recall"]["items"][-2:]] == [
        "retained_outside_summary"
    ] * 2
    assert report["factual_consistency"]["score"] == accuracy
    assert report["compaction"]["usage"]["cache_read_tokens"] == 2
    assert report["judge"]["usage"]["output_tokens"] == 5
    compression = report["compression"]
    assert compression["uncompacted_target_tokens"] > 0 and compression["protected_tokens"] > 0
    assert compression["ratio"] == compression["original_tokens"] / compression["compacted_tokens"]
    assert compression["complete_context_tokens"] is None


def test_trace_rollup_and_missing_judge_grade(tmp_path: Path) -> None:
    from evals.swebench_run import BudgetedModel
    from handshake.context.condenser import LLMHistoryCondenser
    from handshake.context.state import HistoryEpisode, SourceRange

    artifact = tmp_path / "usage.jsonl"
    scripted = ScriptedModel(
        [
            response({}),
            response(
                {
                    "episodes": [
                        {
                            "start_sequence": 0,
                            "end_sequence": 5,
                            "title": "Done",
                            "summary": "Tests pending.",
                        },
                    ]
                }
            ),
        ]
    )
    model = BudgetedModel(scripted, artifact, 2)
    asyncio.run(model.complete(ModelContext("system", "build", "", (), ()), ()))
    asyncio.run(
        LLMHistoryCondenser(model).roll_up(
            (
                HistoryEpisode(SourceRange(0, 2), "Code", "Source changed."),
                HistoryEpisode(SourceRange(3, 5), "Verification", "Tests pending."),
            )
        )
    )
    args = argparse.Namespace(sample=None, saved=None, trial=tmp_path, judge_model=None)
    report = asyncio.run(compaction.run(args))
    assert report["agent"]["requests"] == 1
    (entry,) = report["samples"]
    assert entry["compaction"]["requests"] == 1
    assert entry["source_ranges"] == [[0, 5]]
    assert entry["judge"]["requests"] == 0
    assert entry["recall"]["mean"] is None
    assert entry["factual_consistency"] is None
    raw = compaction.read_records(artifact)[-1]
    invalid = ScriptedModel([response({"consistency": {"score": 2, "reason": "", "errors": []}})])
    assert asyncio.run(compaction.score_record(raw, invalid))["factual_consistency"] is None
    saved_report = asyncio.run(
        compaction.run(
            argparse.Namespace(
                sample=None,
                saved=artifact,
                trial=None,
                judge_model=None,
            )
        )
    )
    assert saved_report == report


def test_trial_without_compaction_is_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "usage.jsonl").write_text('{"purpose":"agent","usage":null}\n')
    monkeypatch.setattr(
        compaction, "read_deepseek_api_key", lambda: pytest.fail("No Judge request is needed")
    )
    args = argparse.Namespace(sample=None, saved=None, trial=tmp_path, judge_model="unused")
    report = asyncio.run(compaction.run(args))
    assert report["status"] == "not_applicable"
    assert report["samples"] == []
    assert report["agent"]["requests"] == 1


def test_declined_or_malformed_compaction_is_unscored() -> None:
    for output, expected in (("{}", "invalid_or_failed"), ('{"episodes":[]}', "declined")):
        raw = {"request": 1, "usage": None, "compaction": {"output": output, "seconds": 0}}
        report = asyncio.run(compaction.score_record(raw, None))
        assert report["status"] == expected
        assert report["compression"] is None
        assert report["judge"]["requests"] == 0
