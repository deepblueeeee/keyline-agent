import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("docker")
pytest.importorskip("jsonschema")

from evals import swebench_run, tool_calling  # noqa: E402
from evals.judge import check_judge_input  # noqa: E402
from keyline_agent.models.protocol import ModelTurn, ModelUsage, ToolCall  # noqa: E402
from keyline_agent.models.scripted import ScriptedModel  # noqa: E402
from keyline_agent.runtime.state import ToolResult  # noqa: E402
from keyline_agent.tools import create_default_registry  # noqa: E402


@pytest.fixture
def trial(tmp_path: Path) -> Path:
    checkout = tmp_path / "workspace"
    checkout.mkdir()
    (checkout / "logs").mkdir()
    (checkout / "target.py").write_text("before\n")
    (checkout / "other.py").write_text("other\n")
    workspace = Mock(path=checkout)
    workspace.execute_file.side_effect = create_default_registry(lambda _path, _argv: True).execute
    workspace.execute.return_value = ToolResult(ok=True, stdout="test passed", exit_code=0)
    test = {"argv": ["python", "check.py"]}
    calls = (
        ToolCall("read", "read_file", {"path": "target.py"}),
        ToolCall("wrong-tool", "list_files", {"path": "logs"}),
        ToolCall("wrong-path", "read_file", {"path": "other.py"}),
        ToolCall("bad-schema", "read_file", {"path": 123}),
        ToolCall("before", "run_tests", test),
        ToolCall(
            "edit", "edit_file", {"path": "target.py", "old_text": "before", "new_text": "after"}
        ),
        ToolCall("after", "run_tests", test),
        ToolCall("again", "run_tests", test),
    )
    turns = [
        ModelTurn(
            ({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments},),
            (call,),
            "tool_use",
        )
        for call in calls
    ]
    turns.append(ModelTurn(({"type": "text", "text": "done"},), stop_reason="end_turn"))
    asyncio.run(
        swebench_run.run_agent(
            workspace,
            "Inspect, fix, and test target.py",
            "scripted",
            ScriptedModel(turns),
            tmp_path,
            swebench_run.Budget(),
        )
    )
    return tmp_path


def _response(grades: list[dict]) -> ModelTurn:
    return ModelTurn(
        ({"type": "text", "text": json.dumps({"grades": grades})},), usage=ModelUsage(11, 2)
    )


def test_existing_tool_calling_output_blocks_rescoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "tool_calling.json"
    output.write_text("prior scores\n")
    monkeypatch.setattr(sys, "argv", ["tool_calling", str(tmp_path), "--live"])
    monkeypatch.setattr(tool_calling, "prepare", lambda _: pytest.fail("Output exists"))

    with pytest.raises(SystemExit, match="2"):
        tool_calling.main()

    assert "already exists" in capsys.readouterr().err
    assert output.read_text() == "prior scores\n"


def test_tool_grading_distinguishes_choice_parameters_and_repeats(trial: Path) -> None:
    calls, payload = tool_calling.prepare(trial)
    rows = {call["call_id"]: call for call in calls}
    batches = {batch["calls"][0]["call_id"]: batch for batch in payload["batches"]}
    assert len(calls) == 8
    assert rows["wrong-path"]["parameter_legality"] == "valid"
    assert rows["bad-schema"]["schema_errors"]
    assert rows["bad-schema"]["parameter_legality"] == "invalid"
    assert rows["after"]["repeat_of"] == rows["before"]["id"]
    after = batches["after"]["calls"][0]["comparison"]
    again = batches["again"]["calls"][0]["comparison"]
    assert [item["tool"] for item in after["intervening"]] == ["edit_file"]
    assert again["intervening"] == []
    assert again["earlier_result_visible"] is True
    assert any("test passed" in block["excerpt"]
               for block in batches["again"]["context"]["visible"])
    assert batches["read"]["context"]["visible"][0]["role"] == "user"
    assert "/testbed" in batches["read"]["context"]["system_prompt"]
    for batch in payload["batches"]:
        batch_ids = {call["call_id"] for call in batch["calls"]}
        assert len({call["turn"] for call in batch["calls"]}) == 1
        assert not batch_ids.intersection(
            block.get("tool_use_id") for block in batch["context"]["visible"]
        )
    ledger = [json.loads(line) for line in (trial / "usage.jsonl").read_text().splitlines()]
    assert sum("tool_definitions" in record for record in ledger) == 1
    assert ledger[rows["read"]["request"] - 1]["tool_decision"]["calls"][0]["id"] == "read"
    expected = {
        "read": ("selection", {"score": 5, "reason": "Read the requested file"}),
        "wrong-tool": ("selection", {"score": 1, "reason": "Unrelated directory"}),
        "wrong-path": ("parameters", {"score": 1, "reason": "Targets other.py"}),
        "after": ("redundant", {"value": False, "reason": "Edit between tests"}),
        "again": ("redundant", {"value": True, "reason": "Same command and no intervening action"}),
    }
    responses = []
    for batch in payload["batches"]:
        grades = []
        for item in batch["calls"]:
            if (call_id := item["id"].split(":", 1)[1]) in expected:
                dimension, grade = expected[call_id]
                grades.append({"id": item["id"], dimension: {**grade, "evidence": [item["id"]]}})
        responses.append(_response(grades))
    judge = ScriptedModel(responses)
    scored = asyncio.run(tool_calling.score_batches(judge, payload))
    result = tool_calling.summarize(calls, payload["trial_id"], scored)
    assert judge.tool_sets == [()] * len(responses)
    assert result["tool_selection"]["mean"] == 3
    assert result["tool_selection"]["coverage"] == 2 / 8
    assert result["parameter_accuracy"]["semantic"]["mean"] == 1
    assert result["execution_failure_rate"]["rate"] == 1 / 8
    assert result["redundancy"]["candidate_ids"] == [rows["after"]["id"], rows["again"]["id"]]
    assert result["redundancy"]["candidate_coverage"] == 1
    assert result["redundancy"]["rate_among_judged_candidates"] == 1 / 2
    assert result["redundancy"]["redundant"] == 1
    assert result["judge"]["responses_with_usage"] == len(responses)
    assert result["judge"]["usage"]["input_tokens"] == len(responses) * 11


def test_malformed_judge_grade_is_unscored_and_preserves_usage(trial: Path) -> None:
    calls, payload = tool_calling.prepare(trial)
    payload["batches"] = payload["batches"][:1]
    call_id = payload["batches"][0]["calls"][0]["id"]
    response = _response([{
        "id": call_id,
        "selection": {"score": 0},
        "parameters": {"score": 5, "reason": "Correct path", "evidence": ["target.py"]},
    }])
    scored = asyncio.run(tool_calling.score_batches(ScriptedModel([response]), payload))
    result = tool_calling.summarize(calls, payload["trial_id"], scored)
    assert result["tool_selection"]["mean"] is None
    assert result["parameter_accuracy"]["semantic"]["mean"] == 5
    assert result["redundancy"]["rate_among_judged_candidates"] is None
    assert result["judge"]["errors"][0]["ids"] == [call_id]
    assert result["judge"]["usage"]["output_tokens"] == 2


def test_no_calls_are_explicitly_unscored() -> None:
    judge = ScriptedModel([])
    scored = asyncio.run(tool_calling.score_batches(judge, {"batches": []}))
    empty = tool_calling.summarize([], "empty", scored)
    assert empty["status"] == "not_applicable"
    assert empty["judge"]["requests"] == 0
    assert empty["tool_selection"]["coverage"] is None
    assert empty["execution_failure_rate"]["rate"] is None
    assert empty["redundancy"]["rate_among_judged_candidates"] is None


@pytest.mark.parametrize(
    ("stop_reason", "error"), [("end_turn", "invalid_json"), ("max_tokens", "truncated_output")]
)
def test_invalid_judge_json_keeps_usage_and_other_batches(
    trial: Path, stop_reason: str, error: str,
) -> None:
    _, payload = tool_calling.prepare(trial)
    payload["batches"] = payload["batches"][:2]
    first, second = (batch["calls"][0]["id"] for batch in payload["batches"])
    judge = ScriptedModel([
        ModelTurn(
            ({"type": "text", "text": '{"grades":['},),
            stop_reason=stop_reason, usage=ModelUsage(7, 3),
        ),
        _response([{
            "id": second,
            "selection": {"score": 3, "reason": "Limited relevance", "evidence": [second]},
        }]),
    ])
    result = asyncio.run(tool_calling.score_batches(judge, payload))
    assert result["errors"] == [{"ids": [first], "error": error}]
    assert set(result["grades"]) == {second}
    assert result["responses_with_usage"] == 2
    assert result["usage"]["input_tokens"] == 18
    assert result["usage"]["output_tokens"] == 5


def test_outbound_payload_rejects_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_API_TOKEN", "test-private-value")
    for payload in ("test-private-value", '{"api_key":"abcdef12345"}', "configured-key"):
        with pytest.raises(ValueError, match="possible credential"):
            check_judge_input(payload, "configured-key")
    check_judge_input("public source excerpt", "configured-key")
