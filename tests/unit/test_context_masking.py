from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from handshake.artifacts import open_text_artifact
from handshake.config import DeepSeekConfig
from handshake.context import (
    AgentContextManager,
    ContextBudgetPolicy,
    ContextPressureError,
    ContextView,
    RawHistorySegment,
    estimate_model_context,
)
from handshake.models.base import ToolDefinition
from handshake.models.protocol import ModelMessage, ModelTurn, ToolCall
from handshake.models.scripted import ScriptedModel
from handshake.runtime.loop import start_run
from handshake.runtime.state import Run, Session, ToolResult
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools.files import read_file
from handshake.tools.registry import ToolBinding, ToolRegistry
from tests.history_support import TEST_AGENT_SPEC, fresh_store, history, ingest, run_async

_HISTORY: dict[str, list[ModelMessage]] = {}


class CharacterTokenEstimator:
    def estimate_text(self, text: str) -> int:
        return len(text)


def test_default_context_budget_reserves_the_full_model_output_limit() -> None:
    assert ContextBudgetPolicy().reserved_output_tokens == DeepSeekConfig().max_tokens == 16_384


def _policy(
    *,
    context_window: int = 100_000,
    trigger: float = 0.8,
    target: float = 0.6,
    recent_messages: int = 2,
    recent_tokens: int = 0,
    ingress_tokens: int = 10_000,
) -> ContextBudgetPolicy:
    return ContextBudgetPolicy(
        context_window_tokens=context_window,
        reserved_output_tokens=0,
        pressure_trigger_ratio=trigger,
        pressure_target_ratio=target,
        protected_recent_messages=recent_messages,
        protected_recent_tokens=recent_tokens,
        max_tool_result_tokens=ingress_tokens,
    )


def _session_run(tmp_path: Path) -> tuple[Session, Run]:
    session = Session(workspace=tmp_path)
    _HISTORY[session.session_id] = []
    run = Run(goal="Keep working", session_id=session.session_id, agent_spec_id="test")
    return session, run


def _messages(session: Session) -> list[ModelMessage]:
    return _HISTORY[session.session_id]


def _capture_after_ingress(
    manager: AgentContextManager,
    session: Session,
    run: Run,
) -> ContextView:
    start = manager.message_count(session.context_scope)
    ingest(manager, session, _messages(session)[start:], start=start)
    return manager.capture(session, run)


def _apply_pressure(
    manager: AgentContextManager,
    session: Session,
    run: Run,
    tools: tuple[ToolDefinition, ...] = (),
) -> ContextView:
    start = manager.message_count(session.context_scope)
    ingest(manager, session, _messages(session)[start:], start=start)
    pressure = manager.measure_pressure(session, run, tools)
    if pressure >= manager.policy.pressure_trigger_tokens:
        while (
            pressure > manager.policy.pressure_target_tokens
            and manager.mask_oldest_stale_result(session.context_scope)
        ):
            pressure = manager.measure_pressure(session, run, tools)
    manager.ensure_below_hard_limit(pressure)
    return manager.capture(session, run)


def _tool_round(
    run_id: str,
    call_id: str,
    tool_name: str,
    content: str,
    *,
    is_error: bool = False,
) -> tuple[ModelMessage, ModelMessage]:
    return (
        ModelMessage(
            "assistant",
            (
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool_name,
                    "input": {"path": "file.py"},
                },
            ),
            run_id,
        ),
        ModelMessage(
            "user",
            (
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                    "is_error": is_error,
                },
            ),
            run_id,
        ),
    )


def test_ingress_externalizes_every_large_result_but_preserves_raw_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "handshake-home"))
    session, run = _session_run(tmp_path)
    successful = "HEAD-success-" + "x" * 1_000 + "-TAIL-success"
    failed = "HEAD-failed-" + "y" * 1_000 + "-TAIL-failed"
    _messages(session).extend(_tool_round(run.run_id, "read", "read_file", successful))
    _messages(session).extend(
        _tool_round(run.run_id, "shell", "run_shell", failed, is_error=True)
    )
    estimator = CharacterTokenEstimator()
    manager = AgentContextManager(
        _policy(recent_messages=8, ingress_tokens=600),
        estimator,
    )

    view = _capture_after_ingress(manager, session, run)
    context = manager.render(view)

    visible_results = [context.messages[1].content[0], context.messages[3].content[0]]
    assert all(
        estimator.estimate_text(json.dumps(block, ensure_ascii=False, separators=(",", ":"))) <= 600
        for block in visible_results
    )
    assert "HEAD-success" not in visible_results[0]["content"]
    assert "HEAD-failed" not in visible_results[1]["content"]
    assert visible_results[0]["is_error"] is False
    assert visible_results[1]["is_error"] is True
    assert "status=success" in visible_results[0]["content"]
    assert "status=failure" in visible_results[1]["content"]
    assert _messages(session)[1].content[0]["content"] == successful
    assert _messages(session)[3].content[0]["content"] == failed
    assert "tool result externalized" in (
        view.protected_recent.messages[1].message.content[0]["content"]
    )
    successful_stream = open_text_artifact(
        visible_results[0]["content"].split("path=", 1)[1].split("]", 1)[0]
    )
    failed_stream = open_text_artifact(
        visible_results[1]["content"].split("path=", 1)[1].split("]", 1)[0]
    )
    assert successful_stream is not None
    assert failed_stream is not None
    with successful_stream, failed_stream:
        assert successful_stream.read() == successful
        assert failed_stream.read() == failed
    assert f"bytes={len(successful.encode('utf-8'))}" in visible_results[0]["content"]
    assert f"characters={len(successful)}" in visible_results[0]["content"]
    assert "lines=1" in visible_results[0]["content"]


def test_recent_boundary_uses_larger_message_or_token_window(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).extend(
        ModelMessage("user", ({"type": "text", "text": "x"},), run.run_id) for _ in range(12)
    )
    fixed_manager = AgentContextManager(
        _policy(recent_messages=8, recent_tokens=1), CharacterTokenEstimator()
    )
    fixed_view = _capture_after_ingress(fixed_manager, session, run)

    token_manager = AgentContextManager(
        _policy(recent_messages=2, recent_tokens=400), CharacterTokenEstimator()
    )
    token_view = _capture_after_ingress(token_manager, session, run)

    assert len(fixed_view.protected_recent.messages) == 8
    assert len(token_view.protected_recent.messages) > 2
    assert isinstance(fixed_view.history[0], RawHistorySegment)


def test_recent_boundary_does_not_split_tool_round(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "old"},), run.run_id)
    )
    _messages(session).extend(_tool_round(run.run_id, "read", "read_file", "result"))
    manager = AgentContextManager(_policy(recent_messages=1), CharacterTokenEstimator())

    view = _capture_after_ingress(manager, session, run)

    assert tuple(item.sequence for item in view.protected_recent.messages) == (1, 2)


def test_pressure_masks_stale_results_oldest_first_until_target(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    for index in range(5):
        _messages(session).extend(
            _tool_round(
                run.run_id,
                f"read-{index}",
                "read_file",
                f"result-{index}-" + "x" * 2_000,
            )
        )
    estimator = CharacterTokenEstimator()
    manager = AgentContextManager(
        _policy(
            context_window=16_000,
            trigger=0.65,
            target=0.45,
            recent_messages=2,
        ),
        estimator,
    )

    view = _apply_pressure(manager, session, run)
    context = manager.render(view)
    masked_sequences = [
        sequence
        for sequence, message in enumerate(context.messages)
        if message.role == "user"
        and message.content[0].get("type") == "tool_result"
        and "stale tool result masked" in message.content[0]["content"]
    ]

    assert masked_sequences == [1, 3, 5]
    assert "stale tool result masked" in context.messages[1].content[0]["content"]
    assert "result-3" in context.messages[7].content[0]["content"]
    assert estimate_model_context(context, (), estimator) <= manager.policy.pressure_target_tokens


def test_stale_candidates_remain_visible_below_pressure_trigger(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).extend(_tool_round(run.run_id, "read", "read_file", "old result"))
    _messages(session).extend(
        ModelMessage("user", ({"type": "text", "text": f"recent-{index}"},), run.run_id)
        for index in range(2)
    )
    manager = AgentContextManager(
        _policy(context_window=100_000, recent_messages=2),
        CharacterTokenEstimator(),
    )

    view = _apply_pressure(manager, session, run)

    assert view.history
    assert manager.render(view).messages[1].content[0]["content"] == "old result"


def test_pressure_measurement_does_not_execute_masking(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).extend(_tool_round(run.run_id, "read", "read_file", "x" * 2_000))
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "z" * 6_000},), run.run_id)
    )
    manager = AgentContextManager(
        _policy(
            context_window=12_000,
            trigger=0.55,
            target=0.25,
            recent_messages=1,
        ),
        CharacterTokenEstimator(),
    )
    ingest(manager, session, _messages(session))

    pressure = manager.measure_pressure(session, run, ())
    before = manager.build(session, run)

    assert pressure >= manager.policy.pressure_trigger_tokens
    assert before.messages[1].content[0]["content"] == "x" * 2_000
    assert manager.mask_oldest_stale_result(session.context_scope)
    assert "stale tool result masked" in manager.build(session, run).messages[1].content[0][
        "content"
    ]


def test_tool_schemas_contribute_to_pressure(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    manager = AgentContextManager(_policy(context_window=100_000), CharacterTokenEstimator())
    tool = ToolDefinition(
        "large_schema",
        "x" * 1_000,
        {"type": "object", "description": "y" * 1_000},
    )
    context = manager.build(session, run)

    assert estimate_model_context(context, (tool,), manager.estimator) > estimate_model_context(
        context, (), manager.estimator
    )


def test_pressure_eligibility_depends_only_on_stale_boundary(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).extend(_tool_round("one", "same", "read_file", "tiny"))
    _messages(session).extend(
        _tool_round("two", "same", "read_file", "failed", is_error=True)
    )
    _messages(session).extend(_tool_round(run.run_id, "edit", "edit_file", "edited"))
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "z" * 6_000},), run.run_id)
    )
    manager = AgentContextManager(
        _policy(
            context_window=12_000,
            trigger=0.55,
            target=0.25,
            recent_messages=1,
        ),
        CharacterTokenEstimator(),
    )

    view = _apply_pressure(manager, session, run)
    context = manager.render(view)
    stale = [
        (sequence, message.content[0]["content"])
        for sequence, message in enumerate(context.messages)
        if message.role == "user"
        and message.content[0].get("type") == "tool_result"
        and "stale tool result masked" in message.content[0]["content"]
    ]

    assert [sequence for sequence, _content in stale] == [1, 3, 5]
    assert "status=failure" in stale[1][1]


def test_pressure_masks_every_stale_block_in_multi_result_message(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).extend(
        [
            ModelMessage(
                "assistant",
                (
                    {"type": "tool_use", "id": "read", "name": "read_file", "input": {}},
                    {"type": "tool_use", "id": "edit", "name": "edit_file", "input": {}},
                ),
                run.run_id,
            ),
            ModelMessage(
                "user",
                (
                    {
                        "type": "tool_result",
                        "tool_use_id": "read",
                        "content": "read result",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "edit",
                        "content": "edit result",
                        "is_error": False,
                    },
                ),
                run.run_id,
            ),
            ModelMessage("user", ({"type": "text", "text": "z" * 6_000},), run.run_id),
        ]
    )
    manager = AgentContextManager(
        _policy(
            context_window=12_000,
            trigger=0.55,
            target=0.25,
            recent_messages=1,
        ),
        CharacterTokenEstimator(),
    )

    view = _apply_pressure(manager, session, run)
    blocks = manager.render(view).messages[1].content

    assert "stale tool result masked" in blocks[0]["content"]
    assert "stale tool result masked" in blocks[1]["content"]


def test_ingress_projection_is_replaced_when_result_becomes_stale_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "handshake-home"))
    session, run = _session_run(tmp_path)
    _messages(session).extend(_tool_round(run.run_id, "read", "read_file", "x" * 4_000))
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "y" * 5_000},), run.run_id)
    )
    manager = AgentContextManager(
        _policy(
            context_window=8_000,
            trigger=0.6,
            target=0.4,
            recent_messages=1,
            ingress_tokens=1_000,
        ),
        CharacterTokenEstimator(),
    )

    view = _apply_pressure(manager, session, run)
    assert "stale tool result masked" in manager.render(view).messages[1].content[0]["content"]


def test_pressure_exhaustion_only_fails_at_hard_limit(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "x" * 4_000},), run.run_id)
    )
    estimator = CharacterTokenEstimator()
    above_target = AgentContextManager(
        _policy(context_window=10_000, trigger=0.5, target=0.3),
        estimator,
    )

    view = _apply_pressure(above_target, session, run)

    assert estimate_model_context(above_target.render(view), (), estimator) < 10_000

    hard_limit = AgentContextManager(
        _policy(context_window=4_000, trigger=0.5, target=0.3),
        estimator,
    )
    with pytest.raises(ContextPressureError):
        _apply_pressure(hard_limit, session, run)


@run_async
async def test_loop_does_not_call_model_above_hard_context_limit(tmp_path: Path) -> None:
    session, run = _session_run(tmp_path)
    store = fresh_store(tmp_path, session)
    model = ScriptedModel([])
    manager = AgentContextManager(
        _policy(context_window=500, trigger=0.5, target=0.3),
        CharacterTokenEstimator(),
    )

    with pytest.raises(ContextPressureError):
        await start_run(
            session,
            run,
            model,
            ToolRegistry(),
            context_manager=manager,
            store=store,
            agent_spec=TEST_AGENT_SPEC,
        )
    assert model.contexts == []


@run_async
async def test_loop_externalizes_large_result_but_persists_complete_result(
    tmp_path: Path, monkeypatch
) -> None:
    original = "BEGIN\n" + "value\n" * 400 + "END"
    definition = ToolDefinition("large_read", "read", {"type": "object"})
    registry = ToolRegistry()
    registry.register(
        ToolBinding(definition, lambda _workspace, _arguments: ToolResult(True, original))
    )
    model = ScriptedModel(
        [
            ModelTurn(
                content=(
                    {
                        "type": "tool_use",
                        "id": "large",
                        "name": "large_read",
                        "input": {},
                    },
                ),
                tool_calls=(ToolCall("large", "large_read", {}),),
                stop_reason="tool_use",
            ),
            ModelTurn(
                content=({"type": "text", "text": "done"},),
                stop_reason="end_turn",
            ),
        ]
    )
    session, run = _session_run(tmp_path)
    store = SQLiteStateStore(tmp_path / "handshake.db")
    store.create_session(session)
    home = tmp_path / "handshake-home"
    monkeypatch.setenv("HANDSHAKE_HOME", str(home))
    manager = AgentContextManager(
        _policy(recent_messages=8, ingress_tokens=600), CharacterTokenEstimator()
    )

    await start_run(
        session,
        run,
        model,
        registry,
        store=store,
        context_manager=manager,
        agent_spec=replace(
            TEST_AGENT_SPEC,
            allowed_tools=TEST_AGENT_SPEC.allowed_tools | {"large_read"},
        ),
    )

    visible = model.contexts[1].messages[-1].content[0]["content"]
    loaded_session = store.load_session(session.session_id)
    persisted = history(store, loaded_session)[-2].content[0]["content"]
    assert "tool result externalized" in visible
    assert len(visible) <= 600
    assert "lines=407" in visible
    assert json.loads(persisted)["stdout"] == original
    output_reference = visible.split("path=", 1)[1].split("]", 1)[0]
    assert output_reference.startswith("@handshake/artifacts/")
    recovered = read_file(
        session.workspace,
        {"path": output_reference, "start_character": 1, "end_character": 10_000},
    )
    assert recovered.ok
    assert "[tool result]\nok: true\nexit_code: None\n[stdout]\n" in recovered.stdout
    assert original in recovered.stdout
    assert recovered.stdout.endswith("\n[stderr]\n")


def test_new_manager_rebuilds_masking_state_from_raw_sqlite_history(
    tmp_path: Path,
) -> None:
    session, run = _session_run(tmp_path)
    store = SQLiteStateStore(tmp_path / "handshake.db")
    store.create_session(session)
    manager = AgentContextManager(
        _policy(
            context_window=12_000,
            trigger=0.55,
            target=0.25,
            recent_messages=1,
        ),
        CharacterTokenEstimator(),
    )
    _messages(session).extend(_tool_round(run.run_id, "shell", "run_shell", "raw-output"))
    ingest(manager, session, _messages(session))
    _messages(session).append(
        ModelMessage("user", ({"type": "text", "text": "z" * 6_000},), run.run_id),
    )
    ingest(manager, session, _messages(session)[2:], start=2)
    store.create_run(session, run, tuple(_messages(session)))

    pressure = manager.measure_pressure(session, run, ())
    assert pressure >= manager.policy.pressure_trigger_tokens
    assert manager.mask_oldest_stale_result(session.context_scope)
    store.checkpoint_run(session, run)

    loaded_session = store.load_session(session.session_id)
    loaded_run = store.load_run(run.run_id)
    restored = AgentContextManager(manager.policy, manager.estimator)
    restored.ingest_messages(
        loaded_session.context_scope, store.read_messages(loaded_run.context_scope_id)
    )
    restored_pressure = restored.measure_pressure(loaded_session, loaded_run, ())
    assert restored_pressure >= restored.policy.pressure_trigger_tokens
    assert restored.mask_oldest_stale_result(loaded_session.context_scope)
    context = restored.build(loaded_session, loaded_run)
    assert "stale tool result masked" in context.messages[1].content[0]["content"]
    assert history(store, loaded_session)[1].content[0]["content"] == "raw-output"
