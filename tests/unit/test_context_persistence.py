from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from handshake.artifacts import open_text_artifact
from handshake.context import (
    AgentContextManager,
    ContextBudgetPolicy,
    ContextCheckpoint,
    ContextViewError,
    HistoryEpisode,
    RawHistorySegment,
    SourceRange,
)
from handshake.models.protocol import ModelMessage
from handshake.runtime.state import Run, Session
from tests.history_support import fresh_store, history


class Characters:
    def estimate_text(self, text: str) -> int:
        return len(text)


def _messages(run: Run, count: int) -> tuple[ModelMessage, ...]:
    return tuple(
        ModelMessage(
            "user",
            ({"type": "text", "text": f"message {index}"},),
            run.run_id,
        )
        for index in range(count)
    )


def test_restore_skips_episode_sources_and_streams_incremental_suffix(
    tmp_path: Path,
) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="continue", session_id=session.session_id)
    store = fresh_store(tmp_path, session)
    store.create_run(session, run, _messages(run, 6))
    checkpoint = ContextCheckpoint(
        source_message_count=6,
        recent_start=5,
        episodes=(
            HistoryEpisode(SourceRange(0, 3), "Completed stage", "The stage is done."),
        ),
        reduced_message_count=6,
    )
    saved = store.save_context_checkpoint(run.context_scope_id, checkpoint)
    store.append_messages(
        run.context_scope_id,
        (ModelMessage("user", ({"type": "text", "text": "new suffix"},), run.run_id),),
    )
    reads: list[tuple[int, int | None]] = []

    def read(start: int, end: int | None):
        reads.append((start, end))
        yield from store.read_messages(run.context_scope_id, start, end)

    manager = AgentContextManager(
        ContextBudgetPolicy(protected_recent_messages=1, protected_recent_tokens=0),
        Characters(),
    )
    manager.restore(run.context_scope, saved, store.message_count(run.context_scope_id), read)

    view = manager.capture(session, run)
    assert reads == [(4, 7)]
    assert view.history[0] == checkpoint.episodes[0]
    assert isinstance(view.history[1], RawHistorySegment)
    assert tuple(manager._states[run.context_scope_id].messages) == (4, 5, 6)
    assert manager.pending_checkpoint(run.context_scope) is None


def test_masking_checkpoint_restores_identical_view_without_changing_raw_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "handshake-home"))
    session = Session(workspace=tmp_path)
    run = Run(goal="inspect", session_id=session.session_id)
    store = fresh_store(tmp_path, session)
    messages = (
        ModelMessage(
            "assistant",
            ({"type": "tool_use", "id": "read", "name": "read_file", "input": {}},),
            run.run_id,
        ),
        ModelMessage(
            "user",
            (
                {
                    "type": "tool_result",
                    "tool_use_id": "read",
                    "content": "large durable result " * 100,
                    "is_error": False,
                },
            ),
            run.run_id,
        ),
        ModelMessage("user", ({"type": "text", "text": "recent"},), run.run_id),
    )
    appended = store.create_run(session, run, messages)
    policy = ContextBudgetPolicy(
        protected_recent_messages=1,
        protected_recent_tokens=0,
        max_tool_result_tokens=600,
    )
    manager = AgentContextManager(policy, Characters())
    manager.ingest_messages(run.context_scope, appended)
    assert manager.mask_oldest_stale_result(run.context_scope)
    pending = manager.pending_checkpoint(run.context_scope)
    assert pending is not None
    saved = store.save_context_checkpoint(run.context_scope_id, pending)
    manager.checkpoint_saved(run.context_scope, saved)

    restored = AgentContextManager(policy, Characters())
    restored.restore(
        run.context_scope,
        store.load_context_checkpoint(run.context_scope_id),
        store.message_count(run.context_scope_id),
        lambda start, end: store.read_messages(run.context_scope_id, start, end),
    )

    assert restored.build(session, run) == manager.build(session, run)
    assert restored.pending_checkpoint(run.context_scope) is None
    assert "stale tool result masked" in restored.build(session, run).messages[1].content[0][
        "content"
    ]
    assert "large durable result" in history(store, session)[1].content[0]["content"]


def test_restore_rebuilds_externalized_result_and_reuses_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HANDSHAKE_HOME", str(tmp_path / "handshake-home"))
    session = Session(workspace=tmp_path)
    run = Run(goal="inspect", session_id=session.session_id)
    store = fresh_store(tmp_path, session)
    original = "line\n" * 1_000
    messages = (
        ModelMessage(
            "assistant",
            ({"type": "tool_use", "id": "large", "name": "test", "input": {}},),
            run.run_id,
        ),
        ModelMessage(
            "user",
            (
                {
                    "type": "tool_result",
                    "tool_use_id": "large",
                    "content": original,
                    "is_error": False,
                },
            ),
            run.run_id,
        ),
    )
    appended = store.create_run(session, run, messages)
    policy = ContextBudgetPolicy(max_tool_result_tokens=600)
    manager = AgentContextManager(policy, Characters())
    manager.ingest_messages(run.context_scope, appended)
    before = manager.build(session, run)

    restored = AgentContextManager(policy, Characters())
    restored.restore(
        run.context_scope,
        None,
        store.message_count(run.context_scope_id),
        lambda start, end: store.read_messages(run.context_scope_id, start, end),
    )

    assert restored.build(session, run) == before
    marker = before.messages[1].content[0]["content"]
    reference = marker.split("path=", 1)[1].split("]", 1)[0]
    assert reference.startswith("@handshake/artifacts/")
    stream = open_text_artifact(reference)
    assert stream is not None
    with stream:
        assert stream.read() == original
    assert history(store, session)[1].content[0]["content"] == original


@pytest.mark.parametrize(
    "checkpoint",
    (
        ContextCheckpoint(source_message_count=4),
        ContextCheckpoint(
            source_message_count=3,
            recent_start=2,
            episodes=(HistoryEpisode(SourceRange(1, 1), "gap", "invalid"),),
        ),
    ),
)
def test_restore_rejects_incompatible_or_invalid_metadata(
    tmp_path: Path, checkpoint: ContextCheckpoint
) -> None:
    session = Session(workspace=tmp_path)
    run = Run(goal="inspect", session_id=session.session_id)
    store = fresh_store(tmp_path, session)
    store.create_run(session, run, _messages(run, 3))
    manager = AgentContextManager(estimator=Characters())

    with pytest.raises(ContextViewError):
        manager.restore(
            run.context_scope,
            replace(checkpoint, revision=1),
            store.message_count(run.context_scope_id),
            lambda start, end: store.read_messages(run.context_scope_id, start, end),
        )
