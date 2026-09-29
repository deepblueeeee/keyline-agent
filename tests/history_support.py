"""Helpers for tests whose context streams are owned by SQLite."""

from pathlib import Path

from handshake.context import AgentContextManager, SequencedMessage
from handshake.models.protocol import ModelMessage
from handshake.runtime.agents import AgentSpec
from handshake.runtime.persistence import RunStore
from handshake.runtime.state import Run, Session
from handshake.storage.sqlite import SQLiteStateStore
from handshake.tools import create_default_registry, runtime_tool_definitions

TEST_AGENT_SPEC = AgentSpec(
    spec_id="test",
    model_name="scripted",
    instructions="Follow the scripted test turns.",
    allowed_tools=frozenset(
        {
            "not_registered",
            *(definition.name for definition in runtime_tool_definitions()),
            *(
                definition.name
                for definition in create_default_registry(
                    lambda _cwd, _argv: True
                ).definitions()
            ),
        }
    ),
)


def fresh_store(path: Path, session: Session) -> SQLiteStateStore:
    store = SQLiteStateStore(path / f"{session.session_id}.db")
    store.create_session(session)
    return store


def history(store: RunStore, session: Session) -> list[ModelMessage]:
    return [
        item.message
        for item in store.read_messages(session.context_scope.scope_id)
    ]


def register_run(
    store: RunStore,
    session: Session,
    run: Run,
    messages: tuple[ModelMessage, ...] = (),
) -> None:
    store.create_run(session, run, messages)


def ingest(
    manager: AgentContextManager,
    session: Session,
    messages: list[ModelMessage] | tuple[ModelMessage, ...],
    *,
    start: int = 0,
) -> None:
    manager.ingest_messages(
        session.context_scope,
        tuple(
            SequencedMessage(start + index, message)
            for index, message in enumerate(messages)
        ),
    )
