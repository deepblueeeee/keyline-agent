"""SQLite-backed durable sessions, conversations, and run snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from handshake.context.state import (
    ContextCheckpoint,
    HistoryEpisode,
    SequencedMessage,
    SourceRange,
)
from handshake.models.protocol import ModelMessage
from handshake.runtime.persistence import RunStore
from handshake.runtime.state import (
    Dependency,
    DependencyKind,
    DependencyStatus,
    Run,
    RunStatus,
    SecretHandle,
    SecretHandleStatus,
    Session,
    WorkItem,
    WorkItemStatus,
    utc_now,
)
from handshake.storage.errors import RunNotFoundError, SessionNotFoundError, StorageError
from handshake.storage.schema import initialize_schema

_CONTEXT_CHECKPOINT_FORMAT = {
    "version": 1,
    "token_estimator": {"name": "utf8-bytes", "version": 1},
}


class SQLiteStateStore(RunStore):
    """Persist session transcripts and complete stable run snapshots."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            initialize_schema(connection)

    def create_session(self, session: Session) -> None:
        try:
            workspace = session.workspace.expanduser().resolve()
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO sessions (
                        session_id, workspace, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        str(workspace),
                        session.created_at,
                        session.updated_at,
                    ),
                )
            session.workspace = workspace
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def load_session(self, session_id: str) -> Session:
        try:
            with self._connect() as connection:
                return self._load_session(connection, session_id)
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def list_sessions(self) -> tuple[Session, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT sessions.session_id
                    FROM sessions
                    LEFT JOIN runs USING (session_id)
                    GROUP BY sessions.session_id
                    ORDER BY COALESCE(MAX(runs.updated_at), sessions.updated_at) DESC,
                             sessions.session_id
                    """
                ).fetchall()
                return tuple(
                    self._load_session(connection, str(row["session_id"]))
                    for row in rows
                )
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def create_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Atomically append a run and the session transcript containing its prompt."""
        try:
            updated_at = utc_now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_session_workspace(connection, session)
                tail = connection.execute(
                    """
                    SELECT run_id, sequence FROM runs
                    WHERE session_id = ? ORDER BY sequence DESC LIMIT 1
                    """,
                    (session.session_id,),
                ).fetchone()
                tail_id = None if tail is None else str(tail["run_id"])
                next_sequence = 1 if tail is None else int(tail["sequence"]) + 1
                if tail_id != run.previous_run_id or run.sequence != next_sequence:
                    raise StorageError(
                        f"Run {run.run_id} is not the next run for session "
                        f"{session.session_id}"
                    )
                self._insert_run(connection, run, updated_at)
                self._update_session(connection, session, updated_at=updated_at)
                appended = self._append_messages(
                    connection, session.session_id, messages
                )
                self._replace_run_children(connection, run)
            session.updated_at = updated_at
            run.updated_at = updated_at
            return appended
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def checkpoint_run(
        self,
        session: Session,
        run: Run,
        messages: tuple[ModelMessage, ...] = (),
    ) -> tuple[SequencedMessage, ...]:
        """Atomically checkpoint one run and the conversation it has extended."""
        try:
            updated_at = utc_now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_run_membership(connection, session, run)
                self._update_session(connection, session, updated_at=updated_at)
                self._update_run(connection, run, updated_at=updated_at)
                appended = self._append_messages(
                    connection, session.session_id, messages
                )
                self._replace_run_children(connection, run)
            session.updated_at = updated_at
            run.updated_at = updated_at
            return appended
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def load_run(self, run_id: str) -> Run:
        try:
            with self._connect() as connection:
                return self._load_run(connection, run_id)
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def list_runs(self, session_id: str | None = None) -> tuple[Run, ...]:
        try:
            with self._connect() as connection:
                if session_id is None:
                    rows = connection.execute(
                        "SELECT run_id FROM runs ORDER BY updated_at DESC, run_id"
                    ).fetchall()
                else:
                    if connection.execute(
                        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                    ).fetchone() is None:
                        raise SessionNotFoundError(f"Session not found: {session_id}")
                    rows = connection.execute(
                        """
                        SELECT run_id FROM runs
                        WHERE session_id = ? ORDER BY sequence
                        """,
                        (session_id,),
                    ).fetchall()
                return tuple(self._load_run(connection, str(row["run_id"])) for row in rows)
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def _load_session(
        self, connection: sqlite3.Connection, session_id: str
    ) -> Session:
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        try:
            return Session(
                session_id=str(row["session_id"]),
                workspace=Path(row["workspace"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Session {session_id} contains invalid persisted state "
                f"({type(exc).__name__})"
            ) from None

    def _load_run(self, connection: sqlite3.Connection, run_id: str) -> Run:
        row = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run not found: {run_id}")
        try:
            work_items = [
                WorkItem(
                    work_id=str(item_row["work_id"]),
                    description=str(item_row["description"]),
                    status=WorkItemStatus(item_row["status"]),
                )
                for item_row in connection.execute(
                    "SELECT * FROM work_items WHERE run_id = ? ORDER BY position",
                    (run_id,),
                )
            ]
            dependencies = [
                Dependency(
                    dependency_id=str(dependency_row["dependency_id"]),
                    work_id=str(dependency_row["work_id"]),
                    tool_call_id=str(dependency_row["tool_call_id"]),
                    prompt=str(dependency_row["prompt"]),
                    kind=DependencyKind(dependency_row["kind"]),
                    status=DependencyStatus(dependency_row["status"]),
                    created_at=str(dependency_row["created_at"]),
                    resolved_at=dependency_row["resolved_at"],
                )
                for dependency_row in connection.execute(
                    "SELECT * FROM dependencies WHERE run_id = ? ORDER BY position",
                    (run_id,),
                )
            ]
            secret_handles = [
                SecretHandle(
                    secret_handle=str(handle_row["secret_handle"]),
                    dependency_id=str(handle_row["dependency_id"]),
                    purpose=str(handle_row["purpose"]),
                    consumer=str(handle_row["consumer"]),
                    environment_variable=str(handle_row["environment_variable"]),
                    status=SecretHandleStatus(handle_row["status"]),
                    created_at=str(handle_row["created_at"]),
                    updated_at=str(handle_row["updated_at"]),
                )
                for handle_row in connection.execute(
                    "SELECT * FROM secret_handles WHERE run_id = ? ORDER BY position",
                    (run_id,),
                )
            ]
            plan = _load_json(row["plan_json"], "run plan")
            if not isinstance(plan, list) or any(not isinstance(item, str) for item in plan):
                raise StorageError(f"Run {run_id} has an invalid stored plan")
            return Run(
                run_id=str(row["run_id"]),
                session_id=str(row["session_id"]),
                sequence=int(row["sequence"]),
                previous_run_id=row["previous_run_id"],
                goal=str(row["goal"]),
                status=RunStatus(row["status"]),
                plan=plan,
                work_items=work_items,
                dependencies=dependencies,
                secret_handles=secret_handles,
                final_summary=row["final_summary"],
                error=row["error"],
                model_name=str(row["model_name"]),
                model_turns=int(row["model_turns"]),
                max_steps=int(row["max_steps"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StorageError(
                f"Run {run_id} contains invalid persisted state ({type(exc).__name__})"
            ) from None

    def _validate_session_workspace(
        self, connection: sqlite3.Connection, session: Session
    ) -> None:
        row = connection.execute(
            "SELECT workspace FROM sessions WHERE session_id = ?",
            (session.session_id,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session not found: {session.session_id}")
        if Path(row["workspace"]) != session.workspace:
            raise StorageError(f"Session {session.session_id} has an invalid workspace")

    def _validate_run_membership(
        self, connection: sqlite3.Connection, session: Session, run: Run
    ) -> None:
        self._validate_session_workspace(connection, session)
        row = connection.execute(
            """
            SELECT session_id, sequence, previous_run_id, goal
            FROM runs WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run not found: {run.run_id}")
        membership = (
            str(row["session_id"]),
            int(row["sequence"]),
            row["previous_run_id"],
            str(row["goal"]),
        )
        expected = (run.session_id, run.sequence, run.previous_run_id, run.goal)
        if membership != expected or run.session_id != session.session_id:
            raise StorageError(f"Run {run.run_id} has invalid session membership")

    def _insert_run(
        self, connection: sqlite3.Connection, run: Run, updated_at: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, session_id, sequence, previous_run_id, goal, status,
                plan_json, final_summary, error, model_name, model_turns,
                max_steps, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.session_id,
                run.sequence,
                run.previous_run_id,
                run.goal,
                run.status.value,
                _dump_json(run.plan),
                run.final_summary,
                run.error,
                run.model_name,
                run.model_turns,
                run.max_steps,
                run.created_at,
                updated_at,
            ),
        )

    def _update_session(
        self,
        connection: sqlite3.Connection,
        session: Session,
        *,
        updated_at: str,
    ) -> None:
        cursor = connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (updated_at, session.session_id),
        )
        if cursor.rowcount != 1:
            raise SessionNotFoundError(f"Session not found: {session.session_id}")

    def _update_run(
        self,
        connection: sqlite3.Connection,
        run: Run,
        *,
        updated_at: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE runs SET
                status = ?, plan_json = ?, final_summary = ?,
                error = ?, model_name = ?, model_turns = ?, max_steps = ?,
                updated_at = ?
            WHERE run_id = ?
            """,
            (
                run.status.value,
                _dump_json(run.plan),
                run.final_summary,
                run.error,
                run.model_name,
                run.model_turns,
                run.max_steps,
                updated_at,
                run.run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RunNotFoundError(f"Run not found: {run.run_id}")

    @staticmethod
    def _message_count(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COUNT(*), MAX(sequence) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        count = int(row[0])
        if count and int(row[1]) != count - 1:
            raise StorageError("Message sequences are not contiguous")
        return count

    def _append_messages(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        messages: tuple[ModelMessage, ...],
    ) -> tuple[SequencedMessage, ...]:
        start = self._message_count(connection, session_id)
        appended = tuple(
            SequencedMessage(start + index, message)
            for index, message in enumerate(messages)
        )
        connection.executemany(
            """
            INSERT INTO messages (session_id, sequence, run_id, role, content_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    session_id,
                    start + index,
                    message.run_id,
                    message.role,
                    _dump_json(list(message.content)),
                )
                for index, message in enumerate(messages)
            ),
        )
        return appended

    def append_messages(
        self, session_id: str, messages: tuple[ModelMessage, ...]
    ) -> tuple[SequencedMessage, ...]:
        """Append a batch without replacing any existing transcript rows."""
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone() is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                return self._append_messages(connection, session_id, messages)
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def read_messages(
        self, session_id: str, start: int = 0, end: int | None = None
    ) -> Iterator[SequencedMessage]:
        """Stream decoded messages in the half-open range [start, end)."""
        if start < 0 or (end is not None and end < start):
            raise StorageError("Invalid history range")
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone() is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                rows = connection.execute(
                    """
                    SELECT sequence, run_id, role, content_json
                    FROM messages
                    WHERE session_id = ? AND sequence >= ?
                      AND (? IS NULL OR sequence < ?)
                    ORDER BY sequence
                    """,
                    (session_id, start, end, end),
                )
                for row in rows:
                    yield SequencedMessage(
                        sequence=int(row["sequence"]),
                        message=ModelMessage(
                            role=cast(Literal["user", "assistant"], str(row["role"])),
                            content=tuple(
                                _load_json(row["content_json"], "message content")
                            ),
                            run_id=str(row["run_id"]),
                        ),
                    )
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def message_count(self, session_id: str, run_id: str | None = None) -> int:
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone() is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                if run_id is None:
                    return self._message_count(connection, session_id)
                return int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM messages
                        WHERE session_id = ? AND run_id = ?
                        """,
                        (session_id, run_id),
                    ).fetchone()[0]
                )
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def has_run_messages(self, session_id: str, run_id: str) -> bool:
        return self.message_count(session_id, run_id) > 0

    def load_context_checkpoint(self, session_id: str) -> ContextCheckpoint | None:
        """Load view metadata without loading any raw transcript rows."""
        try:
            with self._connect() as connection:
                if connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone() is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                row = connection.execute(
                    "SELECT revision, snapshot_json FROM context_checkpoints WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    return None
                return _load_context_checkpoint(
                    str(row["snapshot_json"]), int(row["revision"])
                )
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def save_context_checkpoint(
        self, session_id: str, checkpoint: ContextCheckpoint
    ) -> ContextCheckpoint:
        """Save one projection recipe with tail and revision checks."""
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone() is None:
                    raise SessionNotFoundError(f"Session not found: {session_id}")
                count = self._message_count(connection, session_id)
                if count != checkpoint.source_message_count:
                    raise StorageError(
                        "Context checkpoint does not describe the current transcript tail"
                    )
                row = connection.execute(
                    "SELECT revision FROM context_checkpoints WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                revision = 0 if row is None else int(row["revision"])
                if checkpoint.revision != revision:
                    raise StorageError("Context checkpoint revision is stale")
                saved = replace(checkpoint, revision=revision + 1)
                connection.execute(
                    """
                    INSERT INTO context_checkpoints (session_id, revision, snapshot_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        revision = excluded.revision,
                        snapshot_json = excluded.snapshot_json
                    """,
                    (session_id, saved.revision, _dump_context_checkpoint(saved)),
                )
                return saved
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def _replace_run_children(
        self, connection: sqlite3.Connection, run: Run
    ) -> None:
        connection.execute("DELETE FROM secret_handles WHERE run_id = ?", (run.run_id,))
        connection.execute("DELETE FROM dependencies WHERE run_id = ?", (run.run_id,))
        connection.execute("DELETE FROM work_items WHERE run_id = ?", (run.run_id,))
        connection.executemany(
            """
            INSERT INTO work_items (
                work_id, run_id, position, description, status
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    item.work_id,
                    run.run_id,
                    index,
                    item.description,
                    item.status.value,
                )
                for index, item in enumerate(run.work_items)
            ),
        )
        connection.executemany(
            """
            INSERT INTO dependencies (
                dependency_id, run_id, work_id, position, tool_call_id, kind,
                prompt, status, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    dependency.dependency_id,
                    run.run_id,
                    dependency.work_id,
                    index,
                    dependency.tool_call_id,
                    dependency.kind.value,
                    dependency.prompt,
                    dependency.status.value,
                    dependency.created_at,
                    dependency.resolved_at,
                )
                for index, dependency in enumerate(run.dependencies)
            ),
        )
        connection.executemany(
            """
            INSERT INTO secret_handles (
                secret_handle, run_id, dependency_id, position, purpose,
                consumer, environment_variable, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    handle.secret_handle,
                    run.run_id,
                    handle.dependency_id,
                    index,
                    handle.purpose,
                    handle.consumer,
                    handle.environment_variable,
                    handle.status.value,
                    handle.created_at,
                    handle.updated_at,
                )
                for index, handle in enumerate(run.secret_handles)
            ),
        )


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dump_context_checkpoint(checkpoint: ContextCheckpoint) -> str:
    return _dump_json(
        {
            "format": _CONTEXT_CHECKPOINT_FORMAT,
            "state": {
                "source_message_count": checkpoint.source_message_count,
                "recent_start": checkpoint.recent_start,
                "episodes": [
                    {
                        "start_sequence": item.source_range.start_sequence,
                        "end_sequence": item.source_range.end_sequence,
                        "title": item.title,
                        "summary": item.summary,
                    }
                    for item in checkpoint.episodes
                ],
                "stale_mask_through": checkpoint.stale_mask_through,
                "max_tool_result_tokens": checkpoint.max_tool_result_tokens,
                "reduced_message_count": checkpoint.reduced_message_count,
            },
        }
    )


def _load_context_checkpoint(value: str, revision: int) -> ContextCheckpoint:
    data = _load_json(value, "context checkpoint")
    if not isinstance(data, dict):
        raise StorageError("Stored context checkpoint is not an object")
    if data.get("format") != _CONTEXT_CHECKPOINT_FORMAT:
        raise StorageError("Stored context checkpoint format is unsupported")
    state = data.get("state")
    if not isinstance(state, dict):
        raise StorageError("Stored context checkpoint state is not an object")
    episodes_data = state.get("episodes")
    if not isinstance(episodes_data, list):
        raise StorageError("Stored context checkpoint episodes are invalid")
    try:
        episodes = tuple(
            HistoryEpisode(
                SourceRange(item["start_sequence"], item["end_sequence"]),
                item["title"],
                item["summary"],
            )
            for item in episodes_data
        )
        cursor = state.get("stale_mask_through")
        stale_mask_through = None if cursor is None else tuple(cursor)
        return ContextCheckpoint(
            source_message_count=state["source_message_count"],
            recent_start=state["recent_start"],
            episodes=episodes,
            stale_mask_through=cast(tuple[int, int] | None, stale_mask_through),
            max_tool_result_tokens=state["max_tool_result_tokens"],
            reduced_message_count=state["reduced_message_count"],
            revision=revision,
        )
    except (KeyError, TypeError, ValueError):
        raise StorageError("Stored context checkpoint fields are invalid") from None


def _load_json(value: str, description: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise StorageError(f"Stored {description} is not valid JSON") from None


def _storage_error(exc: Exception) -> StorageError:
    return StorageError(f"SQLite state storage failed ({type(exc).__name__})")
