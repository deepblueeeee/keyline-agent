"""SQLite-backed durable sessions, conversations, and run snapshots."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from handshake.models.protocol import ModelMessage
from handshake.runtime.errors import (
    ConcurrentRunUpdateError,
    ConcurrentSessionUpdateError,
    RunNotFoundError,
    SessionNotFoundError,
    StorageError,
)
from handshake.runtime.state import (
    Run,
    RunStatus,
    Session,
    WaitCondition,
    WaitKind,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
    utc_now,
)
from handshake.storage.schema import initialize_schema


class SQLiteStateStore:
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
            if session.messages:
                raise StorageError("A new session cannot contain messages before its first run")
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (
                        session.session_id,
                        str(workspace),
                        session.version,
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

    def create_run(self, session: Session, run: Run) -> None:
        """Atomically append a run and the session transcript containing its prompt."""
        try:
            updated_at = utc_now()
            expected_session_version = session.version
            next_session_version = expected_session_version + 1
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
                    raise ConcurrentSessionUpdateError(
                        f"Session {session.session_id} changed before run {run.run_id} "
                        "could be appended"
                    )
                self._insert_run(connection, run, updated_at)
                self._compare_and_update_session(
                    connection,
                    session,
                    expected_version=expected_session_version,
                    next_version=next_session_version,
                    updated_at=updated_at,
                )
                self._replace_session_messages(connection, session)
                self._replace_run_children(connection, run)
            session.version = next_session_version
            session.updated_at = updated_at
            run.updated_at = updated_at
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def checkpoint_run(self, session: Session, run: Run) -> None:
        """Atomically checkpoint one run and the conversation it has extended."""
        try:
            expected_session_version = session.version
            expected_run_version = run.version
            next_session_version = expected_session_version + 1
            next_run_version = expected_run_version + 1
            updated_at = utc_now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_run_membership(connection, session, run)
                self._compare_and_update_session(
                    connection,
                    session,
                    expected_version=expected_session_version,
                    next_version=next_session_version,
                    updated_at=updated_at,
                )
                self._compare_and_update_run(
                    connection,
                    run,
                    expected_version=expected_run_version,
                    next_version=next_run_version,
                    updated_at=updated_at,
                )
                self._replace_session_messages(connection, session)
                self._replace_run_children(connection, run)
            session.version = next_session_version
            session.updated_at = updated_at
            run.version = next_run_version
            run.updated_at = updated_at
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
            messages = [
                ModelMessage(
                    role=message_row["role"],
                    content=tuple(
                        _load_json(message_row["content_json"], "message content")
                    ),
                    run_id=str(message_row["run_id"]),
                )
                for message_row in connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                )
            ]
            return Session(
                session_id=str(row["session_id"]),
                workspace=Path(row["workspace"]),
                messages=messages,
                version=int(row["version"]),
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
                    tool_call_id=str(item_row["tool_call_id"]),
                    status=WorkItemStatus(item_row["status"]),
                )
                for item_row in connection.execute(
                    "SELECT * FROM work_items WHERE run_id = ? ORDER BY position",
                    (run_id,),
                )
            ]
            waits = [
                WaitCondition(
                    wait_id=str(wait_row["wait_id"]),
                    work_id=str(wait_row["work_id"]),
                    tool_call_id=str(wait_row["tool_call_id"]),
                    prompt=str(wait_row["prompt"]),
                    kind=WaitKind(wait_row["kind"]),
                    status=WaitStatus(wait_row["status"]),
                    response_text=wait_row["response_text"],
                    created_at=str(wait_row["created_at"]),
                    resolved_at=wait_row["resolved_at"],
                )
                for wait_row in connection.execute(
                    "SELECT * FROM wait_conditions WHERE run_id = ? ORDER BY position",
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
                waits=waits,
                final_summary=row["final_summary"],
                error=row["error"],
                model_name=str(row["model_name"]),
                model_turns=int(row["model_turns"]),
                max_steps=int(row["max_steps"]),
                version=int(row["version"]),
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
                max_steps, version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                run.version,
                run.created_at,
                updated_at,
            ),
        )

    def _compare_and_update_session(
        self,
        connection: sqlite3.Connection,
        session: Session,
        *,
        expected_version: int,
        next_version: int,
        updated_at: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE sessions SET version = ?, updated_at = ?
            WHERE session_id = ? AND version = ?
            """,
            (next_version, updated_at, session.session_id, expected_version),
        )
        if cursor.rowcount != 1:
            row = connection.execute(
                "SELECT version FROM sessions WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError(f"Session not found: {session.session_id}")
            raise ConcurrentSessionUpdateError(
                f"Session {session.session_id} changed since it was loaded "
                f"(expected version {expected_version}, stored version {int(row['version'])})"
            )

    def _compare_and_update_run(
        self,
        connection: sqlite3.Connection,
        run: Run,
        *,
        expected_version: int,
        next_version: int,
        updated_at: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE runs SET
                status = ?, plan_json = ?, final_summary = ?,
                error = ?, model_name = ?, model_turns = ?, max_steps = ?,
                version = ?, created_at = ?, updated_at = ?
            WHERE run_id = ? AND version = ?
            """,
            (
                run.status.value,
                _dump_json(run.plan),
                run.final_summary,
                run.error,
                run.model_name,
                run.model_turns,
                run.max_steps,
                next_version,
                run.created_at,
                updated_at,
                run.run_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            row = connection.execute(
                "SELECT version FROM runs WHERE run_id = ?", (run.run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(f"Run not found: {run.run_id}")
            raise ConcurrentRunUpdateError(
                f"Run {run.run_id} changed since it was loaded "
                f"(expected version {expected_version}, stored version {int(row['version'])})"
            )

    def _replace_session_messages(
        self, connection: sqlite3.Connection, session: Session
    ) -> None:
        connection.execute(
            "DELETE FROM messages WHERE session_id = ?", (session.session_id,)
        )
        connection.executemany(
            """
            INSERT INTO messages (session_id, sequence, run_id, role, content_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    session.session_id,
                    index,
                    message.run_id,
                    message.role,
                    _dump_json(list(message.content)),
                )
                for index, message in enumerate(session.messages)
            ),
        )

    def _replace_run_children(
        self, connection: sqlite3.Connection, run: Run
    ) -> None:
        connection.execute("DELETE FROM wait_conditions WHERE run_id = ?", (run.run_id,))
        connection.execute("DELETE FROM work_items WHERE run_id = ?", (run.run_id,))
        connection.executemany(
            """
            INSERT INTO work_items (
                work_id, run_id, position, description, tool_call_id, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    item.work_id,
                    run.run_id,
                    index,
                    item.description,
                    item.tool_call_id,
                    item.status.value,
                )
                for index, item in enumerate(run.work_items)
            ),
        )
        connection.executemany(
            """
            INSERT INTO wait_conditions (
                wait_id, run_id, work_id, position, tool_call_id, kind, prompt,
                status, response_text, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    wait.wait_id,
                    run.run_id,
                    wait.work_id,
                    index,
                    wait.tool_call_id,
                    wait.kind.value,
                    wait.prompt,
                    wait.status.value,
                    wait.response_text,
                    wait.created_at,
                    wait.resolved_at,
                )
                for index, wait in enumerate(run.waits)
            ),
        )


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: str, description: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise StorageError(f"Stored {description} is not valid JSON") from None


def _storage_error(exc: Exception) -> StorageError:
    return StorageError(f"SQLite state storage failed ({type(exc).__name__})")
