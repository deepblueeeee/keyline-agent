"""SQLite-backed durable task snapshots for the Week 2 runtime."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from handshake.runtime.errors import (
    ConcurrentTaskUpdateError,
    StorageError,
    TaskNotFoundError,
)
from handshake.runtime.state import (
    ModelMessage,
    Task,
    TaskStatus,
    WaitCondition,
    WaitKind,
    WaitStatus,
    WorkItem,
    WorkItemStatus,
    utc_now,
)

_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    workspace TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    final_summary TEXT,
    error TEXT,
    model_name TEXT NOT NULL,
    model_turns INTEGER NOT NULL,
    max_steps INTEGER NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE messages (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,
    PRIMARY KEY (task_id, sequence)
);

CREATE TABLE work_items (
    work_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    description TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE wait_conditions (
    wait_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    response_text TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX wait_conditions_task_status
ON wait_conditions(task_id, status);
"""


class SQLiteTaskStore:
    """Persist complete stable task snapshots in a single local database."""

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
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise StorageError(
                    f"Task database schema {version} is newer than supported "
                    f"schema {_SCHEMA_VERSION}"
                )
            if version == 0:
                connection.executescript(_SCHEMA)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                version = _SCHEMA_VERSION
            if version < 2:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
                )
                connection.execute("PRAGMA user_version = 2")

    def create_task(self, task: Task) -> None:
        try:
            updated_at = utc_now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._write_task(
                    connection,
                    task,
                    create=True,
                    updated_at=updated_at,
                    expected_version=task.version,
                    next_version=task.version,
                )
            task.updated_at = updated_at
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def checkpoint_task(self, task: Task) -> None:
        try:
            expected_version = task.version
            next_version = expected_version + 1
            updated_at = utc_now()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._write_task(
                    connection,
                    task,
                    create=False,
                    updated_at=updated_at,
                    expected_version=expected_version,
                    next_version=next_version,
                )
            task.version = next_version
            task.updated_at = updated_at
        except (OSError, TypeError, ValueError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def load_task(self, task_id: str) -> Task:
        try:
            with self._connect() as connection:
                return self._load_task(connection, task_id)
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def list_tasks(self) -> tuple[Task, ...]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT task_id FROM tasks ORDER BY updated_at DESC, task_id"
                ).fetchall()
                return tuple(self._load_task(connection, str(row["task_id"])) for row in rows)
        except (OSError, sqlite3.Error) as exc:
            raise _storage_error(exc) from None

    def _write_task(
        self,
        connection: sqlite3.Connection,
        task: Task,
        *,
        create: bool,
        updated_at: str,
        expected_version: int,
        next_version: int,
    ) -> None:
        values = (
            task.task_id,
            task.goal,
            str(task.workspace),
            task.status.value,
            _dump_json(task.plan),
            task.final_summary,
            task.error,
            task.model_name,
            task.model_turns,
            task.max_steps,
            next_version,
            task.created_at,
            updated_at,
        )
        if create:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, goal, workspace, status, plan_json, final_summary, error,
                    model_name, model_turns, max_steps, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        else:
            cursor = connection.execute(
                """
                UPDATE tasks SET
                    goal = ?, workspace = ?, status = ?, plan_json = ?,
                    final_summary = ?, error = ?, model_name = ?, model_turns = ?,
                    max_steps = ?, version = ?, created_at = ?, updated_at = ?
                WHERE task_id = ? AND version = ?
                """,
                (*values[1:], values[0], expected_version),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT version FROM tasks WHERE task_id = ?", (task.task_id,)
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(f"Task not found: {task.task_id}")
                raise ConcurrentTaskUpdateError(
                    f"Task {task.task_id} changed since it was loaded "
                    f"(expected version {expected_version}, "
                    f"stored version {int(row['version'])})"
                )

        connection.execute("DELETE FROM wait_conditions WHERE task_id = ?", (task.task_id,))
        connection.execute("DELETE FROM work_items WHERE task_id = ?", (task.task_id,))
        connection.execute("DELETE FROM messages WHERE task_id = ?", (task.task_id,))
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (
                (task.task_id, index, message.role, _dump_json(list(message.content)))
                for index, message in enumerate(task.messages)
            ),
        )
        connection.executemany(
            """
            INSERT INTO work_items (
                work_id, task_id, position, description, tool_call_id, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    item.work_id,
                    task.task_id,
                    index,
                    item.description,
                    item.tool_call_id,
                    item.status.value,
                )
                for index, item in enumerate(task.work_items)
            ),
        )
        connection.executemany(
            """
            INSERT INTO wait_conditions (
                wait_id, task_id, work_id, position, tool_call_id, kind, prompt,
                status, response_text, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    wait.wait_id,
                    task.task_id,
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
                for index, wait in enumerate(task.waits)
            ),
        )

    def _load_task(self, connection: sqlite3.Connection, task_id: str) -> Task:
        row = connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise TaskNotFoundError(f"Task not found: {task_id}")
        try:
            messages = [
                ModelMessage(
                    role=message_row["role"],
                    content=tuple(_load_json(message_row["content_json"], "message content")),
                )
                for message_row in connection.execute(
                    "SELECT * FROM messages WHERE task_id = ? ORDER BY sequence",
                    (task_id,),
                )
            ]
            work_items = [
                WorkItem(
                    work_id=str(item_row["work_id"]),
                    description=str(item_row["description"]),
                    tool_call_id=str(item_row["tool_call_id"]),
                    status=WorkItemStatus(item_row["status"]),
                )
                for item_row in connection.execute(
                    "SELECT * FROM work_items WHERE task_id = ? ORDER BY position",
                    (task_id,),
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
                    "SELECT * FROM wait_conditions WHERE task_id = ? ORDER BY position",
                    (task_id,),
                )
            ]
            plan = _load_json(row["plan_json"], "task plan")
            if not isinstance(plan, list) or any(not isinstance(item, str) for item in plan):
                raise StorageError(f"Task {task_id} has an invalid stored plan")
            return Task(
                task_id=str(row["task_id"]),
                goal=str(row["goal"]),
                workspace=Path(row["workspace"]),
                status=TaskStatus(row["status"]),
                plan=plan,
                messages=messages,
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
                f"Task {task_id} contains invalid persisted state ({type(exc).__name__})"
            ) from None


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: str, description: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise StorageError(f"Stored {description} is not valid JSON") from None


def _storage_error(exc: Exception) -> StorageError:
    return StorageError(f"SQLite task storage failed ({type(exc).__name__})")
