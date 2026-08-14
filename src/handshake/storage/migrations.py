"""SQLite schema creation and forward-only migrations."""

from __future__ import annotations

import sqlite3
from uuid import uuid4

from handshake.runtime.errors import StorageError

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL,
    previous_task_id TEXT REFERENCES tasks(task_id) ON DELETE RESTRICT,
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
    updated_at TEXT NOT NULL,
    UNIQUE (session_id, sequence),
    UNIQUE (previous_task_id)
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

CREATE INDEX tasks_session_sequence
ON tasks(session_id, sequence);
"""

_TASKS_V3_SCHEMA = """
CREATE TABLE tasks_v3 (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL,
    previous_task_id TEXT REFERENCES tasks_v3(task_id) ON DELETE RESTRICT,
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
    updated_at TEXT NOT NULL,
    UNIQUE (session_id, sequence),
    UNIQUE (previous_task_id)
);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the database to the supported schema version."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise StorageError(
            f"Task database schema {version} is newer than supported "
            f"schema {SCHEMA_VERSION}"
        )
    if version == 0:
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if version < 2:
        connection.execute(
            "ALTER TABLE tasks ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute("PRAGMA user_version = 2")
        version = 2
    if version < 3:
        _migrate_to_v3(connection)


def _migrate_to_v3(connection: sqlite3.Connection) -> None:
    task_rows = connection.execute(
        "SELECT task_id, workspace, created_at FROM tasks ORDER BY created_at, task_id"
    ).fetchall()
    session_ids = {str(row["task_id"]): str(uuid4()) for row in task_rows}

    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (
                (
                    session_ids[str(row["task_id"])],
                    str(row["workspace"]),
                    str(row["created_at"]),
                )
                for row in task_rows
            ),
        )
        connection.execute(_TASKS_V3_SCHEMA)
        for row in task_rows:
            task_id = str(row["task_id"])
            connection.execute(
                """
                INSERT INTO tasks_v3 (
                    task_id, session_id, sequence, previous_task_id, goal,
                    workspace, status, plan_json, final_summary, error,
                    model_name, model_turns, max_steps, version, created_at,
                    updated_at
                )
                SELECT task_id, ?, 1, NULL, goal, workspace, status, plan_json,
                       final_summary, error, model_name, model_turns, max_steps,
                       version, created_at, updated_at
                FROM tasks WHERE task_id = ?
                """,
                (session_ids[task_id], task_id),
            )
        connection.execute("DROP TABLE tasks")
        connection.execute("ALTER TABLE tasks_v3 RENAME TO tasks")
        connection.execute(
            "CREATE INDEX tasks_session_sequence ON tasks(session_id, sequence)"
        )
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise StorageError("Task database migration produced invalid foreign keys")
