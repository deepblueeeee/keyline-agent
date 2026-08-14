"""Current pre-release SQLite schema."""

from __future__ import annotations

import sqlite3

from handshake.runtime.errors import StorageError

SCHEMA_VERSION = 4

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

CREATE INDEX tasks_session_sequence ON tasks(session_id, sequence);
CREATE INDEX wait_conditions_task_status ON wait_conditions(task_id, status);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create a fresh database or reject an incompatible development schema."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version == 0:
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if version != SCHEMA_VERSION:
        raise StorageError(
            f"Database schema {version} is incompatible with current development "
            f"schema {SCHEMA_VERSION}; start with a new database"
        )
