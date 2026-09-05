"""Current pre-release SQLite schema."""

from __future__ import annotations

import sqlite3

from handshake.storage.errors import StorageError

SCHEMA_VERSION = 8

_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL,
    previous_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    final_summary TEXT,
    error TEXT,
    model_name TEXT NOT NULL,
    model_turns INTEGER NOT NULL,
    max_steps INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (session_id, sequence),
    UNIQUE (previous_run_id),
    UNIQUE (run_id, session_id)
);

CREATE TABLE messages (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence),
    FOREIGN KEY (run_id, session_id) REFERENCES runs(run_id, session_id)
        ON DELETE CASCADE
);

CREATE TABLE work_items (
    work_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE dependencies (
    dependency_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE secret_handles (
    secret_handle TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    dependency_id TEXT NOT NULL UNIQUE
        REFERENCES dependencies(dependency_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    purpose TEXT NOT NULL,
    consumer TEXT NOT NULL,
    environment_variable TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX runs_session_sequence ON runs(session_id, sequence);
CREATE INDEX messages_run ON messages(run_id, sequence);
CREATE INDEX dependencies_run_status ON dependencies(run_id, status);
CREATE INDEX secret_handles_run_status ON secret_handles(run_id, status);
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
