"""Current pre-release SQLite schema."""

from __future__ import annotations

import sqlite3

from handshake.storage.errors import StorageError

SCHEMA_VERSION = 16

_WORKTREES = """
CREATE TABLE worktrees (
    owner_run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE RESTRICT,
    record_json TEXT NOT NULL
);
"""

_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE context_scopes (
    scope_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    tail_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
    UNIQUE (scope_id, session_id)
);

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL,
    previous_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
    parent_run_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
    delegation_tool_call_id TEXT,
    agent_spec_id TEXT NOT NULL CHECK (length(trim(agent_spec_id)) > 0),
    context_scope_id TEXT NOT NULL,
    workspace_owner_id TEXT REFERENCES runs(run_id) ON DELETE RESTRICT,
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
    UNIQUE (parent_run_id, delegation_tool_call_id),
    UNIQUE (run_id, session_id),
    FOREIGN KEY (context_scope_id, session_id)
        REFERENCES context_scopes(scope_id, session_id) ON DELETE RESTRICT
);

CREATE TABLE messages (
    scope_id TEXT NOT NULL REFERENCES context_scopes(scope_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    origin_run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    content_json TEXT NOT NULL,
    PRIMARY KEY (scope_id, sequence)
);

CREATE TABLE tool_runs (
    tool_run_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    model_turn INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    replay_policy TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    UNIQUE (run_id, model_turn, tool_call_id)
);

CREATE TABLE context_checkpoints (
    scope_id TEXT PRIMARY KEY REFERENCES context_scopes(scope_id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL
);

CREATE TABLE dependencies (
    dependency_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
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
CREATE INDEX tool_runs_run_position ON tool_runs(run_id, position);
CREATE INDEX dependencies_run_status ON dependencies(run_id, status);
CREATE INDEX secret_handles_run_status ON secret_handles(run_id, status);
""" + _WORKTREES


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create a fresh database or reject an incompatible development schema."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version == 0:
        connection.executescript(_SCHEMA)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if version == 15:
        connection.executescript(f"""
            BEGIN IMMEDIATE;
            ALTER TABLE runs ADD COLUMN workspace_owner_id TEXT
                REFERENCES runs(run_id) ON DELETE RESTRICT;
            {_WORKTREES}
            PRAGMA user_version = {SCHEMA_VERSION};
            COMMIT;
        """)
        return
    if version != SCHEMA_VERSION:
        raise StorageError(
            f"Database schema {version} is incompatible with current development "
            f"schema {SCHEMA_VERSION}; start with a new database"
        )
