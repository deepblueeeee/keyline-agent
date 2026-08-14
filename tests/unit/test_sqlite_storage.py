import sqlite3
from pathlib import Path

import pytest

from handshake.runtime.errors import ConcurrentTaskUpdateError, TaskNotFoundError
from handshake.runtime.state import (
    ModelMessage,
    Session,
    Task,
    TaskStatus,
    WaitCondition,
    WorkItem,
)
from handshake.storage.sqlite import SQLiteStateStore


def _waiting_task(tmp_path: Path, session_id: str) -> Task:
    task = Task(
        goal="选择发布颜色",
        workspace=tmp_path,
        session_id=session_id,
        model_name="deepseek-v4-flash",
        max_steps=7,
        model_turns=2,
    )
    task.transition(TaskStatus.RUNNING)
    task.messages.extend(
        [
            ModelMessage(role="user", content=({"type": "text", "text": task.goal},)),
            ModelMessage(
                role="assistant",
                content=(
                    {"type": "thinking", "thinking": "需要用户决定"},
                    {
                        "type": "tool_use",
                        "id": "ask-1",
                        "name": "wait_for_text",
                        "input": {"prompt": "发布按钮使用什么颜色？"},
                    },
                ),
            ),
        ]
    )
    work = WorkItem(description="选择颜色", tool_call_id="ask-1")
    task.work_items.append(work)
    task.waits.append(
        WaitCondition(
            work_id=work.work_id,
            tool_call_id="ask-1",
            prompt="发布按钮使用什么颜色？",
        )
    )
    task.transition(TaskStatus.WAITING)
    return task


def test_migration_is_idempotent_and_task_round_trips(tmp_path: Path) -> None:
    database = tmp_path / "state" / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    task = _waiting_task(tmp_path, session.session_id)
    store.create_task(task)

    loaded = SQLiteStateStore(database).load_task(task.task_id)

    assert loaded.goal == "选择发布颜色"
    assert loaded.status is TaskStatus.WAITING
    assert loaded.model_turns == 2
    assert loaded.max_steps == 7
    assert loaded.messages[1].content[0]["thinking"] == "需要用户决定"
    assert loaded.waits[0].tool_call_id == "ask-1"
    assert loaded.session_id == session.session_id

    with pytest.raises(TaskNotFoundError):
        store.load_task("missing")


def test_v1_database_migrates_to_optimistic_lock_version(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tasks VALUES (
                'legacy-task', 'legacy goal', ?, 'completed', '[]', 'done', NULL,
                'legacy-model', 3, 10, '2026-01-01', '2026-01-02'
            )
            """,
            (str(tmp_path),),
        )
        connection.execute("PRAGMA user_version = 1")

    SQLiteStateStore(database)
    SQLiteStateStore(database)

    with sqlite3.connect(database) as connection:
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(tasks)")
        }
        session_count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        migrated = connection.execute(
            "SELECT session_id, sequence, previous_task_id, version FROM tasks"
        ).fetchone()
    assert schema_version == 3
    assert columns["version"][3] == 1
    assert session_count == 1
    assert migrated[0]
    assert migrated[1:] == (1, None, 0)


def test_v2_migration_preserves_task_children(tmp_path: Path) -> None:
    database = tmp_path / "handshake.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY, goal TEXT NOT NULL, workspace TEXT NOT NULL,
                status TEXT NOT NULL, plan_json TEXT NOT NULL, final_summary TEXT,
                error TEXT, model_name TEXT NOT NULL, model_turns INTEGER NOT NULL,
                max_steps INTEGER NOT NULL, version INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL, role TEXT NOT NULL, content_json TEXT NOT NULL,
                PRIMARY KEY (task_id, sequence)
            );
            CREATE TABLE work_items (
                work_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                position INTEGER NOT NULL, description TEXT NOT NULL,
                tool_call_id TEXT NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE wait_conditions (
                wait_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
                position INTEGER NOT NULL, tool_call_id TEXT NOT NULL, kind TEXT NOT NULL,
                prompt TEXT NOT NULL, status TEXT NOT NULL, response_text TEXT,
                created_at TEXT NOT NULL, resolved_at TEXT
            );
            CREATE INDEX wait_conditions_task_status
            ON wait_conditions(task_id, status);
            PRAGMA user_version = 2;
            """
        )
        connection.execute(
            """
            INSERT INTO tasks VALUES (
                'task-v2', 'goal', ?, 'waiting', '[]', NULL, NULL, 'model',
                2, 10, 4, '2026-01-01', '2026-01-02'
            )
            """,
            (str(tmp_path),),
        )
        connection.execute(
            "INSERT INTO messages VALUES ('task-v2', 0, 'assistant', ?)",
            ('[{"type":"thinking","thinking":"preserve me"}]',),
        )
        connection.execute(
            """
            INSERT INTO work_items VALUES (
                'work-v2', 'task-v2', 0, 'choose', 'call-v2', 'waiting'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO wait_conditions VALUES (
                'wait-v2', 'task-v2', 'work-v2', 0, 'call-v2', 'text',
                'Choose?', 'open', NULL, '2026-01-01', NULL
            )
            """
        )

    store = SQLiteStateStore(database)
    task = store.load_task("task-v2")

    assert task.sequence == 1
    assert task.version == 4
    assert task.messages[0].content[0]["thinking"] == "preserve me"
    assert task.work_items[0].work_id == "work-v2"
    assert task.waits[0].wait_id == "wait-v2"
    assert store.load_session(task.session_id).workspace == tmp_path


def test_environment_api_key_is_not_written_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary-key-that-must-not-be-persisted"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    database = tmp_path / "handshake.db"
    store = SQLiteStateStore(database)
    session = Session(workspace=tmp_path)
    store.create_session(session)
    store.create_task(_waiting_task(tmp_path, session.session_id))

    assert secret.encode() not in database.read_bytes()


def test_stale_checkpoint_is_rejected_without_overwriting_newer_state(
    tmp_path: Path,
) -> None:
    store = SQLiteStateStore(tmp_path / "handshake.db")
    session = Session(workspace=tmp_path)
    store.create_session(session)
    task = _waiting_task(tmp_path, session.session_id)
    store.create_task(task)
    first = store.load_task(task.task_id)
    stale = store.load_task(task.task_id)

    first.plan = ["newer plan"]
    first.messages.append(
        ModelMessage(role="user", content=({"type": "text", "text": "newer message"},))
    )
    store.checkpoint_task(first)
    stale.plan = ["stale plan"]
    stale.messages.append(
        ModelMessage(role="user", content=({"type": "text", "text": "stale message"},))
    )

    with pytest.raises(ConcurrentTaskUpdateError, match="changed since it was loaded"):
        store.checkpoint_task(stale)

    persisted = store.load_task(task.task_id)
    assert first.version == 1
    assert stale.version == 0
    assert persisted.version == 1
    assert persisted.plan == ["newer plan"]
    assert persisted.messages[-1].content[0]["text"] == "newer message"
