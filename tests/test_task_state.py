from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from convsearch.cli.app import app
from convsearch.config.settings import Settings, database_path
from convsearch.memory.store import set_task_state
from convsearch.storage.database import connect, initialize_database
from convsearch.storage.migrations import migration_files
from convsearch.tasks.query import get_task

runner = CliRunner()


@pytest.fixture()
def db_conn() -> Generator[sqlite3.Connection, None, None]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        initialize_database(workspace)
        conn = connect(database_path(workspace))
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.close()


def _seed_message(conn: sqlite3.Connection) -> tuple[int, int]:
    conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (1, '/tmp/x.zip', 'h1', 'imported')"
    )
    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, content_hash)"
        " VALUES (1, 'src-conv-1', 1, 'My Project', 'ch1')"
    )
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " source_order, is_primary_path, text, content_hash)"
        " VALUES (1, 'msg-1', 1, 'assistant', 0, 1, 'text', 'mh1')"
    )
    conn.commit()
    return 1, 1


def _insert_memory(
    conn: sqlite3.Connection,
    *,
    memory_id: int,
    conv_id: int,
    msg_id: int,
    kind: str = "task",
    task_state: str | None = "open",
    status: str = "active",
) -> int:
    conn.execute(
        """
        INSERT INTO memories
          (memory_id, kind, subject_key, statement, status, confidence, project,
           task_state, conversation_id, message_id, extraction_version, content_hash,
           metadata_json)
        VALUES (?, ?, ?, ?, ?, 0.9, NULL, ?, ?, ?, 'v1', ?, '{}')
        """,
        (
            memory_id,
            kind,
            f"subj-{memory_id}",
            f"statement {memory_id}",
            status,
            task_state,
            conv_id,
            msg_id,
            f"hash-{memory_id}",
        ),
    )
    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) VALUES (?, ?, ?, '', ?)",
        (memory_id, f"statement {memory_id}", kind, status),
    )
    conn.commit()
    return memory_id


# --- migration ---


def test_migration_009_present() -> None:
    assert any(p.stem == "009_task_state_history" for p in migration_files())


def test_migration_idempotent_on_populated_db(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id)
    set_task_state(db_conn, 1, "completed", reason="done")
    db_conn.commit()

    workspace = Path(db_conn.execute("PRAGMA database_list").fetchone()["file"]).parent.parent
    initialize_database(workspace)  # second call: must be a no-op, not raise

    row = db_conn.execute(
        "SELECT task_state, task_state_changed_at FROM memories WHERE memory_id = 1"
    ).fetchone()
    assert row["task_state"] == "completed"
    assert row["task_state_changed_at"] is not None
    history = db_conn.execute("SELECT COUNT(*) AS n FROM task_state_history").fetchone()
    assert history["n"] == 1


# --- set_task_state ---


def test_set_task_state_completes_and_stamps_changed_at(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="open")

    set_task_state(db_conn, 1, "completed", reason="shipped")
    db_conn.commit()

    row = db_conn.execute(
        "SELECT task_state, task_state_changed_at FROM memories WHERE memory_id = 1"
    ).fetchone()
    assert row["task_state"] == "completed"
    assert row["task_state_changed_at"] is not None

    history = db_conn.execute(
        "SELECT old_state, new_state, reason FROM task_state_history WHERE memory_id = 1"
    ).fetchone()
    assert history["old_state"] == "open"
    assert history["new_state"] == "completed"
    assert history["reason"] == "shipped"


def test_set_task_state_reopen_round_trip(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="completed")

    set_task_state(db_conn, 1, "open")
    db_conn.commit()

    row = db_conn.execute("SELECT task_state FROM memories WHERE memory_id = 1").fetchone()
    assert row["task_state"] == "open"
    count = db_conn.execute(
        "SELECT COUNT(*) AS n FROM task_state_history WHERE memory_id = 1"
    ).fetchone()["n"]
    assert count == 1


def test_set_task_state_invalid_state_raises(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id)
    with pytest.raises(ValueError, match="Invalid task state"):
        set_task_state(db_conn, 1, "done")


def test_set_task_state_unknown_memory_raises(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="999"):
        set_task_state(db_conn, 999, "completed")


def test_set_task_state_non_task_memory_raises(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, kind="decision", task_state=None
    )
    with pytest.raises(ValueError, match="not a task"):
        set_task_state(db_conn, 1, "completed")


def test_set_task_state_noop_writes_no_history(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="open")

    set_task_state(db_conn, 1, "open")  # already open
    db_conn.commit()

    row = db_conn.execute(
        "SELECT task_state_changed_at FROM memories WHERE memory_id = 1"
    ).fetchone()
    assert row["task_state_changed_at"] is None
    count = db_conn.execute(
        "SELECT COUNT(*) AS n FROM task_state_history WHERE memory_id = 1"
    ).fetchone()["n"]
    assert count == 0


def test_set_task_state_from_null_records_null_old_state(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state=None)

    set_task_state(db_conn, 1, "open")
    db_conn.commit()

    history = db_conn.execute(
        "SELECT old_state, new_state FROM task_state_history WHERE memory_id = 1"
    ).fetchone()
    assert history["old_state"] is None
    assert history["new_state"] == "open"


# --- get_task ---


def test_get_task_reflects_transition(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="open")

    set_task_state(db_conn, 1, "completed", reason="done")
    db_conn.commit()

    item = get_task(db_conn, 1)
    assert item is not None
    assert item.task_state == "completed"
    assert item.task_state_source == "user"
    assert item.task_state_changed_at is not None


def test_get_task_heuristic_only_has_no_user_source(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="completed")

    item = get_task(db_conn, 1)
    assert item is not None
    assert item.task_state_source == "heuristic"
    assert item.task_state_changed_at is None


def test_get_task_not_a_task_returns_none(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, kind="decision", task_state=None
    )
    assert get_task(db_conn, 1) is None


def test_get_task_unknown_id_returns_none(db_conn: sqlite3.Connection) -> None:
    assert get_task(db_conn, 999) is None


# --- CLI ---


def test_cli_tasks_complete_happy_path(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    Settings().write(workspace)
    initialize_database(workspace)
    conn = connect(database_path(workspace))
    conv_id, msg_id = _seed_message(conn)
    _insert_memory(conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, task_state="open")
    conn.close()

    result = runner.invoke(
        app, ["tasks", "complete", "1", "-w", str(workspace), "--reason", "shipped it"]
    )
    assert result.exit_code == 0, result.output
    assert "open -> completed" in result.output

    result_list = runner.invoke(
        app, ["tasks", "list", "-w", str(workspace), "--state", "completed"]
    )
    assert result_list.exit_code == 0
    assert "statement 1" in result_list.output

    result_reopen = runner.invoke(app, ["tasks", "reopen", "1", "-w", str(workspace)])
    assert result_reopen.exit_code == 0, result_reopen.output
    assert "completed -> open" in result_reopen.output


def test_cli_tasks_complete_unknown_id_clean_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    Settings().write(workspace)
    initialize_database(workspace)

    result = runner.invoke(app, ["tasks", "complete", "999", "-w", str(workspace)])
    assert result.exit_code == 1
    assert "Memory not found" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
