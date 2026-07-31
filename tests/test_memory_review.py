from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.config.settings import database_path
from convsearch.memory.review import (
    build_review_queue,
    confirm_memory,
    invalidate_memory,
    set_memory_pinned,
)
from convsearch.storage.database import connect, initialize_database
from convsearch.storage.migrations import migration_files


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
    status: str = "active",
    confidence: float = 0.9,
    kind: str = "decision",
    project: str | None = None,
    pinned: int = 0,
    reviewed_at: str | None = None,
) -> int:
    content_hash = f"hash-{memory_id}"
    conn.execute(
        """
        INSERT INTO memories
          (memory_id, kind, subject_key, statement, status, confidence, project,
           task_state, conversation_id, message_id, extraction_version, content_hash,
           metadata_json, pinned, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'v1', ?, '{}', ?, ?)
        """,
        (
            memory_id,
            kind,
            f"subj-{memory_id}",
            f"statement {memory_id}",
            status,
            confidence,
            project,
            conv_id,
            msg_id,
            content_hash,
            pinned,
            reviewed_at,
        ),
    )
    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) VALUES (?, ?, ?, ?, ?)",
        (memory_id, f"statement {memory_id}", kind, project or "", status),
    )
    conn.commit()
    return memory_id


# --- migration idempotency ---


def test_migration_007_present() -> None:
    assert any(p.stem == "007_memory_review" for p in migration_files())


def test_migration_idempotent_on_existing_db(db_conn: sqlite3.Connection) -> None:
    # initialize_database is idempotent by construction (schema_migrations gate), and it must
    # be safe to call again against an already-migrated, populated DB.
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id)

    workspace = Path(db_conn.execute("PRAGMA database_list").fetchone()["file"]).parent.parent
    initialize_database(workspace)  # second call: must be a no-op, not raise

    row = db_conn.execute("SELECT pinned, reviewed_at FROM memories WHERE memory_id = 1").fetchone()
    assert row["pinned"] == 0
    assert row["reviewed_at"] is None


def test_new_columns_default(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id)
    row = db_conn.execute("SELECT pinned, reviewed_at FROM memories WHERE memory_id = 1").fetchone()
    assert row["pinned"] == 0
    assert row["reviewed_at"] is None


# --- queue prioritisation ---


def test_queue_prioritisation_order(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="contested")
    _insert_memory(db_conn, memory_id=2, conv_id=conv_id, msg_id=msg_id, status="active")
    _insert_memory(db_conn, memory_id=3, conv_id=conv_id, msg_id=msg_id, status="proposed")
    _insert_memory(db_conn, memory_id=4, conv_id=conv_id, msg_id=msg_id, confidence=0.7)
    _insert_memory(db_conn, memory_id=5, conv_id=conv_id, msg_id=msg_id, status="active")
    db_conn.execute(
        "INSERT INTO memory_relations (from_memory_id, to_memory_id, relation) "
        "VALUES (2, 5, 'conflicts_with')"
    )
    db_conn.commit()

    queue = build_review_queue(db_conn)
    ids = [item.memory_id for item in queue.items]
    # rank1 contested(1) < rank2 conflicts_with(2 and 5, tie-break by id) < rank3 proposed(3)
    # < rank4 low confidence(4)
    assert ids == [1, 2, 5, 3, 4]


def test_review_reason_per_rule(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="contested")
    _insert_memory(db_conn, memory_id=2, conv_id=conv_id, msg_id=msg_id, status="proposed")
    _insert_memory(db_conn, memory_id=3, conv_id=conv_id, msg_id=msg_id, confidence=0.7)

    queue = build_review_queue(db_conn)
    reasons = {item.memory_id: item.review_reason for item in queue.items}
    assert "conflicts with another" in reasons[1]
    assert "never been confirmed" in reasons[2]
    assert "low confidence" in reasons[3]


def test_pinned_excluded_from_pending_queue(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="contested", pinned=1
    )
    queue = build_review_queue(db_conn)
    assert queue.items == ()
    assert queue.total_pinned == 1


def test_invalidated_excluded_but_counted(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="invalidated")
    queue = build_review_queue(db_conn)
    assert queue.items == ()
    assert queue.total_invalidated == 1


def test_include_reviewed(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn,
        memory_id=1,
        conv_id=conv_id,
        msg_id=msg_id,
        status="proposed",
        reviewed_at="2026-01-01T00:00:00",
    )
    queue_default = build_review_queue(db_conn)
    assert queue_default.items == ()

    queue_all = build_review_queue(db_conn, include_reviewed=True)
    assert len(queue_all.items) == 1
    assert queue_all.items[0].memory_id == 1


def test_active_no_rule_excluded(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="active", confidence=0.9
    )
    queue = build_review_queue(db_conn)
    assert queue.items == ()
    assert queue.total_pending == 0


# --- confirm / invalidate ---


def test_confirm_memory_sets_active_and_stamps_reviewed(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="proposed")

    confirm_memory(db_conn, 1, reason="looks right")

    row = db_conn.execute("SELECT status, reviewed_at FROM memories WHERE memory_id = 1").fetchone()
    assert row["status"] == "active"
    assert row["reviewed_at"] is not None

    history = db_conn.execute(
        "SELECT old_status, new_status, reason FROM memory_status_history WHERE memory_id = 1"
    ).fetchone()
    assert history["old_status"] == "proposed"
    assert history["new_status"] == "active"
    assert history["reason"] == "looks right"


def test_invalidate_memory_sets_invalidated_and_stamps_reviewed(
    db_conn: sqlite3.Connection,
) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="contested")

    invalidate_memory(db_conn, 1, reason="wrong")

    row = db_conn.execute("SELECT status, reviewed_at FROM memories WHERE memory_id = 1").fetchone()
    assert row["status"] == "invalidated"
    assert row["reviewed_at"] is not None

    history = db_conn.execute(
        "SELECT new_status FROM memory_status_history WHERE memory_id = 1"
    ).fetchone()
    assert history["new_status"] == "invalidated"


def test_confirm_unknown_memory_raises(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="999"):
        confirm_memory(db_conn, 999)


def test_invalidate_unknown_memory_raises(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="999"):
        invalidate_memory(db_conn, 999)


# --- pin / unpin ---


def test_pin_unpin_round_trip(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id)

    set_memory_pinned(db_conn, 1, True)
    row = db_conn.execute("SELECT pinned FROM memories WHERE memory_id = 1").fetchone()
    assert row["pinned"] == 1

    set_memory_pinned(db_conn, 1, False)
    row = db_conn.execute("SELECT pinned FROM memories WHERE memory_id = 1").fetchone()
    assert row["pinned"] == 0


def test_set_memory_pinned_unknown_raises(db_conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="999"):
        set_memory_pinned(db_conn, 999, True)


# --- evidence / conflicts / superseded_by attached ---


def test_conflicts_and_superseded_by_attached(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(db_conn, memory_id=1, conv_id=conv_id, msg_id=msg_id, status="contested")
    _insert_memory(db_conn, memory_id=2, conv_id=conv_id, msg_id=msg_id, status="contested")
    db_conn.execute(
        "INSERT INTO memory_relations (from_memory_id, to_memory_id, relation, reason) "
        "VALUES (1, 2, 'conflicts_with', 'same subject, different answer')"
    )
    _insert_memory(db_conn, memory_id=3, conv_id=conv_id, msg_id=msg_id, status="proposed")
    _insert_memory(db_conn, memory_id=4, conv_id=conv_id, msg_id=msg_id, status="proposed")
    db_conn.execute(
        "INSERT INTO memory_relations (from_memory_id, to_memory_id, relation) "
        "VALUES (4, 3, 'supersedes')"
    )
    db_conn.commit()

    queue = build_review_queue(db_conn)
    by_id = {item.memory_id: item for item in queue.items}

    assert by_id[1].conflicts[0].memory_id == 2
    assert by_id[1].conflicts[0].reason == "same subject, different answer"
    assert by_id[2].conflicts[0].memory_id == 1

    assert by_id[3].superseded_by[0].memory_id == 4
    assert by_id[4].superseded_by == ()


def test_kind_and_project_filters(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    _insert_memory(
        db_conn,
        memory_id=1,
        conv_id=conv_id,
        msg_id=msg_id,
        status="proposed",
        kind="decision",
        project="alpha",
    )
    _insert_memory(
        db_conn,
        memory_id=2,
        conv_id=conv_id,
        msg_id=msg_id,
        status="proposed",
        kind="task",
        project="beta",
    )

    queue = build_review_queue(db_conn, kind="decision")
    assert [item.memory_id for item in queue.items] == [1]

    queue = build_review_queue(db_conn, project="beta")
    assert [item.memory_id for item in queue.items] == [2]


def test_limit_respected(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    for i in range(1, 6):
        _insert_memory(db_conn, memory_id=i, conv_id=conv_id, msg_id=msg_id, status="proposed")

    queue = build_review_queue(db_conn, limit=2)
    assert len(queue.items) == 2
    assert queue.total_pending == 5
