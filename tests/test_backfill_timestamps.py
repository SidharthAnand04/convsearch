"""Coverage for the 008_backfill_timestamps migration and `count_missing_timestamps`.

Migrations run once (tracked in `schema_migrations`), so to exercise the migration's SQL
directly against hand-seeded NULL rows this re-applies the script body rather than going
through `initialize_database` a second time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from convsearch.storage.database import connect
from convsearch.storage.migrations import MIGRATIONS_DIR
from convsearch.utils import count_missing_timestamps

BACKFILL_SQL = (MIGRATIONS_DIR / "008_backfill_timestamps.sql").read_text(encoding="utf-8")


def _seed(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """One conversation with no created_at, one dated message and one undated message,

    and a memory tied to the undated message. Returns (conversation_id, dated_message_id,
    undated_message_id).
    """
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES ('x', 'h', 'complete')"
    )
    import_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES ('conv-1', ?, 'T', 'ch1')",
        (import_id,),
    )
    conversation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, created_at, "
        "source_order, is_primary_path, text, content_hash) VALUES "
        "('m1', ?, 'user', '2024-03-01T00:00:00+00:00', 0, 1, 'hi', 'mh1')",
        (conversation_id,),
    )
    dated_message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, created_at, "
        "source_order, is_primary_path, text, content_hash) VALUES "
        "('m2', ?, 'user', NULL, 1, 1, 'bye', 'mh2')",
        (conversation_id,),
    )
    undated_message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO memories(kind, subject_key, statement, status, confidence, "
        "conversation_id, message_id, extraction_version, content_hash) VALUES "
        "('decision', 's', 'stmt', 'active', 0.9, ?, ?, 'v1', 'memh1')",
        (conversation_id, undated_message_id),
    )
    conn.commit()
    return conversation_id, dated_message_id, undated_message_id


def test_backfill_derives_from_related_rows(workspace: Path) -> None:
    from convsearch.config.settings import database_path

    conn = connect(database_path(workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    conversation_id, _dated_message_id, undated_message_id = _seed(conn)

    conn.executescript(BACKFILL_SQL)
    conn.commit()

    conv_created_at = conn.execute(
        "SELECT created_at FROM conversations WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()["created_at"]
    undated_message_created_at = conn.execute(
        "SELECT created_at FROM messages WHERE message_id = ?", (undated_message_id,)
    ).fetchone()["created_at"]
    memory_created_at = conn.execute(
        "SELECT created_at FROM memories WHERE message_id = ?", (undated_message_id,)
    ).fetchone()["created_at"]

    assert conv_created_at == "2024-03-01T00:00:00+00:00"
    # The message had no timestamp of its own, so it inherits the (now-backfilled)
    # conversation timestamp.
    assert undated_message_created_at == "2024-03-01T00:00:00+00:00"
    assert memory_created_at == "2024-03-01T00:00:00+00:00"
    conn.close()


def test_backfill_is_idempotent(workspace: Path) -> None:
    from convsearch.config.settings import database_path

    conn = connect(database_path(workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    _seed(conn)

    conn.executescript(BACKFILL_SQL)
    conn.commit()
    first_pass = {
        table: count_missing_timestamps(conn)[table]
        for table in ("conversations", "messages", "memories")
    }

    conn.executescript(BACKFILL_SQL)
    conn.commit()
    second_pass = {
        table: count_missing_timestamps(conn)[table]
        for table in ("conversations", "messages", "memories")
    }

    assert first_pass == second_pass
    conn.close()


def test_backfill_leaves_genuinely_undated_rows_null(workspace: Path) -> None:
    """A conversation with no timestamped messages anywhere in its chain (the

    extension-capture case) stays NULL rather than being fabricated.
    """
    from convsearch.config.settings import database_path

    conn = connect(database_path(workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES ('x', 'h', 'complete')"
    )
    import_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES ('conv-undated', ?, 'T', 'ch-undated')",
        (import_id,),
    )
    conversation_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, created_at, "
        "source_order, is_primary_path, text, content_hash) VALUES "
        "('m-undated', ?, 'user', NULL, 0, 1, 'hi', 'mh-undated')",
        (conversation_id,),
    )
    conn.commit()

    before = count_missing_timestamps(conn)
    conn.executescript(BACKFILL_SQL)
    conn.commit()
    after = count_missing_timestamps(conn)

    assert after == before
    conn.close()


def test_count_missing_timestamps_signature(workspace: Path) -> None:
    from convsearch.config.settings import database_path

    conn = connect(database_path(workspace))
    counts = count_missing_timestamps(conn)
    assert set(counts) == {"conversations", "messages", "memories"}
    assert all(isinstance(value, int) for value in counts.values())
    conn.close()
