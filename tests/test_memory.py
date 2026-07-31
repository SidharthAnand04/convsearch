from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.memory.extract import extract_from_message
from convsearch.memory.review import confirm_memory, set_memory_pinned
from convsearch.memory.search import decision_timeline, get_memory, list_memories, search_memories
from convsearch.memory.store import (
    clear_memories,
    extract_and_store_memories,
    preview_purge,
    set_task_state,
)
from convsearch.storage.database import connect, initialize_database
from convsearch.utils import stable_hash


@pytest.fixture()
def tmp_workspace() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        initialize_database(workspace)
        yield workspace


@pytest.fixture()
def db_conn(tmp_workspace: Path) -> Generator[sqlite3.Connection, None, None]:
    from convsearch.config.settings import database_path

    conn = connect(database_path(tmp_workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


def _insert_base_rows(
    conn: sqlite3.Connection,
    *,
    message_text: str = "We decided to use PostgreSQL over MySQL.",
    message_created_at: str | None = None,
    source_conv_id: str = "conv-1",
    source_msg_id: str = "msg-1",
    role: str = "assistant",
) -> tuple[int, int, int]:
    """Insert minimal import, conversation, message. Returns (import_id, conv_id, msg_id)."""
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        ("/tmp/test.zip", stable_hash("import", source_conv_id), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (source_conv_id, import_id, "Test Conversation", stable_hash("conv", source_conv_id)),
    )
    conv_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
        "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_msg_id,
            conv_id,
            role,
            1,
            1,
            message_text,
            stable_hash("msg", source_msg_id),
            message_created_at,
        ),
    )
    msg_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
        "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conv_id,
            msg_id,
            1,
            message_text,
            0,
            len(message_text),
            len(message_text.split()),
            stable_hash("passage", source_msg_id),
        ),
    )

    conn.commit()
    return import_id, conv_id, msg_id


# --- Test a: Extraction ---


def test_extract_kinds_and_offsets() -> None:
    text = "We decided to use PostgreSQL. TODO: set up migrations. I prefer tabs over spaces."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "decision" in kinds
    assert "task" in kinds
    assert "preference" in kinds

    # Verify quote is exact slice of text
    for m in memories:
        assert text[m.start_offset : m.end_offset] == m.quote


def test_extract_quote_is_exact_slice() -> None:
    text = "We decided to use FastAPI for the backend. Need to write tests for the API."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    assert len(memories) >= 1
    for m in memories:
        assert text[m.start_offset : m.end_offset] == m.quote, (
            f"quote mismatch: {m.quote!r} != {text[m.start_offset : m.end_offset]!r}"
        )


# --- Test b: Idempotency ---


def test_idempotency(db_conn: sqlite3.Connection) -> None:
    _insert_base_rows(db_conn, message_text="We decided to use Redis for caching.")

    summary1 = extract_and_store_memories(db_conn)
    assert summary1.inserted >= 1

    summary2 = extract_and_store_memories(db_conn)
    assert summary2.inserted == 0


def test_summary_reports_rejected_count(db_conn: sqlite3.Connection) -> None:
    """MemoryExtractionSummary.rejected/rejected_by_reason surface what the quality filter

    dropped -- previously only visible via logging.debug.
    """
    _insert_base_rows(
        db_conn,
        message_text="We decided to use Redis for caching. Account constraint.",
    )

    summary = extract_and_store_memories(db_conn)
    assert summary.inserted >= 1  # the decision still comes through
    assert summary.rejected >= 1  # "Account constraint." is below the min word count
    assert sum(summary.rejected_by_reason.values()) == summary.rejected
    assert any("word count" in reason for reason in summary.rejected_by_reason)


def test_memory_extraction_summary_defaults_are_additive() -> None:
    """Existing construction sites that don't pass rejected/rejected_by_reason still work."""
    from convsearch.memory.store import MemoryExtractionSummary

    summary = MemoryExtractionSummary(
        extracted=1, inserted=1, superseded=0, contested=0, entities=0
    )
    assert summary.rejected == 0
    assert summary.rejected_by_reason == {}


def test_extracted_memory_inherits_conversation_date_when_message_date_missing(
    db_conn: sqlite3.Connection,
) -> None:
    """A memory pulled from a message with no created_at falls back to the conversation's

    date rather than wall-clock "now" -- the memory should be dated by its source, not by
    when extraction happened to run.
    """
    conn = db_conn
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        ("/tmp/test3.zip", stable_hash("import", "fallback-test"), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, created_at, "
        "content_hash) VALUES (?, ?, ?, ?, ?)",
        (
            "conv-fallback",
            import_id,
            "Fallback Conv",
            "2024-03-01T00:00:00+00:00",
            stable_hash("conv", "conv-fallback"),
        ),
    )
    conv_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    text = "We decided to use gRPC for the API."
    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
        "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("msg-fallback-1", conv_id, "assistant", 1, 1, text, stable_hash("msg", "1"), None),
    )
    msg_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
        "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, msg_id, 1, text, 0, len(text), len(text.split()), stable_hash("passage", "1")),
    )
    conn.commit()

    extract_and_store_memories(conn)

    row = conn.execute("SELECT created_at FROM memories WHERE message_id = ?", (msg_id,)).fetchone()
    assert row is not None
    assert row["created_at"] == "2024-03-01T00:00:00+00:00"


def test_extracted_memory_created_at_null_when_undated(db_conn: sqlite3.Connection) -> None:
    """When neither the message nor its conversation has a timestamp, the memory stays NULL

    rather than being stamped with wall-clock "now".
    """
    _insert_base_rows(db_conn, message_created_at=None)

    extract_and_store_memories(db_conn)

    rows = db_conn.execute("SELECT created_at FROM memories").fetchall()
    assert rows
    assert all(row["created_at"] is None for row in rows)


# --- Test c: Supersession ---


def test_supersession(db_conn: sqlite3.Connection) -> None:
    """Two decisions for same subject_key with different created_at -> older is superseded."""
    conn = db_conn
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        ("/tmp/test2.zip", stable_hash("import", "sup-test"), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES (?, ?, ?, ?)",
        ("conv-sup", import_id, "Supersession Conv", stable_hash("conv", "conv-sup")),
    )
    conv_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Older message — subject_key will be 'db' (all-caps identifier)
    old_text = "We decided to use DB sharding approach for storage."
    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
        "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "msg-sup-1",
            conv_id,
            "assistant",
            1,
            1,
            old_text,
            stable_hash("msg", "msg-sup-1"),
            "2024-01-01T10:00:00",
        ),
    )
    msg1_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
        "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conv_id,
            msg1_id,
            1,
            old_text,
            0,
            len(old_text),
            len(old_text.split()),
            stable_hash("p", "msg-sup-1"),
        ),
    )

    # Newer message — same subject_key 'db', different choice, later date
    new_text = "We decided to use DB clustering approach for storage."
    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
        "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "msg-sup-2",
            conv_id,
            "assistant",
            2,
            1,
            new_text,
            stable_hash("msg", "msg-sup-2"),
            "2024-01-02T10:00:00",
        ),
    )
    msg2_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
        "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            conv_id,
            msg2_id,
            1,
            new_text,
            0,
            len(new_text),
            len(new_text.split()),
            stable_hash("p", "msg-sup-2"),
        ),
    )
    conn.commit()

    summary = extract_and_store_memories(db_conn)
    assert summary.superseded >= 1

    statuses = {
        row["status"]
        for row in conn.execute("SELECT status FROM memories WHERE kind = 'decision'").fetchall()
    }
    assert "superseded" in statuses
    assert "active" in statuses

    # memory_relations has a supersedes row
    relation_count = conn.execute(
        "SELECT COUNT(*) FROM memory_relations WHERE relation = 'supersedes'"
    ).fetchone()[0]
    assert relation_count >= 1

    # memory_status_history has an entry
    history_count = conn.execute(
        "SELECT COUNT(*) FROM memory_status_history WHERE new_status = 'superseded'"
    ).fetchone()[0]
    assert history_count >= 1


# --- Test d: Contested ---


def test_contested(db_conn: sqlite3.Connection) -> None:
    """Two decisions same subject_key, same created_at -> both contested."""
    conn = db_conn
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        ("/tmp/test3.zip", stable_hash("import", "cont-test"), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES (?, ?, ?, ?)",
        ("conv-cont", import_id, "Contested Conv", stable_hash("conv", "conv-cont")),
    )
    conv_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    same_date = "2024-01-01T10:00:00"

    # Both sentences have the same subject_key 'db' (all-caps identifier)
    for i, approach in enumerate(("DB sharding option", "DB clustering option"), start=1):
        text = f"We decided to use {approach} for storage."
        conn.execute(
            "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
            "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"msg-cont-{i}",
                conv_id,
                "assistant",
                i,
                1,
                text,
                stable_hash("msg", f"msg-cont-{i}"),
                same_date,
            ),
        )
        msg_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
            "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                msg_id,
                1,
                text,
                0,
                len(text),
                len(text.split()),
                stable_hash("p", f"msg-cont-{i}"),
            ),
        )
    conn.commit()

    summary = extract_and_store_memories(db_conn)
    assert summary.contested >= 2

    # Both memories for same subject should be contested
    contested_rows = conn.execute(
        "SELECT memory_id FROM memories WHERE kind = 'decision' AND status = 'contested'"
    ).fetchall()
    assert len(contested_rows) >= 2

    # conflicts_with relation exists
    relation_count = conn.execute(
        "SELECT COUNT(*) FROM memory_relations WHERE relation = 'conflicts_with'"
    ).fetchone()[0]
    assert relation_count >= 1


# --- Test e: Search ---


def test_search_memories(db_conn: sqlite3.Connection) -> None:
    _insert_base_rows(
        db_conn,
        message_text="We decided to use Kubernetes for deployment.",
        source_conv_id="conv-search",
        source_msg_id="msg-search",
    )
    extract_and_store_memories(db_conn)

    results = search_memories(db_conn, "Kubernetes")
    assert len(results) >= 1
    assert any("Kubernetes" in r.statement for r in results)

    # get_memory / list_memories round-trip
    fetched = get_memory(db_conn, results[0].memory_id)
    assert fetched is not None
    assert fetched.memory_id == results[0].memory_id

    listed = list_memories(db_conn, kind="decision")
    assert any(r.memory_id == results[0].memory_id for r in listed)

    # status filter: active should find it, invalidated should not
    active_results = search_memories(db_conn, "Kubernetes", statuses=["active"])
    assert len(active_results) >= 1

    invalid_results = search_memories(db_conn, "Kubernetes", statuses=["invalidated"])
    assert len(invalid_results) == 0


# --- Test f: Timeline ---


def test_decision_timeline(db_conn: sqlite3.Connection) -> None:
    """decision_timeline returns chronological order including superseded entries."""
    conn = db_conn
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        ("/tmp/test-tl.zip", stable_hash("import", "tl-test"), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES (?, ?, ?, ?)",
        ("conv-tl", import_id, "Timeline Conv", stable_hash("conv", "conv-tl")),
    )
    conv_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Both use subject_key='db' (all-caps identifier), different dates
    for i, (approach, date) in enumerate(
        [
            ("DB sharding for the backend", "2024-01-01T10:00:00"),
            ("DB clustering for the backend", "2024-06-01T10:00:00"),
        ],
        start=1,
    ):
        text = f"We decided to use {approach}."
        conn.execute(
            "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
            "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"msg-tl-{i}",
                conv_id,
                "assistant",
                i,
                1,
                text,
                stable_hash("msg", f"msg-tl-{i}"),
                date,
            ),
        )
        msg_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO passages(conversation_id, message_id, passage_order, text, "
            "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                msg_id,
                1,
                text,
                0,
                len(text),
                len(text.split()),
                stable_hash("p", f"msg-tl-{i}"),
            ),
        )
    conn.commit()

    extract_and_store_memories(db_conn)

    # Get the subject_key from the stored decisions for this conversation
    rows = conn.execute(
        "SELECT subject_key FROM memories WHERE kind = 'decision' AND conversation_id = ?",
        (conv_id,),
    ).fetchall()
    assert len(rows) >= 2

    subject = rows[0]["subject_key"]
    timeline = decision_timeline(db_conn, subject)
    assert len(timeline) >= 2

    statuses_in_timeline = {r.status for r in timeline}
    assert "superseded" in statuses_in_timeline or "active" in statuses_in_timeline

    # Timeline is chronological: first entry has earlier created_at or None
    if len(timeline) >= 2 and timeline[0].created_at and timeline[1].created_at:
        assert timeline[0].created_at <= timeline[1].created_at


# --- Test e: clear_memories / re-extraction purge ---


def _insert_memory(
    conn: sqlite3.Connection,
    conv_id: int,
    msg_id: int,
    *,
    content_hash: str,
    extraction_version: str = "rules-v1",
    pinned: int = 0,
    reviewed_at: str | None = None,
    statement: str = "We decided to use SQLite.",
    kind: str = "decision",
    task_state: str | None = None,
) -> int:
    """Insert a minimal memory row directly (bypassing extraction) for purge tests."""
    conn.execute(
        "INSERT INTO memories(kind, subject_key, statement, status, confidence, project, "
        "task_state, conversation_id, message_id, extraction_version, content_hash, "
        "metadata_json, pinned, reviewed_at) "
        "VALUES (?, 'sqlite', ?, 'active', 0.9, NULL, ?, ?, ?, ?, ?, '{}', ?, ?)",
        (
            kind,
            statement,
            task_state,
            conv_id,
            msg_id,
            extraction_version,
            content_hash,
            pinned,
            reviewed_at,
        ),
    )
    conn.commit()
    memory_id: int = conn.execute(
        "SELECT memory_id FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()["memory_id"]
    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) "
        "VALUES (?, ?, ?, '', 'active')",
        (memory_id, statement, kind),
    )
    conn.execute(
        "INSERT INTO memory_evidence(memory_id, message_id, quote, start_offset, end_offset) "
        "VALUES (?, ?, ?, 0, ?)",
        (memory_id, msg_id, statement, len(statement)),
    )
    conn.commit()
    return memory_id


def test_clear_memories_scoped_by_version(db_conn: sqlite3.Connection) -> None:
    _, conv_id, msg_id = _insert_base_rows(db_conn, source_conv_id="c1", source_msg_id="m1")
    old_id = _insert_memory(
        db_conn, conv_id, msg_id, content_hash="h-old", extraction_version="rules-v1"
    )
    new_id = _insert_memory(
        db_conn, conv_id, msg_id, content_hash="h-new", extraction_version="rules-v2"
    )

    summary = clear_memories(db_conn, extraction_version="rules-v1")
    assert summary.deleted == 1
    assert summary.preserved == 0

    remaining = {r["memory_id"] for r in db_conn.execute("SELECT memory_id FROM memories")}
    assert old_id not in remaining
    assert new_id in remaining


def test_clear_memories_preserves_curated_rows(db_conn: sqlite3.Connection) -> None:
    _, conv_id, msg_id = _insert_base_rows(db_conn, source_conv_id="c2", source_msg_id="m2")
    plain_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-plain")
    pinned_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-pinned")
    reviewed_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-reviewed")
    manual_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-manual")

    set_memory_pinned(db_conn, pinned_id, True)
    confirm_memory(db_conn, reviewed_id, reason="looks right")
    # A manual status change with a reason, but not via confirm_memory (no reviewed_at
    # stamp) -- still must not be swept up by an automatic-reconcile-only purge.
    db_conn.execute(
        "INSERT INTO memory_status_history(memory_id, old_status, new_status, reason) "
        "VALUES (?, 'active', 'active', 'manually re-confirmed')",
        (manual_id,),
    )
    db_conn.commit()

    summary = clear_memories(db_conn)
    assert summary.deleted == 1
    assert summary.preserved == 3

    remaining = {r["memory_id"] for r in db_conn.execute("SELECT memory_id FROM memories")}
    assert plain_id not in remaining
    assert pinned_id in remaining
    assert reviewed_id in remaining
    assert manual_id in remaining


def test_clear_memories_preserves_task_state_history(db_conn: sqlite3.Connection) -> None:
    _, conv_id, msg_id = _insert_base_rows(db_conn, source_conv_id="c2b", source_msg_id="m2b")
    plain_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-plain-task")
    task_id = _insert_memory(
        db_conn,
        conv_id,
        msg_id,
        content_hash="h-task",
        kind="task",
        task_state="open",
        statement="Finish the migration.",
    )

    # Curated via the task-completion mechanism only -- no pin, no reviewed_at.
    set_task_state(db_conn, task_id, "completed", reason="shipped")
    db_conn.commit()

    summary = clear_memories(db_conn)
    assert summary.deleted == 1
    assert summary.preserved == 1

    remaining = {r["memory_id"] for r in db_conn.execute("SELECT memory_id FROM memories")}
    assert plain_id not in remaining
    assert task_id in remaining

    row = db_conn.execute(
        "SELECT task_state FROM memories WHERE memory_id = ?", (task_id,)
    ).fetchone()
    assert row["task_state"] == "completed"

    history = db_conn.execute(
        "SELECT COUNT(*) AS n FROM task_state_history WHERE memory_id = ?", (task_id,)
    ).fetchone()["n"]
    assert history == 1


def test_clear_memories_cleans_dependent_rows_no_orphans(db_conn: sqlite3.Connection) -> None:
    _, conv_id, msg_id = _insert_base_rows(db_conn, source_conv_id="c3", source_msg_id="m3")
    old_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-a")
    new_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-b")
    db_conn.execute(
        "INSERT INTO memory_relations(from_memory_id, to_memory_id, relation) "
        "VALUES (?, ?, 'supersedes')",
        (new_id, old_id),
    )
    # Give new_id a status-history row too (reason IS NULL, i.e. automatic-looking) so it
    # isn't curated by that signal either -- both memories here are uncurated and must go,
    # along with every dependent row that names them.
    db_conn.execute(
        "INSERT INTO memory_status_history(memory_id, old_status, new_status, reason) "
        "VALUES (?, 'active', 'active', NULL)",
        (new_id,),
    )
    db_conn.commit()

    summary = clear_memories(db_conn)
    assert summary.deleted == 2  # neither is curated, both go

    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id IN (?, ?)", (old_id, new_id)
        ).fetchone()[0]
        == 0
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM memory_relations WHERE from_memory_id = ? OR to_memory_id = ?",
            (new_id, old_id),
        ).fetchone()[0]
        == 0
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM memory_fts WHERE rowid IN (?, ?)", (old_id, new_id)
        ).fetchone()[0]
        == 0
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM memory_status_history WHERE memory_id IN (?, ?)", (old_id, new_id)
        ).fetchone()[0]
        == 0
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM task_state_history WHERE memory_id IN (?, ?)", (old_id, new_id)
        ).fetchone()[0]
        == 0
    )


def test_clear_memories_empty_database_is_noop(db_conn: sqlite3.Connection) -> None:
    summary = clear_memories(db_conn)
    assert summary.deleted == 0
    assert summary.preserved == 0

    summary_scoped = clear_memories(db_conn, extraction_version="rules-v2")
    assert summary_scoped.deleted == 0
    assert summary_scoped.preserved == 0


def test_preview_purge_matches_clear_memories_without_deleting(
    db_conn: sqlite3.Connection,
) -> None:
    _, conv_id, msg_id = _insert_base_rows(db_conn, source_conv_id="c4", source_msg_id="m4")
    plain_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-plain-preview")
    pinned_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-pinned-preview")
    reviewed_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-reviewed-preview")
    manual_id = _insert_memory(db_conn, conv_id, msg_id, content_hash="h-manual-preview")
    task_id = _insert_memory(
        db_conn,
        conv_id,
        msg_id,
        content_hash="h-task-preview",
        kind="task",
        task_state="open",
    )
    set_memory_pinned(db_conn, pinned_id, True)
    confirm_memory(db_conn, reviewed_id, reason="looks right")
    db_conn.execute(
        "INSERT INTO memory_status_history(memory_id, old_status, new_status, reason) "
        "VALUES (?, 'active', 'active', 'manually re-confirmed')",
        (manual_id,),
    )
    db_conn.commit()
    set_task_state(db_conn, task_id, "completed", reason="shipped")
    db_conn.commit()

    preview = preview_purge(db_conn)
    assert preview.deleted == 1
    assert preview.preserved == 4

    # Read-only: nothing was actually removed.
    remaining = {r["memory_id"] for r in db_conn.execute("SELECT memory_id FROM memories")}
    assert plain_id in remaining
    assert pinned_id in remaining
    assert reviewed_id in remaining
    assert manual_id in remaining
    assert task_id in remaining

    # And it agrees with what clear_memories would actually do.
    actual = clear_memories(db_conn)
    assert actual.deleted == preview.deleted
    assert actual.preserved == preview.preserved
