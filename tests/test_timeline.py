from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.storage.database import connect, initialize_database
from convsearch.timeline import build_timeline
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


def _insert_conversation(conn: sqlite3.Connection, source_conv_id: str, title: str) -> int:
    conn.execute(
        "INSERT INTO imports(source_path, source_hash, status) VALUES (?, ?, ?)",
        (f"/tmp/{source_conv_id}.zip", stable_hash("import", source_conv_id), "imported"),
    )
    import_id: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO conversations(source_conversation_id, import_id, title, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (source_conv_id, import_id, title, stable_hash("conv", source_conv_id)),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_message(
    conn: sqlite3.Connection,
    conv_id: int,
    source_msg_id: str,
    text: str,
    created_at: str | None,
    order: int,
) -> int:
    conn.execute(
        "INSERT INTO messages(source_message_id, conversation_id, role, source_order, "
        "is_primary_path, text, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_msg_id,
            conv_id,
            "assistant",
            order,
            1,
            text,
            stable_hash("msg", source_msg_id),
            created_at,
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_memory(
    conn: sqlite3.Connection,
    *,
    kind: str,
    subject_key: str,
    statement: str,
    status: str,
    confidence: float,
    project: str | None,
    conversation_id: int,
    message_id: int,
    created_at: str | None,
    content_hash_seed: str,
) -> int:
    content_hash = stable_hash("memory", content_hash_seed)
    conn.execute(
        """
        INSERT INTO memories
          (kind, subject_key, statement, status, confidence, project,
           conversation_id, message_id, created_at, extraction_version, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'test-v1', ?)
        """,
        (
            kind,
            subject_key,
            statement,
            status,
            confidence,
            project,
            conversation_id,
            message_id,
            created_at,
            content_hash,
        ),
    )
    memory_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) VALUES (?, ?, ?, ?, ?)",
        (memory_id, statement, kind, project or "", status),
    )
    return memory_id


def _insert_evidence(conn: sqlite3.Connection, memory_id: int, message_id: int, quote: str) -> None:
    conn.execute(
        """
        INSERT INTO memory_evidence
          (memory_id, passage_id, message_id, quote, start_offset, end_offset)
        VALUES (?, NULL, ?, ?, 0, ?)
        """,
        (memory_id, message_id, quote, len(quote)),
    )


def _insert_relation(
    conn: sqlite3.Connection,
    from_memory_id: int,
    to_memory_id: int,
    relation: str,
    reason: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO memory_relations (from_memory_id, to_memory_id, relation, reason)
        VALUES (?, ?, ?, ?)
        """,
        (from_memory_id, to_memory_id, relation, reason),
    )


def _build_graph(conn: sqlite3.Connection) -> dict[str, int]:
    """Build: decision A (old, superseded) -> decision B (new, active) supersedes A with a
    reason; a contested pair C1/C2; and one unrelated memory that should never surface."""
    conv_id = _insert_conversation(conn, "conv-timeline", "Timeline Test Conversation")

    msg_a = _insert_message(
        conn, conv_id, "msg-a", "We decided to use REST for the API.", "2024-01-01T00:00:00", 1
    )
    msg_b = _insert_message(
        conn, conv_id, "msg-b", "We decided to use GraphQL for the API.", "2024-06-01T00:00:00", 2
    )
    msg_c1 = _insert_message(
        conn, conv_id, "msg-c1", "We decided to cache in Redis.", "2024-03-01T00:00:00", 3
    )
    msg_c2 = _insert_message(
        conn, conv_id, "msg-c2", "We decided to cache in Memcached.", "2024-03-01T00:00:00", 4
    )
    msg_u = _insert_message(
        conn, conv_id, "msg-u", "We prefer tabs over spaces.", "2024-02-01T00:00:00", 5
    )

    mem_a = _insert_memory(
        conn,
        kind="decision",
        subject_key="api-protocol",
        statement="Use REST for the API",
        status="superseded",
        confidence=0.9,
        project="demo",
        conversation_id=conv_id,
        message_id=msg_a,
        created_at="2024-01-01T00:00:00",
        content_hash_seed="mem-a",
    )
    _insert_evidence(conn, mem_a, msg_a, "We decided to use REST for the API.")

    mem_b = _insert_memory(
        conn,
        kind="decision",
        subject_key="api-protocol",
        statement="Use GraphQL for the API",
        status="active",
        confidence=0.9,
        project="demo",
        conversation_id=conv_id,
        message_id=msg_b,
        created_at="2024-06-01T00:00:00",
        content_hash_seed="mem-b",
    )
    _insert_evidence(conn, mem_b, msg_b, "We decided to use GraphQL for the API.")
    _insert_relation(conn, mem_b, mem_a, "supersedes", "REST couldn't support nested queries")

    mem_c1 = _insert_memory(
        conn,
        kind="decision",
        subject_key="cache-store",
        statement="Cache in Redis",
        status="contested",
        confidence=0.7,
        project="demo",
        conversation_id=conv_id,
        message_id=msg_c1,
        created_at="2024-03-01T00:00:00",
        content_hash_seed="mem-c1",
    )
    mem_c2 = _insert_memory(
        conn,
        kind="decision",
        subject_key="cache-store",
        statement="Cache in Memcached",
        status="contested",
        confidence=0.7,
        project="demo",
        conversation_id=conv_id,
        message_id=msg_c2,
        created_at="2024-03-01T00:00:00",
        content_hash_seed="mem-c2",
    )
    _insert_relation(conn, mem_c1, mem_c2, "conflicts_with", None)

    mem_u = _insert_memory(
        conn,
        kind="preference",
        subject_key="indentation",
        statement="Prefer tabs over spaces",
        status="active",
        confidence=0.5,
        project="demo",
        conversation_id=conv_id,
        message_id=msg_u,
        created_at="2024-02-01T00:00:00",
        content_hash_seed="mem-u",
    )

    conn.commit()
    return {"a": mem_a, "b": mem_b, "c1": mem_c1, "c2": mem_c2, "u": mem_u}


def test_timeline_oldest_first_and_expansion(db_conn: sqlite3.Connection) -> None:
    ids = _build_graph(db_conn)

    # "GraphQL" only matches decision B; decision A ("REST") should be pulled in via the
    # one-hop supersedes expansion even though it doesn't match the query text.
    timeline = build_timeline(db_conn, "GraphQL")

    node_ids = [n.memory_id for n in timeline.nodes]
    assert ids["a"] in node_ids
    assert ids["b"] in node_ids
    assert ids["u"] not in node_ids

    # oldest-first
    a_index = node_ids.index(ids["a"])
    b_index = node_ids.index(ids["b"])
    assert a_index < b_index

    assert timeline.matched_count == 1


def test_active_superseded_partition_and_reasons(db_conn: sqlite3.Connection) -> None:
    ids = _build_graph(db_conn)
    timeline = build_timeline(db_conn, "GraphQL")

    active_ids = {n.memory_id for n in timeline.active}
    superseded_ids = {n.memory_id for n in timeline.superseded}
    assert ids["b"] in active_ids
    assert ids["a"] in superseded_ids

    node_a = next(n for n in timeline.nodes if n.memory_id == ids["a"])
    node_b = next(n for n in timeline.nodes if n.memory_id == ids["b"])
    assert "REST couldn't support nested queries" in node_a.reasons
    assert "REST couldn't support nested queries" in node_b.reasons
    assert "Use REST for the API" in node_b.supersedes
    assert "Use GraphQL for the API" in node_a.superseded_by

    # rejected: superseded AND target of an explicit supersedes relation
    rejected_ids = {n.memory_id for n in timeline.rejected}
    assert ids["a"] in rejected_ids
    assert ids["b"] not in rejected_ids


def test_contested_partition(db_conn: sqlite3.Connection) -> None:
    ids = _build_graph(db_conn)
    timeline = build_timeline(db_conn, "Redis")

    contested_ids = {n.memory_id for n in timeline.contested}
    assert ids["c1"] in contested_ids
    # one-hop expansion via conflicts_with is not followed (only supersedes is), so the
    # Memcached alternative is only present if it matched the query text directly.
    node_c1 = next(n for n in timeline.nodes if n.memory_id == ids["c1"])
    assert "Cache in Memcached" in node_c1.conflicts_with


def test_empty_topic_returns_empty(db_conn: sqlite3.Connection) -> None:
    _build_graph(db_conn)

    timeline = build_timeline(db_conn, "")
    assert timeline.nodes == ()
    assert timeline.matched_count == 0

    timeline_no_match = build_timeline(db_conn, "quantum blockchain nonsense")
    assert timeline_no_match.nodes == ()
    assert timeline_no_match.matched_count == 0


def test_project_filter(db_conn: sqlite3.Connection) -> None:
    _build_graph(db_conn)

    timeline = build_timeline(db_conn, "GraphQL", project="demo")
    assert len(timeline.nodes) >= 1

    timeline_other = build_timeline(db_conn, "GraphQL", project="other-project")
    assert timeline_other.nodes == ()
    assert timeline_other.matched_count == 0
