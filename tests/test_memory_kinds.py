from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.config.settings import database_path
from convsearch.memory.extract import extract_from_message
from convsearch.memory.models import MEMORY_KINDS
from convsearch.storage.database import connect, initialize_database


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
    """Insert minimal import/conversation/message rows. Returns (conversation_id, message_id)."""
    conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (1, '/tmp/x.zip', 'h1', 'imported')"
    )
    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, content_hash)"
        " VALUES (1, 'src-conv-1', 1, 'Conv', 'ch1')"
    )
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " source_order, is_primary_path, text, content_hash)"
        " VALUES (1, 'msg-1', 1, 'assistant', 0, 1, 'text', 'mh1')"
    )
    conn.commit()
    return 1, 1


# --- MEMORY_KINDS contains the new kinds ---


def test_new_kinds_registered() -> None:
    assert "constraint" in MEMORY_KINDS
    assert "open_question" in MEMORY_KINDS


# --- Extractor tags the new kinds ---


def test_extract_constraint() -> None:
    text = "The service must not exceed 200ms latency."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "constraint" in kinds


def test_extract_open_question() -> None:
    text = "The choice of database engine remains unclear."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "open_question" in kinds


def test_extract_open_question_phrase() -> None:
    text = "Open question: do we shard by tenant or by region?"
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "open_question" in kinds


# --- Decision recall: additional realistic phrasings (see task report) ---


def test_extract_decision_i_would_use() -> None:
    text = "For your exact strategy, I would use MarketData.app Free Forever first."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    decisions = [m for m in memories if m.kind == "decision"]
    assert len(decisions) == 1
    assert decisions[0].confidence == 0.9


def test_extract_decision_lets_use() -> None:
    text = "Let's use SQLite for the authoritative store."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "decision" in kinds


def test_extract_decision_ill_go_with() -> None:
    text = "I'll go with Postgres for the analytics warehouse."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "decision" in kinds


def test_extract_decision_switched_from() -> None:
    text = "We switched from Pinecone to FAISS for local vector search."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    kinds = {m.kind for m in memories}
    assert "decision" in kinds


def test_extract_decision_label_continuation() -> None:
    """A 'Decision:' label on its own line (segmentation splits it from the statement)

    still attributes the following sentence to kind=decision, even though that sentence
    carries no trigger phrase of its own.
    """
    text = (
        "Decision:\nUse SQLite as the authoritative store, FTS5 for lexical search, "
        "and FAISS for vector search."
    )
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    decisions = [m for m in memories if m.kind == "decision"]
    assert len(decisions) == 1
    assert decisions[0].statement.startswith("Use SQLite")
    assert decisions[0].confidence == 0.9


def test_extract_decision_label_continuation_only_one_sentence_ahead() -> None:
    """The label should not leak forward past the sentence immediately following it."""
    text = (
        "Decision:\nUse SQLite as the authoritative store.\n"
        "This is an unrelated later sentence about something else."
    )
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    decisions = [m for m in memories if m.kind == "decision"]
    assert len(decisions) == 1
    assert "unrelated" not in decisions[0].statement


# --- Fresh-DB CHECK constraint accepts the new kinds ---


def test_insert_constraint_kind_allowed(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    db_conn.execute(
        "INSERT INTO memories"
        "(kind, subject_key, statement, status, confidence, project, task_state,"
        " conversation_id, message_id, extraction_version, content_hash, metadata_json)"
        " VALUES ('constraint', 'perf/latency', 'must not exceed 200ms', 'active', 0.9,"
        " NULL, NULL, ?, ?, 'v1', 'hc1', '{}')",
        (conv_id, msg_id),
    )
    db_conn.commit()
    row = db_conn.execute("SELECT kind FROM memories WHERE content_hash = 'hc1'").fetchone()
    assert row["kind"] == "constraint"


def test_insert_open_question_kind_allowed(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    db_conn.execute(
        "INSERT INTO memories"
        "(kind, subject_key, statement, status, confidence, project, task_state,"
        " conversation_id, message_id, extraction_version, content_hash, metadata_json)"
        " VALUES ('open_question', 'arch/db', 'which engine?', 'active', 0.7,"
        " NULL, NULL, ?, ?, 'v1', 'hq1', '{}')",
        (conv_id, msg_id),
    )
    db_conn.commit()
    row = db_conn.execute("SELECT kind FROM memories WHERE content_hash = 'hq1'").fetchone()
    assert row["kind"] == "open_question"


def test_unknown_kind_still_rejected(db_conn: sqlite3.Connection) -> None:
    conv_id, msg_id = _seed_message(db_conn)
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO memories"
            "(kind, subject_key, statement, status, confidence, project, task_state,"
            " conversation_id, message_id, extraction_version, content_hash, metadata_json)"
            " VALUES ('bogus_kind', 's', 'x', 'active', 0.5,"
            " NULL, NULL, ?, ?, 'v1', 'hx1', '{}')",
            (conv_id, msg_id),
        )
