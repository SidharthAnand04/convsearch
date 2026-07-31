"""Tests for the opt-in LLM-assisted memory extraction path (memory/llm_extract.py).

The LLM is always mocked here -- no network, no local Ollama dependency. These verify the
contract that matters most for this module: a proposal is only ever accepted if its quote is a
verified verbatim substring of the real message text this function fetched itself, the existing
precision filter still applies, malformed output never raises, and LLM unavailability degrades
cleanly rather than propagating.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from convsearch.cli.app import app
from convsearch.config.settings import Settings, database_path
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.llm.generate import GenerationResult, LLMUnavailableError
from convsearch.memory import llm_extract as llm_extract_mod
from convsearch.memory.llm_extract import LlmProposalResult, propose_memories
from convsearch.storage.database import connect, connection, initialize_database
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
    conn = connect(database_path(tmp_workspace))
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


def _insert_message(
    conn: sqlite3.Connection,
    *,
    text: str,
    source_conv_id: str = "conv-1",
    source_msg_id: str = "msg-1",
) -> tuple[int, int]:
    """Insert minimal import/conversation/message/passage rows. Returns (conv_id, msg_id)."""
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
        (source_msg_id, conv_id, "assistant", 1, 1, text, stable_hash("msg", source_msg_id), None),
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
            stable_hash("p", source_msg_id),
        ),
    )
    conn.commit()
    return conv_id, msg_id


def _fake_generate(monkeypatch: pytest.MonkeyPatch, responses: list[str] | str) -> None:
    """Patch generate_text (as imported into llm_extract) to return canned text in order."""
    queue = [responses] if isinstance(responses, str) else list(responses)

    def _fake(system: str, prompt: str, *, settings: Settings, max_tokens: int | None = None):
        text = queue.pop(0) if queue else "[]"
        return GenerationResult(text=text, backend="ollama", model="fake-model")

    monkeypatch.setattr(llm_extract_mod, "generate_text", _fake)


def _fake_generate_raises(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def _fake(system: str, prompt: str, *, settings: Settings, max_tokens: int | None = None):
        raise exc

    monkeypatch.setattr(llm_extract_mod, "generate_text", _fake)


# --- Well-formed proposal accepted ---


def test_wellformed_proposal_with_verbatim_quote_is_accepted(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "After long discussion we decided to use SQLite for local storage going forward."
    _insert_message(db_conn, text=text)

    payload = json.dumps(
        [
            {
                "kind": "decision",
                "statement": "The team decided to use SQLite for local storage.",
                "quote": "we decided to use SQLite for local storage going forward",
                "project": "convsearch",
            }
        ]
    )
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert isinstance(result, LlmProposalResult)
    assert result.proposed == 1
    assert len(result.accepted) == 1
    mem = result.accepted[0]
    assert mem.kind == "decision"
    assert mem.quote == "we decided to use SQLite for local storage going forward"
    # Evidence must be an exact slice of the real message text at the resolved offsets.
    assert text[mem.start_offset : mem.end_offset] == mem.quote
    assert result.discarded_not_verbatim == 0
    assert result.discarded_malformed == 0
    assert result.llm_calls == 1


# --- Non-verbatim quote discarded ---


def test_proposal_with_nonverbatim_quote_is_discarded(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We are going to use PostgreSQL for the main database."
    _insert_message(db_conn, text=text)

    payload = json.dumps(
        [
            {
                "kind": "decision",
                "statement": "The team decided to use MongoDB for the database.",
                "quote": "we decided to use MongoDB because it scales better",
                "project": None,
            }
        ]
    )
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert result.proposed == 1
    assert result.accepted == ()
    assert result.discarded_not_verbatim == 1


# --- Malformed / non-JSON output discarded without raising ---


def test_malformed_json_output_discarded_without_raising(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use FastAPI for the backend service going forward."
    _insert_message(db_conn, text=text)
    _fake_generate(monkeypatch, "this is not json at all { broken")

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_malformed == 1


def test_json_with_missing_fields_discarded_as_malformed(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use FastAPI for the backend service going forward."
    _insert_message(db_conn, text=text)
    payload = json.dumps([{"kind": "decision", "statement": "missing quote field"}])
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_malformed == 1


def test_json_with_invalid_kind_discarded_as_malformed(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use FastAPI for the backend service going forward."
    _insert_message(db_conn, text=text)
    payload = json.dumps(
        [
            {
                "kind": "not_a_real_kind",
                "statement": "We decided to use FastAPI.",
                "quote": "We decided to use FastAPI",
                "project": None,
            }
        ]
    )
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_malformed == 1


def test_non_list_json_discarded_as_malformed(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use FastAPI for the backend service going forward."
    _insert_message(db_conn, text=text)
    _fake_generate(monkeypatch, json.dumps({"kind": "decision"}))

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_malformed == 1


# --- LLMUnavailableError handled cleanly ---


def test_llm_unavailable_does_not_raise_and_returns_empty(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use FastAPI for the backend service going forward."
    _insert_message(db_conn, text=text)
    _fake_generate_raises(monkeypatch, LLMUnavailableError("no ollama, no cloud"))

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.llm_calls == 0
    assert result.proposed == 0


# --- Precision filter still applied to LLM survivors ---


def test_precision_filter_still_rejects_trailing_colon_fragment(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "For your strategy, you need to know: latency and cost considerations matter a lot."
    _insert_message(db_conn, text=text)
    payload = json.dumps(
        [
            {
                "kind": "task",
                "statement": "For your strategy, you need to know:",
                "quote": "For your strategy, you need to know:",
                "project": None,
            }
        ]
    )
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_precision_filter == 1
    assert result.discarded_not_verbatim == 0


# --- Dedup against existing memories ---


def test_dedup_against_existing_memory_in_store(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "We decided to use SQLite for local storage going forward, over Postgres."
    conv_id, msg_id = _insert_message(db_conn, text=text)

    db_conn.execute(
        "INSERT INTO memories(kind, subject_key, statement, status, confidence, project, "
        "task_state, conversation_id, message_id, created_at, extraction_version, "
        "content_hash, metadata_json) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "decision",
            "sqlite",
            "we decided to use sqlite for local storage",
            0.9,
            None,
            None,
            conv_id,
            msg_id,
            None,
            "rules-v2",
            stable_hash("decision", "sqlite", "dup", msg_id),
            "{}",
        ),
    )
    db_conn.commit()

    payload = json.dumps(
        [
            {
                "kind": "decision",
                "statement": "We decided to use SQLite for local storage",
                "quote": "We decided to use SQLite for local storage going forward",
                "project": None,
            }
        ]
    )
    _fake_generate(monkeypatch, payload)

    result = propose_memories(db_conn, settings=Settings())
    assert result.accepted == ()
    assert result.discarded_duplicate == 1


def test_dedup_within_same_call_across_conversations(
    db_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    text1 = "We decided to use Redis for caching across the whole system going forward."
    _insert_message(db_conn, text=text1, source_conv_id="conv-a", source_msg_id="msg-a")
    text2 = "Separately, we decided to use Redis for caching across the whole system going forward."
    _insert_message(db_conn, text=text2, source_conv_id="conv-b", source_msg_id="msg-b")

    payload = json.dumps(
        [
            {
                "kind": "decision",
                "statement": "We decided to use Redis for caching",
                "quote": "we decided to use Redis for caching across the whole system going"
                " forward",
                "project": None,
            }
        ]
    )
    # Same canned response for both conversations (two LLM calls).
    _fake_generate(monkeypatch, [payload, payload])

    result = propose_memories(db_conn, settings=Settings())
    assert len(result.accepted) == 1
    assert result.discarded_duplicate == 1
    assert result.llm_calls == 2


# --- CLI wiring: `memories extract --llm` routes through and persists ---


def _llm_extraction_versions(workspace: Path) -> list[str]:
    with connection(workspace) as conn:
        rows = conn.execute(
            "SELECT extraction_version FROM memories WHERE extraction_version = 'llm-v1'"
        ).fetchall()
    return [row["extraction_version"] for row in rows]


def test_cli_memories_extract_llm_flag_routes_through_and_inserts(
    workspace: Path,
    settings: Settings,
    export_zip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The export fixture's assistant message: "The local-first architecture uses FTS5 and
    # FAISS for hybrid search." -- the quote below is a verbatim substring of it.
    import_chatgpt_zip(export_zip, workspace, settings)
    payload = json.dumps(
        [
            {
                "kind": "decision",
                "statement": "The system uses FTS5 and FAISS for hybrid search.",
                "quote": "uses FTS5 and FAISS for hybrid search",
                "project": "convsearch",
            }
        ]
    )
    _fake_generate(monkeypatch, [payload])

    runner = CliRunner()
    result = runner.invoke(app, ["memories", "extract", "--workspace", str(workspace), "--llm"])
    assert result.exit_code == 0, result.output
    # Counts are colorized by rich, so assert on the labels here and on the exact count via the
    # database below.
    assert "LLM accepted:" in result.output
    assert "LLM inserted:" in result.output
    # Persisted through the shared store path under the LLM extraction_version: exactly one row.
    assert _llm_extraction_versions(workspace) == ["llm-v1"]


def test_cli_memories_extract_llm_unavailable_degrades_cleanly(
    workspace: Path,
    settings: Settings,
    export_zip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    _fake_generate_raises(monkeypatch, LLMUnavailableError("no backend"))

    runner = CliRunner()
    result = runner.invoke(app, ["memories", "extract", "--workspace", str(workspace), "--llm"])
    assert result.exit_code == 0, result.output
    assert "LLM backend unavailable; kept rules-only results" in result.output
    # Nothing was persisted from the LLM path.
    assert _llm_extraction_versions(workspace) == []


def test_cli_memories_extract_rejects_bad_backend(workspace: Path, export_zip: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memories", "extract", "--workspace", str(workspace), "--llm", "--backend", "bogus"],
    )
    assert result.exit_code != 0
