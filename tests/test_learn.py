from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.config.settings import Settings, database_path
from convsearch.feedback.learn import list_learned_preferences, run_self_improvement
from convsearch.feedback.models import InteractionEvent
from convsearch.feedback.store import record_event
from convsearch.llm.generate import GenerationResult, LLMUnavailableError
from convsearch.storage.database import connect, initialize_database


@pytest.fixture()
def db_conn() -> Generator[sqlite3.Connection, None, None]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        initialize_database(workspace)
        conn = connect(database_path(workspace))
        try:
            yield conn
        finally:
            conn.close()


def _settings() -> Settings:
    return Settings()


def _seed_interactions(conn: sqlite3.Connection) -> None:
    record_event(conn, InteractionEvent(event_type="search", query="faiss storage"))
    record_event(conn, InteractionEvent(event_type="search", query="faiss storage"))
    record_event(conn, InteractionEvent(event_type="search", query="chrome extension"))
    record_event(
        conn, InteractionEvent(event_type="open", query="faiss storage", conversation_id=1)
    )


def test_run_self_improvement_with_llm_writes_notes(db_conn: sqlite3.Connection) -> None:
    _seed_interactions(db_conn)

    def fake_generate(system: str, prompt: str, *, settings: Settings, **_: object):  # type: ignore[no-untyped-def]
        return GenerationResult(
            "Prioritize FAISS storage decisions.\nSurface Chrome extension design.",
            "ollama",
            "llama3.2:1b",
        )

    summary = run_self_improvement(db_conn, _settings(), use_llm=True, generate=fake_generate)

    assert summary.backend == "ollama"
    assert summary.model == "llama3.2:1b"
    assert summary.notes_written == 2
    assert summary.events_read == 4
    assert summary.notes == [
        "Prioritize FAISS storage decisions.",
        "Surface Chrome extension design.",
    ]

    rows = list_learned_preferences(db_conn)
    assert len(rows) == 2
    # newest first; weights descend by rank.
    assert rows[0][1] in set(summary.notes)
    weights = {row[1]: row[2] for row in rows}
    assert weights["Prioritize FAISS storage decisions."] == pytest.approx(1.0)
    assert weights["Surface Chrome extension design."] == pytest.approx(0.9)
    source = db_conn.execute("SELECT DISTINCT source FROM learned_preferences").fetchone()[0]
    assert source == "llm"


def test_empty_db_does_not_call_generate(db_conn: sqlite3.Connection) -> None:
    def boom(*_: object, **__: object) -> GenerationResult:
        raise AssertionError("generate must not be called on an empty interaction log")

    summary = run_self_improvement(db_conn, _settings(), use_llm=True, generate=boom)

    assert summary.notes_written == 0
    assert summary.events_read == 0
    assert summary.backend == "none"
    assert summary.model == "none"
    assert summary.notes == []
    assert list_learned_preferences(db_conn) == []


def test_llm_unavailable_falls_back_to_heuristic(db_conn: sqlite3.Connection) -> None:
    _seed_interactions(db_conn)

    def unavailable(*_: object, **__: object) -> GenerationResult:
        raise LLMUnavailableError("no ollama")

    summary = run_self_improvement(db_conn, _settings(), use_llm=True, generate=unavailable)

    assert summary.backend == "none"
    assert summary.notes_written >= 1
    # most popular query drives the first heuristic note.
    assert any("faiss storage" in note for note in summary.notes)

    rows = list_learned_preferences(db_conn)
    assert len(rows) == summary.notes_written
    source = db_conn.execute("SELECT DISTINCT source FROM learned_preferences").fetchone()[0]
    assert source == "heuristic"


def test_no_llm_uses_heuristic_without_generate(db_conn: sqlite3.Connection) -> None:
    _seed_interactions(db_conn)

    def boom(*_: object, **__: object) -> GenerationResult:
        raise AssertionError("generate must not be called when use_llm is False")

    summary = run_self_improvement(db_conn, _settings(), use_llm=False, generate=boom)

    assert summary.backend == "none"
    assert summary.notes_written >= 1
    assert any("faiss storage" in note for note in summary.notes)
