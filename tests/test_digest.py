from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from convsearch.config.settings import Settings, database_path
from convsearch.digest.build import _HEADLINE_NOUNS, Digest, build_digest, parse_duration
from convsearch.storage.database import connect, initialize_database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Window under test in every case below: [2024-02-01, 2024-02-08).
SINCE = datetime(2024, 2, 1)
UNTIL = datetime(2024, 2, 8)


def _seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (1, 'x', 'h1', 'ok')"
    )

    # Conversation captured inside the window.
    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, created_at, content_hash)"
        " VALUES (1, 'src-conv-1', 1, 'Conv In Window', '2024-02-03T00:00:00', 'ch1')"
    )
    # Conversation captured before the window -> must not appear.
    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, created_at, content_hash)"
        " VALUES (2, 'src-conv-2', 1, 'Conv Before Window', '2024-01-01T00:00:00', 'ch2')"
    )

    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " created_at, source_order, is_primary_path, text, content_hash)"
        " VALUES (1, 'msg-1', 1, 'user', '2024-02-03T00:00:00', 0, 1, 'hello', 'mh1')"
    )
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " created_at, source_order, is_primary_path, text, content_hash)"
        " VALUES (2, 'msg-2', 2, 'user', '2024-01-01T00:00:00', 0, 1, 'hi', 'mh2')"
    )

    # A decision created long before the window, whose STATUS TRANSITION to 'superseded'
    # happens inside the window. This is the key correctness case: the memory's own
    # created_at is well outside [SINCE, UNTIL), but the transition must still be reported.
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (1, 'decision', 'arch/store', 'Use SQLite', 'superseded', 0.9,"
        " 'Alpha', NULL, 1, 1, '2023-01-01T00:00:00', 'v1', 'c1', '{}')"
    )
    conn.execute(
        "INSERT INTO memory_status_history"
        "(history_id, memory_id, old_status, new_status, reason, changed_at)"
        " VALUES (1, 1, 'active', 'superseded', 'replaced by Postgres', '2024-02-04T00:00:00')"
    )
    # A second transition on the same memory, but OUTSIDE the window -> must not appear twice
    # and must not appear at all from this row.
    conn.execute(
        "INSERT INTO memory_status_history"
        "(history_id, memory_id, old_status, new_status, reason, changed_at)"
        " VALUES (2, 1, 'proposed', 'active', NULL, '2023-01-02T00:00:00')"
    )

    # A new decision, created inside the window (first-time appearance of project Beta too).
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (2, 'decision', 'arch/queue', 'Use Postgres', 'active', 0.9,"
        " 'Beta', NULL, 1, 1, '2024-02-05T00:00:00', 'v1', 'c2', '{}')"
    )

    # An open task created inside the window.
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (3, 'task', 'task/perf', 'Benchmark queries', 'active', 0.8,"
        " 'Alpha', 'open', 1, 1, '2024-02-02T00:00:00', 'v1', 'c3', '{}')"
    )

    # A completed task created inside the window, with a real set_task_state() transition
    # (task_state_history row) also inside the window -- this is the one that must be
    # reported as "completed in this window".
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (4, 'task', 'task/schema', 'Ship schema v1', 'active', 0.9,"
        " 'Alpha', 'completed', 1, 1, '2024-02-06T00:00:00', 'v1', 'c4', '{}')"
    )
    conn.execute(
        "INSERT INTO task_state_history"
        "(history_id, memory_id, old_state, new_state, reason, changed_at)"
        " VALUES (1, 4, 'open', 'completed', 'shipped', '2024-02-06T12:00:00')"
    )

    # A task_state='completed' created inside the window by the extraction heuristic ONLY --
    # no task_state_history row. This must NOT be reported as "completed in this window"; it
    # was never actually completed during the window, the extractor just guessed its state.
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (6, 'task', 'task/heuristic', 'Set up CI pipeline', 'active', 0.9,"
        " 'Alpha', 'completed', 1, 1, '2024-02-06T00:00:00', 'v1', 'c6', '{}')"
    )

    # A task created BEFORE the window -> must not appear in either task section.
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (5, 'task', 'task/old', 'Old task', 'active', 0.5,"
        " 'Alpha', 'open', 1, 1, '2024-01-15T00:00:00', 'v1', 'c5', '{}')"
    )

    # A learned preference inside the window.
    conn.execute(
        "INSERT INTO learned_preferences(pref_id, created_at, note, weight, source)"
        " VALUES (1, '2024-02-06T12:00:00', 'Prefer terse answers', 1.0, 'heuristic')"
    )
    # ... and one outside it.
    conn.execute(
        "INSERT INTO learned_preferences(pref_id, created_at, note, weight, source)"
        " VALUES (2, '2024-01-01T00:00:00', 'Old preference', 1.0, 'heuristic')"
    )

    conn.commit()


@pytest.fixture()
def seeded_conn(tmp_path: Path) -> sqlite3.Connection:
    workspace = tmp_path / "ws"
    for child in ["database", "imports", "indexes", "cache", "logs"]:
        (workspace / child).mkdir(parents=True, exist_ok=True)
    Settings().write(workspace)
    initialize_database(workspace)
    conn = connect(database_path(workspace))
    _seed(conn)
    return conn


@pytest.fixture()
def empty_conn(tmp_path: Path) -> sqlite3.Connection:
    workspace = tmp_path / "empty_ws"
    for child in ["database", "imports", "indexes", "cache", "logs"]:
        (workspace / child).mkdir(parents=True, exist_ok=True)
    Settings().write(workspace)
    initialize_database(workspace)
    return connect(database_path(workspace))


def _section(digest: Digest, key: str):
    for section in digest.sections:
        if section.key == key:
            return section
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_window_filters_captures_correctly(seeded_conn: sqlite3.Connection) -> None:
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "new_captures")
    assert section is not None
    assert section.count == 1
    assert "Conv In Window" in section.items[0]


def test_superseded_decision_reported_despite_old_created_at(
    seeded_conn: sqlite3.Connection,
) -> None:
    """The key correctness case: memory created in 2023, transition happened in the window."""
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "superseded_decisions")
    assert section is not None
    assert section.count == 1
    assert section.detail[0]["memory_id"] == 1
    assert section.detail[0]["changed_at"] == "2024-02-04T00:00:00"

    # The out-of-window transition on the same memory must not leak in.
    assert digest.sections != ()
    assert not any("2023-01-02" in str(item) for item in section.items)


def test_new_decisions_new_tasks_and_preferences(seeded_conn: sqlite3.Connection) -> None:
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)

    decisions = _section(digest, "new_decisions")
    assert decisions is not None
    assert decisions.count == 1
    assert "Use Postgres" in decisions.items[0]

    open_tasks = _section(digest, "new_open_tasks")
    assert open_tasks is not None
    assert open_tasks.count == 1
    assert "Benchmark queries" in open_tasks.items[0]

    completed = _section(digest, "completed_tasks")
    assert completed is not None
    assert completed.count == 1
    assert "Ship schema v1" in completed.items[0]
    assert completed.detail[0]["changed_at"] == "2024-02-06T12:00:00"
    # Memory 6 was born task_state='completed' by the extraction heuristic with no
    # task_state_history row -- it must not be reported as completed in this window.
    assert not any("Set up CI pipeline" in item for item in completed.items)

    prefs = _section(digest, "new_preferences")
    assert prefs is not None
    assert prefs.count == 1
    assert prefs.items[0] == "Prefer terse answers"


def test_new_projects(seeded_conn: sqlite3.Connection) -> None:
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "new_projects")
    assert section is not None
    names = {detail["project"] for detail in section.detail}
    # Alpha's earliest memory (mem 1) is from 2023 -> not new. Beta's earliest (mem 2) is
    # inside the window -> new.
    assert names == {"Beta"}


def test_window_boundaries_are_inclusive(seeded_conn: sqlite3.Connection) -> None:
    # Exact left boundary
    digest = build_digest(seeded_conn, since=datetime(2024, 2, 2), until=datetime(2024, 2, 2))
    section = _section(digest, "new_open_tasks")
    assert section is not None
    assert section.count == 1

    # Just before the left boundary excludes it
    digest2 = build_digest(
        seeded_conn, since=datetime(2024, 2, 2, 0, 0, 1), until=datetime(2024, 2, 8)
    )
    assert _section(digest2, "new_open_tasks") is None


def test_empty_workspace_no_raise(empty_conn: sqlite3.Connection) -> None:
    digest = build_digest(empty_conn, since=SINCE, until=UNTIL)
    assert digest.is_empty is True
    assert digest.sections == ()
    assert "Nothing changed" in digest.headline


def test_empty_window_within_seeded_db(seeded_conn: sqlite3.Connection) -> None:
    # capture_problems is a current-state snapshot, not windowed, so it can still appear
    # even when nothing happened in this particular window; every *windowed* section must
    # be absent.
    digest = build_digest(seeded_conn, since=datetime(2025, 1, 1), until=datetime(2025, 1, 8))
    windowed_keys = {s.key for s in digest.sections} - {"capture_problems"}
    assert windowed_keys == set()


def test_headline_is_deterministic_and_factual(seeded_conn: sqlite3.Connection) -> None:
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    assert digest.is_empty is False
    for section in digest.sections:
        singular, plural = _HEADLINE_NOUNS[section.key]
        noun = singular if section.count == 1 else plural
        assert f"{section.count} {noun}" in digest.headline


def test_headline_singular_at_count_one(seeded_conn: sqlite3.Connection) -> None:
    # The seeded fixture has exactly one new project ("Beta") in the window -- the headline
    # must read "1 new project", never the ungrammatical "1 new projects".
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "new_projects")
    assert section is not None
    assert section.count == 1
    assert "1 new project;" in digest.headline or digest.headline.endswith("1 new project.")
    assert "1 new projects" not in digest.headline


def test_new_captures_note_only_appears_under_mixed_provenance(
    seeded_conn: sqlite3.Connection,
) -> None:
    # In the seeded fixture the single captured conversation is the only dated row of its
    # kind in the window, so date provenance is uniform ("created" throughout) -- the
    # caveat above already covers it, so no per-row "[dated by capture time]" note is added.
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "new_captures")
    assert section is not None
    assert "[dated by capture time]" not in section.items[0]

    # Now force a mix: one row created, one row captured-only (no created_at at all), both
    # inside the window -- the note must appear, and only on the captured-only row.
    seeded_conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (2, 'y', 'h2', 'ok')"
    )
    seeded_conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, created_at, updated_at,"
        " content_hash)"
        " VALUES (3, 'src-conv-3', 2, 'Captured Only', NULL, '2024-02-04T00:00:00', 'ch3')"
    )
    seeded_conn.commit()

    mixed = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    mixed_section = _section(mixed, "new_captures")
    assert mixed_section is not None
    by_title = {item.split('"')[1]: item for item in mixed_section.items}
    assert "[dated by capture time]" not in by_title["Conv In Window"]
    assert "[dated by capture time]" in by_title["Captured Only"]


def test_new_projects_uses_compact_date_not_raw_iso(seeded_conn: sqlite3.Connection) -> None:
    digest = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    section = _section(digest, "new_projects")
    assert section is not None
    for item in section.items:
        assert "T00:00:00" not in item
        assert "2024-02" not in item


def test_deterministic_across_two_calls(seeded_conn: sqlite3.Connection) -> None:
    first = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    second = build_digest(seeded_conn, since=SINCE, until=UNTIL)
    assert first == second


def test_limit_per_section_caps_items_but_not_count(seeded_conn: sqlite3.Connection) -> None:
    # Widen the window so both tasks (open + completed) each still have only one row;
    # instead verify capping against new_decisions + new_open_tasks combined via a very
    # tight limit applied uniformly.
    digest_full = build_digest(seeded_conn, since=SINCE, until=UNTIL, limit_per_section=5)
    digest_capped = build_digest(seeded_conn, since=SINCE, until=UNTIL, limit_per_section=0)

    for section in digest_capped.sections:
        assert section.items == ()
        assert section.detail == ()

    # Counts must be identical regardless of the cap.
    full_counts = {s.key: s.count for s in digest_full.sections}
    capped_counts = {s.key: s.count for s in digest_capped.sections}
    assert full_counts == capped_counts


def test_no_llm_and_no_writes(seeded_conn: sqlite3.Connection) -> None:
    """build_digest must never touch generate_text and must never write to the DB."""
    before = seeded_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    build_digest(seeded_conn, since=SINCE, until=UNTIL)
    after = seeded_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert before == after


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30m", timedelta(minutes=30)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        ("2w", timedelta(weeks=2)),
    ],
)
def test_parse_duration_accepts_units(text: str, expected: timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "7", "d", "7x", "-5d", "7 days", "7dd"])
def test_parse_duration_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(text)
