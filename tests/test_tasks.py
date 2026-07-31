from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from convsearch.config.settings import Settings, database_path
from convsearch.storage.database import connect, initialize_database
from convsearch.tasks.query import TaskList, list_tasks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert minimal rows and return a mapping of logical name -> primary key."""
    ids: dict[str, int] = {}

    conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (1, 'x', 'h1', 'ok')"
    )

    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, content_hash)"
        " VALUES (1, 'src-conv-1', 1, 'Conv Alpha', 'ch1')"
    )
    conn.execute(
        "INSERT INTO conversations"
        "(conversation_id, source_conversation_id, import_id, title, content_hash)"
        " VALUES (2, 'src-conv-2', 1, 'Conv Beta', 'ch2')"
    )

    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " created_at, source_order, is_primary_path, text, content_hash)"
        " VALUES (1, 'msg-1', 1, 'user', '2024-01-01T00:00:00', 0, 1, 'msg text alpha', 'mh1')"
    )
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " created_at, source_order, is_primary_path, text, content_hash)"
        " VALUES (2, 'msg-2', 2, 'assistant', '2024-02-01T00:00:00', 0, 1, 'msg text beta', 'mh2')"
    )

    conn.execute(
        "INSERT INTO passages"
        "(passage_id, conversation_id, message_id, passage_order,"
        " text, start_offset, end_offset, word_count, content_hash)"
        " VALUES (1, 1, 1, 0, 'passage text alpha', 0, 18, 3, 'ph1')"
    )

    # 1. open task, project Alpha, has evidence
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (1, 'task', 'task/perf', 'Benchmark query performance', 'active', 0.8,"
        " 'Alpha', 'open', 1, 1, '2024-01-03', 'v1', 'c1', '{}')"
    )
    ids["open_alpha"] = 1

    # 2. completed task, project Alpha, no evidence
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (2, 'task', 'task/schema', 'Design initial schema', 'active', 0.9,"
        " 'Alpha', 'completed', 1, 1, '2024-01-04', 'v1', 'c2', '{}')"
    )
    ids["completed_alpha"] = 2

    # 3. open task, project Beta, invalidated -> must be excluded
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (3, 'task', 'task/old', 'Old task no longer relevant', 'invalidated', 0.5,"
        " 'Beta', 'open', 2, 2, '2024-01-05', 'v1', 'c3', '{}')"
    )
    ids["invalidated_beta"] = 3

    # 4. open task, project Beta, superseded -> must be excluded
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (4, 'task', 'task/dup', 'Duplicate task', 'superseded', 0.5,"
        " 'Beta', 'open', 2, 2, '2024-01-06', 'v1', 'c4', '{}')"
    )
    ids["superseded_beta"] = 4

    # 5. open task, project Beta, newest, used for ordering + since filter
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (5, 'task', 'task/beta', 'Wire the Beta endpoint', 'active', 0.7,"
        " 'Beta', 'open', 2, 2, '2024-03-01', 'v1', 'c5', '{}')"
    )
    ids["open_beta"] = 5

    # 6. a decision, must never show up as a task
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (6, 'decision', 'arch/store', 'Use SQLite for storage', 'active', 0.9,"
        " 'Alpha', NULL, 1, 1, '2024-01-01', 'v1', 'c6', '{}')"
    )
    ids["decision_alpha"] = 6

    # Evidence for open_alpha (mem 1)
    conn.execute(
        "INSERT INTO memory_evidence"
        "(evidence_id, memory_id, passage_id, message_id, quote, start_offset, end_offset)"
        " VALUES (1, 1, 1, 1, 'we should benchmark this', 0, 25)"
    )

    conn.commit()
    return ids


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_open_filter_excludes_completed_and_invalidated(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        result = list_tasks(seeded_conn, state="open")
        memory_ids = {item.memory_id for item in result.items}
        # mem 1 (open, active) and mem 5 (open, active) qualify; mem 3/4 excluded by status,
        # mem 2 excluded by task_state, mem 6 excluded since it isn't kind='task'.
        assert memory_ids == {1, 5}

    def test_completed_filter(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="completed")
        memory_ids = {item.memory_id for item in result.items}
        assert memory_ids == {2}

    def test_all_filter_still_excludes_invalidated_and_superseded(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        result = list_tasks(seeded_conn, state="all")
        memory_ids = {item.memory_id for item in result.items}
        assert memory_ids == {1, 2, 5}
        assert 3 not in memory_ids
        assert 4 not in memory_ids

    def test_project_filter(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="all", project="alpha")
        memory_ids = {item.memory_id for item in result.items}
        assert memory_ids == {1, 2}

    def test_invalidated_excluded_by_default(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="open", project="Beta")
        memory_ids = {item.memory_id for item in result.items}
        assert memory_ids == {5}

    def test_evidence_attached(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="open")
        by_id = {item.memory_id: item for item in result.items}
        open_alpha = by_id[1]
        assert open_alpha.has_evidence
        assert len(open_alpha.evidence) == 1
        ev = open_alpha.evidence[0]
        assert ev.quote == "we should benchmark this"
        assert ev.conversation_id == 1
        assert ev.conversation_title == "Conv Alpha"
        assert ev.message_id == 1
        assert ev.passage_id == 1
        assert ev.role == "user"
        assert ev.timestamp == "2024-01-01T00:00:00"

    def test_evidenceless_task_still_returned(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="completed")
        by_id = {item.memory_id: item for item in result.items}
        completed_alpha = by_id[2]
        assert completed_alpha.evidence == ()
        assert not completed_alpha.has_evidence

    def test_include_evidence_false_skips_lookup(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="open", include_evidence=False)
        for item in result.items:
            assert item.evidence == ()

    def test_has_evidence_truthful_when_evidence_not_fetched(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        """Regression for the bug where has_evidence read False for every task whenever
        include_evidence=False, even for tasks that demonstrably have evidence rows.
        has_evidence must reflect real evidence existence (evidence_count), independent of
        whether the evidence itself was loaded."""
        result = list_tasks(seeded_conn, state="open", include_evidence=False)
        by_id = {item.memory_id: item for item in result.items}
        open_alpha = by_id[1]
        assert open_alpha.has_evidence is True
        assert open_alpha.evidence_count == 1
        assert open_alpha.evidence == ()

    def test_ordering_newest_first_with_tiebreak(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="all")
        ordered_ids = [item.memory_id for item in result.items]
        # created_at desc: mem 5 (2024-03-01) > mem 2 (2024-01-04) > mem 1 (2024-01-03)
        assert ordered_ids == [5, 2, 1]

    def test_since_filter(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="all", since=datetime(2024, 2, 1))
        memory_ids = {item.memory_id for item in result.items}
        assert memory_ids == {5}

    def test_limit(self, seeded_conn: sqlite3.Connection) -> None:
        result = list_tasks(seeded_conn, state="all", limit=1)
        assert len(result.items) == 1
        assert result.items[0].memory_id == 5

    def test_total_counts_and_projects_ignore_limit(self, seeded_conn: sqlite3.Connection) -> None:
        result: TaskList = list_tasks(seeded_conn, state="all", limit=1)
        assert result.total_open == 2
        assert result.total_completed == 1
        assert result.projects == ("Alpha", "Beta")

    def test_invalid_state_raises(self, seeded_conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError):
            list_tasks(seeded_conn, state="bogus")
