from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from convsearch.config.settings import Settings, database_path
from convsearch.projects.reconstruct import (
    ProjectReport,
    ProjectSummary,
    list_projects,
    reconstruct_project,
)
from convsearch.storage.database import connect, initialize_database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert minimal rows and return a mapping of logical name -> primary key."""
    ids: dict[str, int] = {}

    # imports
    conn.execute(
        "INSERT INTO imports(import_id, source_path, source_hash, status)"
        " VALUES (1, 'x', 'h1', 'ok')"
    )

    # conversations
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
    ids["conv_alpha"] = 1
    ids["conv_beta"] = 2

    # messages (one per conversation is enough)
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " source_order, is_primary_path, text, content_hash)"
        " VALUES (1, 'msg-1', 1, 'user', 0, 1, 'msg text alpha', 'mh1')"
    )
    conn.execute(
        "INSERT INTO messages"
        "(message_id, source_message_id, conversation_id, role,"
        " source_order, is_primary_path, text, content_hash)"
        " VALUES (2, 'msg-2', 2, 'user', 0, 1, 'msg text beta', 'mh2')"
    )
    ids["msg_alpha"] = 1
    ids["msg_beta"] = 2

    # passages (optional but tests passage_id reference in evidence)
    conn.execute(
        "INSERT INTO passages"
        "(passage_id, conversation_id, message_id, passage_order,"
        " text, start_offset, end_offset, word_count, content_hash)"
        " VALUES (1, 1, 1, 0, 'passage text alpha', 0, 18, 3, 'ph1')"
    )
    ids["passage_alpha"] = 1

    # ---------- Project "Alpha" memories ----------
    # 1. active decision (with rejected_alternative in metadata)
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (1, 'decision', 'arch/store', 'Use SQLite for storage', 'active', 0.9,"
        " 'Alpha', NULL, 1, 1, '2024-01-01', 'v1', 'c1', ?)",
        (json.dumps({"rejected_alternative": "PostgreSQL"}),),
    )
    ids["mem_active_decision"] = 1

    # 2. superseded decision (with rejected_alternative in metadata)
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (2, 'decision', 'arch/cache', 'Use in-process cache', 'superseded', 0.7,"
        " 'Alpha', NULL, 1, 1, '2024-01-02', 'v1', 'c2', ?)",
        (json.dumps({"rejected_alternative": "Redis"}),),
    )
    ids["mem_superseded_decision"] = 2

    # 3. open task
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (3, 'task', 'task/perf', 'Benchmark query performance', 'active', 0.8,"
        " 'Alpha', 'open', 1, 1, '2024-01-03', 'v1', 'c3', '{}')"
    )
    ids["mem_open_task"] = 3

    # 4. completed task
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (4, 'task', 'task/schema', 'Design initial schema', 'active', 0.9,"
        " 'Alpha', 'completed', 1, 1, '2024-01-04', 'v1', 'c4', '{}')"
    )
    ids["mem_completed_task"] = 4

    # 5. risk
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (5, 'risk', 'risk/lock', 'SQLite write lock under load', 'active', 0.6,"
        " 'Alpha', NULL, 1, 1, '2024-01-05', 'v1', 'c5', '{}')"
    )
    ids["mem_risk"] = 5

    # 6. project_state (active) — long statement split across lines for line-length
    state_stmt = "Alpha is a local-first search tool"
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (6, 'project_state', 'state/main', ?, 'active', 1.0,"
        " 'Alpha', NULL, 1, 1, '2024-01-06', 'v1', 'c6', '{}')",
        (state_stmt,),
    )
    ids["mem_project_state"] = 6

    # ---------- Project "Beta" memory (different project) ----------
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (7, 'decision', 'arch/model', 'Beta uses transformer embeddings',"
        " 'active', 0.85, 'Beta', NULL, 2, 2, '2024-02-01', 'v1', 'c7', '{}')"
    )
    ids["mem_beta_decision"] = 7

    # ---------- Project "Delta" memories (supersession-link fixtures) ----------
    # High memory_ids (100+) so they never collide with ids used by later tests that insert
    # ad hoc rows into the shared "Alpha"/"Beta" fixture (e.g. memory_id=9, 10 below).
    # 101. superseded decision, replaced by 102, reason recorded
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (101, 'decision', 'arch/cache2', 'Use Redis for caching', 'superseded', 0.8,"
        " 'Delta', NULL, 1, 1, '2024-03-01', 'v1', 'c101', '{}')"
    )
    ids["mem_superseded_with_reason"] = 101
    # 102. replacement for 101
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (102, 'decision', 'arch/cache2', 'Use Memcached for caching', 'active', 0.9,"
        " 'Delta', NULL, 1, 1, '2024-03-02', 'v1', 'c102', '{}')"
    )
    ids["mem_replacement_with_reason"] = 102
    # 103. superseded decision, replaced by 104, no reason recorded
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (103, 'decision', 'arch/updates', 'Use polling for updates', 'superseded', 0.8,"
        " 'Delta', NULL, 1, 1, '2024-03-03', 'v1', 'c103', '{}')"
    )
    ids["mem_superseded_no_reason"] = 103
    # 104. replacement for 103
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (104, 'decision', 'arch/updates', 'Use websockets for updates', 'active', 0.9,"
        " 'Delta', NULL, 1, 1, '2024-03-04', 'v1', 'c104', '{}')"
    )
    ids["mem_replacement_no_reason"] = 104
    # 105. superseded decision with no recorded supersession link at all
    conn.execute(
        "INSERT INTO memories"
        "(memory_id, kind, subject_key, statement, status, confidence,"
        " project, task_state, conversation_id, message_id, created_at, extraction_version,"
        " content_hash, metadata_json)"
        " VALUES (105, 'decision', 'arch/backups', 'Use manual backups', 'superseded', 0.7,"
        " 'Delta', NULL, 1, 1, '2024-03-05', 'v1', 'c105', '{}')"
    )
    ids["mem_superseded_no_link"] = 105

    conn.execute(
        "INSERT INTO memory_relations"
        "(relation_id, from_memory_id, to_memory_id, relation, reason)"
        " VALUES (1, 102, 101, 'supersedes', 'Redis added ops overhead we did not need')"
    )
    conn.execute(
        "INSERT INTO memory_relations"
        "(relation_id, from_memory_id, to_memory_id, relation, reason)"
        " VALUES (2, 104, 103, 'supersedes', NULL)"
    )

    # ---------- Evidence rows ----------
    # Evidence for active decision (mem 1) — references passage_id 1
    conn.execute(
        "INSERT INTO memory_evidence"
        "(evidence_id, memory_id, passage_id, message_id, quote, start_offset, end_offset)"
        " VALUES (1, 1, 1, 1, 'SQLite chosen over PostgreSQL', 0, 30)"
    )
    # Evidence for superseded decision (mem 2) — no passage
    conn.execute(
        "INSERT INTO memory_evidence"
        "(evidence_id, memory_id, passage_id, message_id, quote, start_offset, end_offset)"
        " VALUES (2, 2, NULL, 1, 'in-process cache was discussed', 0, 25)"
    )
    # Evidence for risk (mem 5)
    conn.execute(
        "INSERT INTO memory_evidence"
        "(evidence_id, memory_id, passage_id, message_id, quote, start_offset, end_offset)"
        " VALUES (3, 5, NULL, 1, 'lock contention under high write load', 5, 35)"
    )

    conn.commit()
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


class TestListProjects:
    def test_returns_both_projects(self, seeded_conn: sqlite3.Connection) -> None:
        summaries = list_projects(seeded_conn)
        names = [s.name for s in summaries]
        assert "Alpha" in names
        assert "Beta" in names

    def test_alpha_counts(self, seeded_conn: sqlite3.Connection) -> None:
        summaries = {s.name: s for s in list_projects(seeded_conn)}
        alpha: ProjectSummary = summaries["Alpha"]
        # 6 memories in Alpha
        assert alpha.memory_count == 6
        # all 6 touch conv 1 (conversation_id=1), so 1 distinct conversation
        assert alpha.conversation_count == 1
        # 2 decision rows (active + superseded)
        assert alpha.decision_count == 2
        # 1 open task
        assert alpha.open_task_count == 1
        # last_activity = max created_at across Alpha
        assert alpha.last_activity == "2024-01-06"

    def test_beta_counts(self, seeded_conn: sqlite3.Connection) -> None:
        summaries = {s.name: s for s in list_projects(seeded_conn)}
        beta: ProjectSummary = summaries["Beta"]
        assert beta.memory_count == 1
        assert beta.decision_count == 1
        assert beta.open_task_count == 0

    def test_last_activity_falls_back_to_capture_time(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        """A memory whose own `created_at` is NULL, and whose conversation also has
        `created_at = NULL`, must still surface as `last_activity` via the conversation's
        `updated_at` (capture time) -- not leave the column blank."""
        seeded_conn.execute(
            "INSERT INTO conversations"
            "(conversation_id, source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES (3, 'src-conv-3', 1, 'Conv Gamma', NULL,"
            " '2024-05-05', 'ch3')"
        )
        seeded_conn.execute(
            "INSERT INTO messages"
            "(message_id, source_message_id, conversation_id, role,"
            " source_order, is_primary_path, text, content_hash)"
            " VALUES (3, 'msg-3', 3, 'user', 0, 1, 'msg text gamma', 'mh3')"
        )
        seeded_conn.execute(
            "INSERT INTO memories"
            "(memory_id, kind, subject_key, statement, status, confidence,"
            " project, task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES (8, 'decision', 'arch/gamma', 'Gamma memory with no created_at', 'active',"
            " 0.5, 'Gamma', NULL, 3, 3, NULL, 'v1', 'c8', '{}')"
        )
        seeded_conn.commit()

        summaries = {s.name: s for s in list_projects(seeded_conn)}
        assert summaries["Gamma"].last_activity == "2024-05-05"

    def test_null_project_excluded(self, seeded_conn: sqlite3.Connection) -> None:
        # Insert a memory with project=NULL; it should not appear in list_projects
        seeded_conn.execute(
            "INSERT INTO memories"
            "(memory_id, kind, subject_key, statement, status, confidence,"
            " project, task_state, conversation_id, message_id, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES (99, 'risk', 'risk/x', 'orphan risk', 'active', 0.5,"
            " NULL, NULL, 1, 1, 'v1', 'cx99', '{}')"
        )
        seeded_conn.commit()
        names = [s.name for s in list_projects(seeded_conn)]
        assert None not in names


class TestReconstructProject:
    def test_unknown_project_returns_none(self, seeded_conn: sqlite3.Connection) -> None:
        assert reconstruct_project(seeded_conn, "NoSuchProject") is None

    def test_case_insensitive_name_match(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "alpha")
        assert report is not None
        assert report.name == "alpha"

    def test_returns_project_report(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert isinstance(report, ProjectReport)

    def test_summary_uses_active_project_state(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert report.summary == "Alpha is a local-first search tool"

    def test_summary_fallback_when_no_project_state(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Beta")
        assert report is not None
        # Beta has no project_state memory — expect synthetic one-liner
        assert "decision" in report.summary

    def test_timeline_contains_all_alpha_memories(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.timeline) == 6

    def test_timeline_ordered_by_created_at(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        dates = [e.created_at for e in report.timeline if e.created_at is not None]
        assert dates == sorted(dates)

    def test_superseded_decision_in_timeline(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        timeline_ids = {e.memory_id for e in report.timeline}
        assert 2 in timeline_ids  # superseded decision memory_id=2

    def test_active_decision_bucket(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.decisions) == 1
        assert report.decisions[0].memory_id == 1
        assert report.decisions[0].status == "active"

    def test_superseded_decision_bucket(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.superseded_decisions) == 1
        assert report.superseded_decisions[0].memory_id == 2
        assert report.superseded_decisions[0].status == "superseded"

    def test_superseded_decision_not_in_active_decisions(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        active_ids = {d.memory_id for d in report.decisions}
        assert 2 not in active_ids

    def test_rejected_alternatives_captured(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert "PostgreSQL" in report.rejected_alternatives
        assert "Redis" in report.rejected_alternatives
        # order-preserving; PostgreSQL (mem 1, earlier) should come first
        alts = list(report.rejected_alternatives)
        assert alts.index("PostgreSQL") < alts.index("Redis")

    def test_rejected_alternatives_deduplicated(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        alts = list(report.rejected_alternatives)
        assert len(alts) == len(set(alts))

    def test_open_tasks(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.open_tasks) == 1
        assert report.open_tasks[0].memory_id == 3

    def test_completed_tasks(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.completed_tasks) == 1
        assert report.completed_tasks[0].memory_id == 4

    def test_risks(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.risks) == 1
        assert report.risks[0].memory_id == 5

    def test_architecture(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.architecture) == 1
        assert report.architecture[0].memory_id == 6

    def test_conversations(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert len(report.conversations) == 1
        conv_id, title = report.conversations[0]
        assert conv_id == 1
        assert title == "Conv Alpha"

    def test_evidence_on_active_decision(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        decision = report.decisions[0]
        assert len(decision.evidence) == 1
        ev = decision.evidence[0]
        assert ev.quote == "SQLite chosen over PostgreSQL"
        assert ev.passage_id == 1
        assert ev.conversation_title == "Conv Alpha"

    def test_evidence_on_superseded_decision_no_passage(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        sd = report.superseded_decisions[0]
        assert len(sd.evidence) == 1
        assert sd.evidence[0].passage_id is None
        assert sd.evidence[0].quote == "in-process cache was discussed"

    def test_zero_evidence_item_has_empty_tuple(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        # open task (mem 3) has no evidence rows
        assert report.open_tasks[0].evidence == ()

    def test_evidence_count(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        # 3 evidence rows seeded for Alpha
        assert report.evidence_count == 3

    def test_beta_project_filtered_out(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        # No Beta memories should appear
        all_mem_ids = (
            {e.memory_id for e in report.timeline}
            | {d.memory_id for d in report.decisions}
            | {d.memory_id for d in report.superseded_decisions}
            | {t.memory_id for t in report.open_tasks}
            | {t.memory_id for t in report.completed_tasks}
            | {r.memory_id for r in report.risks}
            | {a.memory_id for a in report.architecture}
        )
        assert 7 not in all_mem_ids  # Beta's memory_id=7

    def test_beta_reconstruct_independent(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Beta")
        assert report is not None
        assert len(report.decisions) == 1
        assert report.decisions[0].memory_id == 7
        assert len(report.timeline) == 1

    def test_project_item_date_source_created(self, seeded_conn: sqlite3.Connection) -> None:
        """All seeded Alpha memories have a real `created_at`, so every ProjectItem bucket
        (not just the timeline) reports date_source='created'."""
        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        assert report.decisions[0].date_source == "created"
        assert report.open_tasks[0].date_source == "created"
        assert report.risks[0].date_source == "created"
        assert report.architecture[0].date_source == "created"

    def test_project_item_date_source_captured_fallback(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        """A memory with `created_at = NULL`, on a conversation with `created_at = NULL` but
        a real `updated_at`, must resolve to date_source='captured' and created_at equal to
        the conversation's capture time -- never fall through to 'unknown'."""
        seeded_conn.execute(
            "INSERT INTO conversations"
            "(conversation_id, source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES (4, 'src-conv-4', 1, 'Conv Delta', NULL,"
            " '2024-06-06', 'ch4')"
        )
        seeded_conn.execute(
            "INSERT INTO messages"
            "(message_id, source_message_id, conversation_id, role,"
            " source_order, is_primary_path, text, content_hash)"
            " VALUES (4, 'msg-4', 4, 'user', 0, 1, 'msg text delta', 'mh4')"
        )
        seeded_conn.execute(
            "INSERT INTO memories"
            "(memory_id, kind, subject_key, statement, status, confidence,"
            " project, task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES (9, 'risk', 'risk/delta', 'Delta risk with no created_at', 'active',"
            " 0.5, 'Alpha', NULL, 4, 4, NULL, 'v1', 'c9', '{}')"
        )
        seeded_conn.commit()

        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        delta_risk = next(r for r in report.risks if r.memory_id == 9)
        assert delta_risk.date_source == "captured"
        assert delta_risk.created_at == "2024-06-06"
        delta_timeline_entry = next(e for e in report.timeline if e.memory_id == 9)
        assert delta_timeline_entry.date_source == "captured"

    def test_timeline_ordering_uses_effective_timestamp(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        """A memory whose only date is the conversation's capture time must still sort in
        chronological position among memories with a real created_at, not be pushed to the
        end (or treated as unordered) because its own created_at is NULL."""
        seeded_conn.execute(
            "INSERT INTO conversations"
            "(conversation_id, source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES (5, 'src-conv-5', 1, 'Conv Epsilon', NULL,"
            " '2024-01-03T12:00:00', 'ch5')"
        )
        seeded_conn.execute(
            "INSERT INTO messages"
            "(message_id, source_message_id, conversation_id, role,"
            " source_order, is_primary_path, text, content_hash)"
            " VALUES (5, 'msg-5', 5, 'user', 0, 1, 'msg text epsilon', 'mh5')"
        )
        # Capture time (2024-01-03T12:00:00) falls between mem_open_task (2024-01-03) and
        # mem_completed_task (2024-01-04).
        seeded_conn.execute(
            "INSERT INTO memories"
            "(memory_id, kind, subject_key, statement, status, confidence,"
            " project, task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES (10, 'risk', 'risk/epsilon', 'Epsilon risk with no created_at', 'active',"
            " 0.5, 'Alpha', NULL, 5, 5, NULL, 'v1', 'c10', '{}')"
        )
        seeded_conn.commit()

        report = reconstruct_project(seeded_conn, "Alpha")
        assert report is not None
        ordered_ids = [e.memory_id for e in report.timeline]
        assert ordered_ids.index(3) < ordered_ids.index(10) < ordered_ids.index(4)


class TestSupersededByLink:
    """The 'Delta' fixture memories (101-105) exercise the supersession-link join: a
    superseded decision with a known replacement and reason, one with a known replacement
    but no reason, and one with no recorded supersession link at all."""

    def test_replacement_and_reason_known(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Delta")
        assert report is not None
        item = next(d for d in report.superseded_decisions if d.memory_id == 101)
        assert item.superseded_by is not None
        assert item.superseded_by.memory_id == 102
        assert item.superseded_by.statement == "Use Memcached for caching"
        assert item.superseded_by.reason == "Redis added ops overhead we did not need"

    def test_replacement_known_reason_absent(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Delta")
        assert report is not None
        item = next(d for d in report.superseded_decisions if d.memory_id == 103)
        assert item.superseded_by is not None
        assert item.superseded_by.memory_id == 104
        assert item.superseded_by.statement == "Use websockets for updates"
        assert item.superseded_by.reason is None

    def test_neither_replacement_nor_reason_recorded(self, seeded_conn: sqlite3.Connection) -> None:
        report = reconstruct_project(seeded_conn, "Delta")
        assert report is not None
        item = next(d for d in report.superseded_decisions if d.memory_id == 105)
        assert item.superseded_by is None

    def test_active_decision_has_no_superseded_by(self, seeded_conn: sqlite3.Connection) -> None:
        """The replacement itself is an active decision, not a superseded one, and it did
        not replace anything -- it must not carry a supersession link of its own."""
        report = reconstruct_project(seeded_conn, "Delta")
        assert report is not None
        replacement = next(d for d in report.decisions if d.memory_id == 102)
        assert replacement.superseded_by is None

    def test_multiple_superseded_decisions_resolved_in_one_batch(
        self, seeded_conn: sqlite3.Connection
    ) -> None:
        """All three superseded Delta decisions resolve correctly from a single batched
        query -- not just the first one, which an N+1-style bug could hide."""
        report = reconstruct_project(seeded_conn, "Delta")
        assert report is not None
        by_id = {d.memory_id: d for d in report.superseded_decisions}
        assert set(by_id) == {101, 103, 105}
        assert by_id[101].superseded_by is not None and by_id[101].superseded_by.memory_id == 102
        assert by_id[103].superseded_by is not None and by_id[103].superseded_by.memory_id == 104
        assert by_id[105].superseded_by is None
