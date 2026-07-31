"""Origin allow-list for the local server.

The HTTP surface (capture, idempotency, reindex, search, the popup) is covered by the
Playwright suite in `tests-e2e/`, which exercises the real extension against the real
server. What remains here is the one thing that suite cannot easily reach: the exact
boundary of which browser origins may read the server's responses.

That boundary is security-critical rather than cosmetic. The server holds the user's entire
conversation history on a loopback port with no authentication, so a permissive
`Access-Control-Allow-Origin` would let any site they happen to visit read all of it. These
are pure-function checks — no server, no model, no measurable runtime.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import closing
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from convsearch.config.settings import Settings, database_path
from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.server.app import _allowed_origin, make_handler
from convsearch.storage.database import connect
from tests.conftest import index_with_test_embeddings


@pytest.mark.parametrize(
    "origin",
    [
        "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
        "moz-extension://11111111-2222-3333-4444-555555555555",
        "https://chatgpt.com",
        "https://chat.openai.com",
    ],
)
def test_permitted_origins_are_echoed_back(origin: str) -> None:
    assert _allowed_origin(origin) == origin


@pytest.mark.parametrize(
    "origin",
    [
        # A lookalike host must not pass a prefix/suffix comparison.
        "https://chatgpt.com.evil.example",
        "https://evil.example/chatgpt.com",
        "https://notchatgpt.com",
        # Scheme downgrade on an otherwise permitted host.
        "http://chatgpt.com",
        # Any ordinary web page the user might have open.
        "https://evil.example",
        "null",
        # No Origin header at all.
        None,
    ],
)
def test_other_origins_are_refused(origin: str | None) -> None:
    assert _allowed_origin(origin) is None


def test_wildcard_is_never_returned() -> None:
    """A `*` would expose the whole history to every site the user visits."""
    assert _allowed_origin("*") is None


# ---------------------------------------------------------------------------
# End-to-end HTTP tests for the JSON API.
#
# These spin up the real handler over a loopback ThreadingHTTPServer on an
# ephemeral port, seeded with a deterministic (network-free) embedding provider
# and hand-inserted memory rows, so every new endpoint can be exercised without
# a model download or any cloud call.
# ---------------------------------------------------------------------------


def _seed_memories(workspace: Path) -> None:
    """Insert one project's worth of memories against the imported conversation."""
    with closing(connect(database_path(workspace))) as conn, conn:
        cid = conn.execute("SELECT conversation_id FROM conversations LIMIT 1").fetchone()[0]
        mid = conn.execute(
            "SELECT message_id FROM messages WHERE conversation_id = ? LIMIT 1", (cid,)
        ).fetchone()[0]

        def add_memory(
            kind: str,
            subject: str,
            statement: str,
            status: str,
            confidence: float,
            task_state: str | None,
            created_at: str,
            content_hash: str,
        ) -> int:
            cur = conn.execute(
                "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
                " task_state, conversation_id, message_id, created_at, extraction_version,"
                " content_hash, metadata_json)"
                " VALUES (?, ?, ?, ?, ?, 'Alpha', ?, ?, ?, ?, 'v1', ?, '{}')",
                (
                    kind,
                    subject,
                    statement,
                    status,
                    confidence,
                    task_state,
                    cid,
                    mid,
                    created_at,
                    content_hash,
                ),
            )
            new_id = int(cur.lastrowid or 0)
            # memory_fts has no trigger; the store populates it by hand, so tests must too.
            conn.execute(
                "INSERT INTO memory_fts(rowid, statement, kind, project, status)"
                " VALUES (?, ?, ?, 'Alpha', ?)",
                (new_id, statement, kind, status),
            )
            return new_id

        memory_id = add_memory(
            "decision",
            "arch/store",
            "Use SQLite for local storage",
            "active",
            0.9,
            None,
            "2024-01-01",
            "hash-mem-1",
        )
        add_memory(
            "task",
            "task/index",
            "Build the FAISS index",
            "active",
            0.8,
            "open",
            "2024-01-02",
            "hash-mem-2",
        )
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, passage_id, message_id, quote,"
            " start_offset, end_offset) VALUES (?, NULL, ?, 'SQLite chosen', 0, 13)",
            (memory_id, mid),
        )
        conn.execute(
            "INSERT INTO memory_status_history(memory_id, old_status, new_status, reason,"
            " changed_at) VALUES (?, 'proposed', 'active', 'confirmed', '2024-01-03')",
            (memory_id,),
        )


def _seed_captured_provenance_task(workspace: Path) -> None:
    """Insert a task memory whose only date is capture time, no real creation date.

    The conversation has `created_at = NULL` (only `updated_at`, the capture timestamp,
    is set) and the memory itself has `created_at = NULL` too, so the fallback chain in
    `convsearch.utils.memory_effective_timestamp_source_sql` has nothing to resolve to
    "created" and must report "captured" -- proving `date_source` reflects real
    provenance rather than a hardcoded default.
    """
    with closing(connect(database_path(workspace))) as conn, conn:
        import_id = conn.execute("SELECT import_id FROM conversations LIMIT 1").fetchone()[0]
        cur = conn.execute(
            "INSERT INTO conversations(source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES ('captured-conv-1', ?, 'Captured Convo', NULL,"
            " '2024-06-01', 'hash-captured-conv')",
            (import_id,),
        )
        captured_cid = int(cur.lastrowid or 0)
        cur = conn.execute(
            "INSERT INTO messages(source_message_id, conversation_id, parent_message_id, role,"
            " created_at, source_order, is_primary_path, text, content_hash) VALUES"
            " ('captured-msg-1', ?, NULL, 'user', NULL, 0, 1, 'SQLite capture note',"
            " 'hash-captured-msg')",
            (captured_cid,),
        )
        captured_mid = int(cur.lastrowid or 0)
        cur = conn.execute(
            "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
            " task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json) VALUES ('task', 'task/captured',"
            " 'Investigate SQLite capture-time task', 'active', 0.85, 'Alpha', 'open', ?, ?,"
            " NULL, 'v1', 'hash-mem-captured', '{}')",
            (captured_cid, captured_mid),
        )
        captured_memory_id = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO memory_fts(rowid, statement, kind, project, status) VALUES"
            " (?, 'Investigate SQLite capture-time task', 'task', 'Alpha', 'active')",
            (captured_memory_id,),
        )


@pytest.fixture
def live_server(
    workspace: Path, settings: Settings, export_zip: Path
) -> Iterator[tuple[str, dict[str, int]]]:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    _seed_memories(workspace)
    with closing(connect(database_path(workspace))) as conn:
        cid = conn.execute("SELECT conversation_id FROM conversations LIMIT 1").fetchone()[0]
        memory_id = conn.execute(
            "SELECT memory_id FROM memories WHERE content_hash = 'hash-mem-1'"
        ).fetchone()[0]
        task_memory_id = conn.execute(
            "SELECT memory_id FROM memories WHERE content_hash = 'hash-mem-2'"
        ).fetchone()[0]
    handler = make_handler(workspace, settings, DeterministicEmbeddingProvider)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield (
            f"http://127.0.0.1:{port}",
            {
                "conversation_id": cid,
                "memory_id": memory_id,
                "task_memory_id": task_memory_id,
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(base: str, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(base + path, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(base: str, path: str, body: Any) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_search_default_stays_byte_compatible(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _get(base, "/search?q=SQLite")
    assert status == 200
    assert set(data) == {"query", "profile", "count", "results"}
    # The default conversation response must not grow a `level` key.
    assert "level" not in data


def test_search_segment_level(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/search?q=SQLite&level=segment")
    # segment_search relies on bm25() over a joined FTS query, which some SQLite builds
    # refuse ("unable to use function bm25 in the requested context"). The endpoint must
    # surface that as a graceful 503 rather than crashing; when the engine supports it we
    # get the documented 200 shape.
    assert status in (200, 503)
    if status != 200:
        return
    assert data["level"] == "segment"
    assert data["count"] == len(data["results"])
    for result in data["results"]:
        assert {
            "segment_id",
            "conversation_id",
            "conversation_title",
            "title",
            "score",
            "passages",
        } <= set(result)


def test_search_passage_level(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/search?q=SQLite&level=passage")
    assert status == 200
    assert data["level"] == "passage"
    assert data["count"] == len(data["results"])
    assert data["results"], "SQLite is in the indexed corpus"
    top = data["results"][0]
    assert "passage_id" in top
    assert "score" in top


def test_search_explain(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/search?q=SQLite&explain=1")
    assert status == 200
    assert data["results"]
    result = data["results"][0]
    assert isinstance(result["reason"], str) and result["reason"]
    assert result["passages"]
    explain = result["passages"][0]["explain"]
    assert isinstance(explain, dict)
    assert "channels" in explain


def test_memories_list(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/memories")
    assert status == 200
    assert data["count"] >= 2
    keys = {
        "memory_id",
        "statement",
        "kind",
        "status",
        "project",
        "subject_key",
        "confidence",
        "conversation_title",
        "created_at",
    }
    assert keys <= set(data["memories"][0])


def test_memories_search(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/memories?q=SQLite")
    assert status == 200
    assert data["query"] == "SQLite"
    assert any("SQLite" in m["statement"] for m in data["memories"])


def test_memory_detail(live_server: tuple[str, dict[str, int]]) -> None:
    base, ids = live_server
    status, data = _get(base, f"/memories/{ids['memory_id']}")
    assert status == 200
    assert data["memory_id"] == ids["memory_id"]
    assert isinstance(data["evidence"], list) and data["evidence"]
    assert isinstance(data["relations"], list)
    assert isinstance(data["status_history"], list) and data["status_history"]
    assert data["status_history"][0]["new_status"] == "active"


def test_memory_detail_missing_is_404(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, _data = _get(base, "/memories/999999")
    assert status == 404


def test_projects_list(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/projects")
    assert status == 200
    assert data["count"] >= 1
    names = {p["name"] for p in data["projects"]}
    assert "Alpha" in names


def test_project_detail(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/projects/Alpha")
    assert status == 200
    assert data["name"] == "Alpha"
    for key in (
        "decisions",
        "open_tasks",
        "timeline",
        "conversations",
        "known_bugs",
        "next_milestones",
    ):
        assert key in data
    assert data["known_bugs"] == []
    assert data["next_milestones"] == []


def test_project_detail_missing_is_404(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, _data = _get(base, "/projects/DoesNotExist")
    assert status == 404


def test_conversation_detail(live_server: tuple[str, dict[str, int]]) -> None:
    base, ids = live_server
    status, data = _get(base, f"/conversation/{ids['conversation_id']}")
    assert status == 200
    assert data["conversation_id"] == ids["conversation_id"]
    assert data["source_conversation_id"]
    assert data["url"] and data["url"].startswith("https://chatgpt.com/c/")
    assert data["messages"]
    msg = data["messages"][0]
    assert {"message_id", "role", "text", "created_at", "is_primary_path", "source_order"} <= set(
        msg
    )


def test_conversation_missing_is_404(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, _data = _get(base, "/conversation/999999")
    assert status == 404


def test_ask_without_matches_returns_no_answer(
    live_server: tuple[str, dict[str, int]],
) -> None:
    """With zero passages requested there are no sources, so /ask answers without an LLM."""
    base, _ = live_server
    status, data = _get(base, "/ask?q=SQLite&passages=0")
    assert status == 200
    assert data["backend"] == "none"
    assert data["sources"] == []
    assert set(data) == {"question", "answer", "backend", "model", "sources"}


def test_ask_with_matches_is_graceful_without_llm(
    live_server: tuple[str, dict[str, int]],
) -> None:
    """With sources but no local/cloud LLM, /ask must 503 (never crash the server)."""
    base, _ = live_server
    status, data = _get(base, "/ask?q=SQLite")
    # 200 if an Ollama happens to be running locally, 503 otherwise — either is graceful.
    assert status in (200, 503)
    if status == 503:
        assert "error" in data and "detail" in data


# ---------------------------------------------------------------------------
# Interaction logging, suggestions, and learned click-boost.
# ---------------------------------------------------------------------------


def test_feedback_records_event_and_learn_stats_reflects_it(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, ids = live_server
    before_status, before = _get(base, "/learn/stats")
    assert before_status == 200
    assert before["stats"]["open"] == 0

    status, data = _post(
        base,
        "/feedback",
        {"event_type": "open", "query": "SQLite", "conversation_id": ids["conversation_id"]},
    )
    assert status == 200
    assert data["ok"] is True
    assert isinstance(data["event_id"], int) and data["event_id"] > 0

    after_status, after = _get(base, "/learn/stats")
    assert after_status == 200
    assert after["stats"]["open"] == 1
    assert after["stats"]["total"] == before["stats"]["total"] + 1


def test_feedback_rejects_bad_event_type(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _post(base, "/feedback", {"event_type": "bogus"})
    assert status == 400
    assert "error" in data


def test_suggestions_reports_logged_searches(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    for _ in range(2):
        status, _data = _post(base, "/feedback", {"event_type": "search", "query": "vectors"})
        assert status == 200
    status, _data = _post(base, "/feedback", {"event_type": "search", "query": "sqlite"})
    assert status == 200

    status, data = _get(base, "/suggestions")
    assert status == 200
    assert set(data) == {"recent", "popular"}
    assert "vectors" in data["recent"]
    assert "sqlite" in data["recent"]
    popular = dict(data["popular"])
    assert popular.get("vectors") == 2


def test_suggestions_respects_limit(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    for query in ("alpha", "beta", "gamma"):
        status, _data = _post(base, "/feedback", {"event_type": "search", "query": query})
        assert status == 200
    status, data = _get(base, "/suggestions?limit=1")
    assert status == 200
    assert len(data["recent"]) <= 1
    assert len(data["popular"]) <= 1


def test_search_boost_still_returns_documented_shape(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, ids = live_server
    # No interactions yet: boost=1 must be a no-op that keeps the documented shape.
    status, data = _get(base, "/search?q=SQLite&boost=1")
    assert status == 200
    assert set(data) == {"query", "profile", "count", "results"}

    # Log an open on a query overlapping the search, then confirm search still returns
    # the same shape (and does not crash) with a real boost signal present.
    _post(
        base,
        "/feedback",
        {"event_type": "open", "query": "SQLite", "conversation_id": ids["conversation_id"]},
    )
    status, data = _get(base, "/search?q=SQLite&boost=1")
    assert status == 200
    assert set(data) == {"query", "profile", "count", "results"}

    # boost=0 must still work and keep the shape.
    status, data = _get(base, "/search?q=SQLite&boost=0")
    assert status == 200
    assert set(data) == {"query", "profile", "count", "results"}


# ---------------------------------------------------------------------------
# Planner and self-improvement "learn" endpoints.
# ---------------------------------------------------------------------------


def test_plan_returns_intent_answer_and_evidence(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _get(base, "/plan?q=what+did+we+decide+about+storage")
    assert status == 200
    assert {"query", "intent", "answer", "steps", "calls", "findings"} <= set(data)
    assert isinstance(data["intent"], str) and data["intent"]
    assert isinstance(data["answer"], str)
    assert isinstance(data["steps"], list) and data["steps"]
    for step in data["steps"]:
        assert {"order", "tool", "rationale"} <= set(step)
    assert isinstance(data["calls"], list)
    for call in data["calls"]:
        assert {"tool", "result_count", "result_summary"} <= set(call)
    assert isinstance(data["findings"], list)


def test_plan_missing_query_is_400(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _get(base, "/plan")
    assert status == 400
    assert "error" in data


def test_learn_offline_is_deterministic(
    live_server: tuple[str, dict[str, int]],
) -> None:
    """use_llm=false keeps the job network-free: a deterministic heuristic summary."""
    base, _ = live_server
    # Log a couple of searches so there is signal to learn from.
    for _ in range(2):
        status, _data = _post(base, "/feedback", {"event_type": "search", "query": "vectors"})
        assert status == 200

    status, data = _post(base, "/learn", {"use_llm": False})
    assert status == 200
    assert set(data) == {"events_read", "notes_written", "backend", "model", "notes"}
    assert data["events_read"] >= 2
    assert data["backend"] == "none"
    assert isinstance(data["notes"], list)
    assert data["notes_written"] == len(data["notes"])


def test_learn_rejects_non_object_body(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _post(base, "/learn", [1, 2, 3])
    assert status == 400
    assert "error" in data


def test_learn_preferences_lists_written_notes(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    # Empty is valid before any learn run.
    status, data = _get(base, "/learn/preferences")
    assert status == 200
    assert set(data) == {"preferences"}
    assert isinstance(data["preferences"], list)

    # After a learn run over logged searches, notes should surface here.
    _post(base, "/feedback", {"event_type": "search", "query": "vectors"})
    _post(base, "/feedback", {"event_type": "search", "query": "vectors"})
    status, learned = _post(base, "/learn", {"use_llm": False})
    assert status == 200

    status, data = _get(base, "/learn/preferences")
    assert status == 200
    if learned["notes_written"]:
        assert data["preferences"]
        pref = data["preferences"][0]
        assert {"pref_id", "note", "weight", "created_at"} <= set(pref)


# ---------------------------------------------------------------------------
# Tasks, timeline, captures, project export, and diagnostics endpoints.
# ---------------------------------------------------------------------------


def test_tasks_default_open_shape(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/tasks")
    assert status == 200
    assert {"total_open", "total_completed", "projects", "count", "items"} <= set(data)
    assert data["count"] >= 1
    item = data["items"][0]
    assert {
        "memory_id",
        "statement",
        "project",
        "task_state",
        "status",
        "confidence",
        "created_at",
        "date_source",
        "conversation_id",
        "conversation_title",
        "has_evidence",
        "evidence",
    } <= set(item)
    assert item["task_state"] == "open"
    assert item["date_source"] in ("created", "captured", "unknown")


def test_tasks_date_source_reflects_capture_time_fallback(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, _ = live_server
    _seed_captured_provenance_task(workspace)
    status, data = _get(base, "/tasks?state=all")
    assert status == 200
    captured_items = [item for item in data["items"] if item["statement"].startswith("Investigate")]
    assert captured_items
    assert captured_items[0]["date_source"] == "captured"


def test_tasks_since_rejects_bad_duration(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/tasks?since=nonsense")
    assert status == 400
    assert "error" in data


def test_tasks_since_accepts_compact_duration(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    # The seeded memories are dated 2024-01-xx, well outside the last 7 days, so a valid
    # duration is expected to filter them out (count 0) rather than error.
    status, data = _get(base, "/tasks?since=7d&state=all")
    assert status == 200
    assert data["count"] == 0

    status, data = _get(base, "/tasks?since=9999d&state=all")
    assert status == 200
    assert data["count"] >= 1


def test_tasks_unknown_state_is_400(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/tasks?state=bogus")
    assert status == 400
    assert "error" in data


def test_timeline_returns_nodes_and_evidence(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/timeline?q=SQLite&evidence=1")
    assert status == 200
    assert {
        "topic",
        "matched_count",
        "first_seen",
        "first_seen_source",
        "last_seen",
        "last_seen_source",
        "nodes",
        "active",
        "superseded",
        "contested",
        "rejected",
    } <= set(data)
    assert data["nodes"]
    node = data["nodes"][0]
    assert {
        "memory_id",
        "kind",
        "statement",
        "status",
        "project",
        "created_at",
        "date_source",
        "confidence",
        "conversation_id",
        "conversation_title",
        "supersedes",
        "superseded_by",
        "conflicts_with",
        "reasons",
        "evidence",
    } <= set(node)
    assert node["date_source"] in ("created", "captured", "unknown")


def test_timeline_missing_query_is_400(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/timeline")
    assert status == 400
    assert "error" in data


def test_timeline_date_source_reflects_capture_time_fallback(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, _ = live_server
    _seed_captured_provenance_task(workspace)
    status, data = _get(base, "/timeline?q=Investigate")
    assert status == 200
    assert data["nodes"]
    node = data["nodes"][0]
    assert node["date_source"] == "captured"
    assert data["first_seen_source"] in ("created", "captured", "unknown", None)
    assert data["last_seen_source"] in ("created", "captured", "unknown", None)


def test_captures_default_shape(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/captures")
    assert status == 200
    assert {
        "total",
        "live_captured",
        "imported",
        "not_indexed",
        "not_segmented",
        "stale_index",
        "count",
        "items",
    } <= set(data)
    assert data["count"] >= 1
    item = data["items"][0]
    assert {
        "conversation_id",
        "source_conversation_id",
        "title",
        "captured_at",
        "date_source",
        "updated_at",
        "message_count",
        "source",
        "indexed",
        "segmented",
        "memories_extracted",
        "passage_count",
        "memory_count",
        "source_url",
        "warnings",
    } <= set(item)
    assert item["date_source"] in ("created", "captured", "imported", "unknown")


def test_captures_unknown_source_is_400(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/captures?source=bogus")
    assert status == 400
    assert "error" in data


def test_captures_date_source_reflects_capture_time_fallback(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    """A conversation with `created_at = NULL` and a real `updated_at` must resolve to
    'captured' (not 'unknown', not silently treated as a creation date)."""
    base, _ = live_server
    _seed_captured_provenance_task(workspace)
    status, data = _get(base, "/captures")
    assert status == 200
    captured_items = [item for item in data["items"] if item["title"] == "Captured Convo"]
    assert captured_items
    assert captured_items[0]["date_source"] == "captured"
    assert captured_items[0]["captured_at"] == "2024-06-01"


def test_project_export_returns_markdown_json(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/projects/Alpha/export")
    assert status == 200
    assert set(data) == {"name", "markdown"}
    assert data["name"] == "Alpha"
    assert isinstance(data["markdown"], str) and data["markdown"].startswith("# Alpha")


def test_project_export_missing_is_404(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/projects/DoesNotExist/export")
    assert status == 404
    assert "error" in data


def test_diagnostics_returns_readiness_and_doctor_checks(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _get(base, "/diagnostics")
    assert status == 200
    assert {"ready", "backend", "summary", "remediation", "llm_checks", "doctor_checks"} <= set(
        data
    )
    assert isinstance(data["llm_checks"], list) and data["llm_checks"]
    assert isinstance(data["doctor_checks"], list) and data["doctor_checks"]
    for check in data["llm_checks"]:
        assert {"name", "ok", "detail"} <= set(check)


def test_privacy_shape_and_loopback(live_server: tuple[str, dict[str, int]]) -> None:
    base, _ = live_server
    status, data = _get(base, "/privacy")
    assert status == 200
    assert set(data) == {
        "local_only",
        "workspace_path",
        "database_path",
        "index_path",
        "server_bind",
        "loopback_only",
        "llm",
        "cloud_payload_note",
        "counts",
    }
    assert isinstance(data["local_only"], bool)
    # The test server always binds to 127.0.0.1 (see the `live_server` fixture above).
    assert data["loopback_only"] is True
    assert data["server_bind"].startswith("127.0.0.1:")
    for key in ("workspace_path", "database_path", "index_path"):
        assert Path(data[key]).is_absolute()
    llm = data["llm"]
    assert {
        "backend_mode",
        "effective_backend",
        "ollama_host",
        "cloud_configured",
        "cloud_would_be_used",
    } <= set(llm)
    assert isinstance(llm["cloud_configured"], bool)
    assert isinstance(llm["cloud_would_be_used"], bool)
    counts = data["counts"]
    assert {"conversations", "messages", "memories"} <= set(counts)
    assert counts["conversations"] >= 1
    assert counts["messages"] >= 1


def test_privacy_never_leaks_the_api_key(
    live_server: tuple[str, dict[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sentinel key value must never appear anywhere in the serialised response."""
    sentinel = "sk-ant-do-not-leak-this-value-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    base, _ = live_server
    status, data = _get(base, "/privacy")
    assert status == 200
    raw = json.dumps(data)
    assert sentinel not in raw
    assert isinstance(data["llm"]["cloud_configured"], bool)
    assert data["llm"]["cloud_configured"] is True


def test_privacy_local_only_and_cloud_flags_are_consistent(
    live_server: tuple[str, dict[str, int]],
) -> None:
    base, _ = live_server
    status, data = _get(base, "/privacy")
    assert status == 200
    local_only = data["local_only"]
    cloud_would_be_used = data["llm"]["cloud_would_be_used"]
    # These must never both claim "cloud" and "local only" at the same time.
    assert not (local_only and cloud_would_be_used)


# ---------------------------------------------------------------------------
# Review queue endpoints: GET /memories/review, POST /memories/{id}/confirm,
# POST /memories/{id}/invalidate, POST /memories/{id}/pin.
# ---------------------------------------------------------------------------


def _add_review_candidate(workspace: Path) -> int:
    """Insert a low-confidence 'proposed' memory so the review queue is non-empty.

    Both memories in `_seed_memories` are 'active' with confidence above the review
    threshold and unpinned/unreviewed, so neither qualifies for the pending queue.
    """
    with closing(connect(database_path(workspace))) as conn, conn:
        cid = conn.execute("SELECT conversation_id FROM conversations LIMIT 1").fetchone()[0]
        mid = conn.execute(
            "SELECT message_id FROM messages WHERE conversation_id = ? LIMIT 1", (cid,)
        ).fetchone()[0]
        cur = conn.execute(
            "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
            " task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES ('preference', 'pref/theme', 'Prefers dark mode', 'proposed', 0.7,"
            " 'Alpha', NULL, ?, ?, '2024-01-04', 'v1', 'hash-mem-review', '{}')",
            (cid, mid),
        )
        new_id = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO memory_fts(rowid, statement, kind, project, status)"
            " VALUES (?, 'Prefers dark mode', 'preference', 'Alpha', 'proposed')",
            (new_id,),
        )
        return new_id


def _fresh_memory_row(workspace: Path, memory_id: int) -> sqlite3.Row:
    """Re-read a memory row on a brand-new connection, to prove a write really persisted."""
    with closing(connect(database_path(workspace))) as conn:
        row = conn.execute(
            "SELECT status, pinned, reviewed_at FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
    assert row is not None
    return row


def test_memories_review_shape_and_not_shadowed_by_detail(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    review_id = _add_review_candidate(workspace)
    status, data = _get(base, "/memories/review")
    assert status == 200
    assert {
        "total_pending",
        "total_pinned",
        "total_contested",
        "total_invalidated",
        "count",
        "items",
    } == set(data)
    assert data["count"] == len(data["items"])
    assert data["total_pending"] >= 1
    ids_seen = {item["memory_id"] for item in data["items"]}
    assert review_id in ids_seen
    # The 'active', high-confidence, unconflicted seed memory must NOT be pending.
    assert ids["memory_id"] not in ids_seen
    item = next(i for i in data["items"] if i["memory_id"] == review_id)
    assert {
        "memory_id",
        "kind",
        "statement",
        "status",
        "project",
        "confidence",
        "created_at",
        "pinned",
        "reviewed_at",
        "conversation_id",
        "conversation_title",
        "evidence",
        "conflicts",
        "superseded_by",
        "review_reason",
    } == set(item)
    assert item["pinned"] is False
    assert item["reviewed_at"] is None
    assert isinstance(item["review_reason"], str) and item["review_reason"]


def test_memories_review_is_not_parsed_as_a_memory_id(
    live_server: tuple[str, dict[str, int]],
) -> None:
    """`/memories/review` must route to the review queue, never `/memories/{id}`."""
    base, _ = live_server
    status, data = _get(base, "/memories/review")
    assert status == 200
    assert "items" in data
    assert "memory_id" not in data


def test_memory_confirm_persists(live_server: tuple[str, dict[str, int]], workspace: Path) -> None:
    base, _ids = live_server
    review_id = _add_review_candidate(workspace)
    status, data = _post(base, f"/memories/{review_id}/confirm", {"reason": "looks right"})
    assert status == 200
    assert data["memory_id"] == review_id
    assert data["status"] == "active"
    assert data["reviewed_at"] is not None
    row = _fresh_memory_row(workspace, review_id)
    assert row["status"] == "active"
    assert row["reviewed_at"] is not None
    # The audit trail is the point of routing through set_memory_status.
    with closing(connect(database_path(workspace))) as conn:
        history = conn.execute(
            "SELECT new_status, reason FROM memory_status_history WHERE memory_id = ?"
            " ORDER BY history_id DESC LIMIT 1",
            (review_id,),
        ).fetchone()
    assert history["new_status"] == "active"
    assert history["reason"] == "looks right"


def test_memory_invalidate_persists(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, _ = live_server
    review_id = _add_review_candidate(workspace)
    status, data = _post(base, f"/memories/{review_id}/invalidate", {})
    assert status == 200
    assert data["memory_id"] == review_id
    assert data["status"] == "invalidated"
    row = _fresh_memory_row(workspace, review_id)
    assert row["status"] == "invalidated"
    assert row["reviewed_at"] is not None


def test_memory_pin_round_trips_and_persists(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    memory_id = ids["memory_id"]
    status, data = _post(base, f"/memories/{memory_id}/pin", {"pinned": True})
    assert status == 200
    assert data["pinned"] is True
    row = _fresh_memory_row(workspace, memory_id)
    assert bool(row["pinned"]) is True

    status, data = _post(base, f"/memories/{memory_id}/pin", {"pinned": False, "reason": "meh"})
    assert status == 200
    assert data["pinned"] is False
    row = _fresh_memory_row(workspace, memory_id)
    assert bool(row["pinned"]) is False


@pytest.mark.parametrize("route", ["confirm", "invalidate", "pin"])
def test_memory_mutations_404_on_unknown_id(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    base, _ = live_server
    body: dict[str, Any] = {"pinned": True} if route == "pin" else {}
    status, data = _post(base, f"/memories/999999/{route}", body)
    assert status == 404
    assert "error" in data


@pytest.mark.parametrize("route", ["confirm", "invalidate", "pin"])
def test_memory_mutations_400_on_non_integer_id(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    base, _ = live_server
    body: dict[str, Any] = {"pinned": True} if route == "pin" else {}
    status, data = _post(base, f"/memories/not-a-number/{route}", body)
    assert status == 400
    assert "error" in data


def test_memory_pin_400_on_missing_pinned(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    status, data = _post(base, f"/memories/{ids['memory_id']}/pin", {})
    assert status == 400
    assert "error" in data


def test_memory_pin_400_on_non_boolean_pinned(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    status, data = _post(base, f"/memories/{ids['memory_id']}/pin", {"pinned": "yes"})
    assert status == 400
    assert "error" in data


@pytest.mark.parametrize("route", ["confirm", "invalidate", "pin"])
def test_memory_mutations_400_on_malformed_json_body(
    live_server: tuple[str, dict[str, int]], workspace: Path, route: str
) -> None:
    base, ids = live_server
    memory_id = ids["memory_id"]
    req = urllib.request.Request(
        f"{base}/memories/{memory_id}/{route}",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, data = resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status, data = exc.code, json.loads(exc.read().decode("utf-8"))
    assert status == 400
    assert "error" in data


# ---------------------------------------------------------------------------
# Task completion endpoints: POST /tasks/{id}/complete, POST /tasks/{id}/reopen.
# ---------------------------------------------------------------------------


def _fresh_task_state(workspace: Path, memory_id: int) -> sqlite3.Row:
    """Re-read task_state on a brand-new connection, to prove a write really persisted."""
    with closing(connect(database_path(workspace))) as conn:
        row = conn.execute(
            "SELECT task_state, task_state_changed_at FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
    assert row is not None
    return row


def test_task_complete_persists_and_returns_updated_item(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    task_id = ids["task_memory_id"]
    status, data = _post(base, f"/tasks/{task_id}/complete", {"reason": "shipped"})
    assert status == 200
    assert data["memory_id"] == task_id
    assert data["task_state"] == "completed"
    assert data["task_state_source"] == "user"
    assert data["task_state_changed_at"] is not None

    row = _fresh_task_state(workspace, task_id)
    assert row["task_state"] == "completed"
    assert row["task_state_changed_at"] is not None

    with closing(connect(database_path(workspace))) as conn:
        history = conn.execute(
            "SELECT old_state, new_state, reason FROM task_state_history WHERE memory_id = ?"
            " ORDER BY history_id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert history["old_state"] == "open"
    assert history["new_state"] == "completed"
    assert history["reason"] == "shipped"


def test_task_reopen_after_complete_round_trips(
    live_server: tuple[str, dict[str, int]], workspace: Path
) -> None:
    base, ids = live_server
    task_id = ids["task_memory_id"]
    _post(base, f"/tasks/{task_id}/complete", {})
    status, data = _post(base, f"/tasks/{task_id}/reopen", {"reason": "not done yet"})
    assert status == 200
    assert data["task_state"] == "open"
    row = _fresh_task_state(workspace, task_id)
    assert row["task_state"] == "open"


@pytest.mark.parametrize("route", ["complete", "reopen"])
def test_task_mutations_404_on_unknown_id(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    base, _ = live_server
    status, data = _post(base, f"/tasks/999999/{route}", {})
    assert status == 404
    assert "error" in data


@pytest.mark.parametrize("route", ["complete", "reopen"])
def test_task_mutations_404_on_non_task_memory(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    """The decision seed memory (hash-mem-1) is kind='decision', not a task."""
    base, ids = live_server
    status, data = _post(base, f"/tasks/{ids['memory_id']}/{route}", {})
    assert status == 404
    assert "error" in data


@pytest.mark.parametrize("route", ["complete", "reopen"])
def test_task_mutations_400_on_non_integer_id(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    base, _ = live_server
    status, data = _post(base, f"/tasks/not-a-number/{route}", {})
    assert status == 400
    assert "error" in data


@pytest.mark.parametrize("route", ["complete", "reopen"])
def test_task_mutations_400_on_malformed_json_body(
    live_server: tuple[str, dict[str, int]], route: str
) -> None:
    base, ids = live_server
    task_id = ids["task_memory_id"]
    req = urllib.request.Request(
        f"{base}/tasks/{task_id}/{route}",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, data = resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status, data = exc.code, json.loads(exc.read().decode("utf-8"))
    assert status == 400
    assert "error" in data
