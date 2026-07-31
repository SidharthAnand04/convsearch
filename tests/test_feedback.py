from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from convsearch.config.settings import database_path
from convsearch.domain.models import ConversationResult
from convsearch.feedback import (
    InteractionEvent,
    apply_click_boost,
    clear_interactions,
    click_boosts,
    interaction_stats,
    popular_queries,
    recent_queries,
    record_event,
)
from convsearch.storage.database import connect, initialize_database


@pytest.fixture()
def conn() -> Generator[sqlite3.Connection, None, None]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        initialize_database(workspace)
        connection = connect(database_path(workspace))
        try:
            yield connection
        finally:
            connection.close()


def _result(conversation_id: int, score: float) -> ConversationResult:
    return ConversationResult(
        conversation_id=conversation_id,
        title=f"Conversation {conversation_id}",
        created_at=None,
        updated_at=None,
        score=score,
        best_passages=[],
        distinct_message_count=1,
    )


def test_record_event_returns_id_and_fills_created_at(conn: sqlite3.Connection) -> None:
    event_id = record_event(conn, InteractionEvent(event_type="search", query="sqlite index"))
    assert event_id > 0
    row = conn.execute(
        "SELECT created_at, event_type, query FROM interactions WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    assert row["created_at"] is not None
    assert row["created_at"].endswith("+00:00")
    assert row["event_type"] == "search"
    assert row["query"] == "sqlite index"


def test_record_event_preserves_supplied_created_at(conn: sqlite3.Connection) -> None:
    event_id = record_event(
        conn,
        InteractionEvent(event_type="ask", query="q", created_at="2020-01-01T00:00:00+00:00"),
    )
    row = conn.execute(
        "SELECT created_at FROM interactions WHERE event_id = ?", (event_id,)
    ).fetchone()
    assert row["created_at"] == "2020-01-01T00:00:00+00:00"


def test_recent_queries_ordering_and_distinct(conn: sqlite3.Connection) -> None:
    record_event(conn, InteractionEvent(event_type="search", query="alpha"))
    record_event(conn, InteractionEvent(event_type="search", query="beta"))
    record_event(conn, InteractionEvent(event_type="search", query="alpha"))
    record_event(conn, InteractionEvent(event_type="search", query=""))
    record_event(conn, InteractionEvent(event_type="open", query="gamma", conversation_id=1))
    assert recent_queries(conn) == ["alpha", "beta"]
    assert recent_queries(conn, limit=1) == ["alpha"]


def test_popular_queries_counts(conn: sqlite3.Connection) -> None:
    for _ in range(3):
        record_event(conn, InteractionEvent(event_type="search", query="popular"))
    record_event(conn, InteractionEvent(event_type="search", query="rare"))
    assert popular_queries(conn) == [("popular", 3), ("rare", 1)]


def test_click_boosts_signal_and_no_signal(conn: sqlite3.Connection) -> None:
    # Conversation 7 opened twice under token-overlapping queries.
    record_event(
        conn,
        InteractionEvent(event_type="open", query="vector search sqlite", conversation_id=7),
    )
    record_event(
        conn,
        InteractionEvent(event_type="inspect", query="sqlite tuning", conversation_id=7),
    )
    # Conversation 9 opened under an unrelated query.
    record_event(
        conn, InteractionEvent(event_type="open", query="unrelated topic", conversation_id=9)
    )
    boosts = click_boosts(conn, "sqlite performance")
    assert boosts == {7: pytest.approx(0.2)}
    assert 9 not in boosts
    assert click_boosts(conn, "nothing overlaps here") == {}
    assert click_boosts(conn, "") == {}


def test_click_boosts_capped(conn: sqlite3.Connection) -> None:
    for _ in range(10):
        record_event(conn, InteractionEvent(event_type="open", query="sqlite", conversation_id=3))
    assert click_boosts(conn, "sqlite") == {3: pytest.approx(0.5)}


def test_apply_click_boost_reorders_without_mutation(conn: sqlite3.Connection) -> None:
    record_event(conn, InteractionEvent(event_type="open", query="sqlite index", conversation_id=2))
    record_event(conn, InteractionEvent(event_type="open", query="sqlite index", conversation_id=2))
    results = [_result(1, 0.5), _result(2, 0.4)]
    original_scores = [(r.conversation_id, r.score) for r in results]
    boosted = apply_click_boost(results, "sqlite index", conn)
    # Conversation 2 gains 0.2 -> 0.6, overtaking conversation 1.
    assert [r.conversation_id for r in boosted] == [2, 1]
    assert boosted[0].score == pytest.approx(0.6)
    # Inputs untouched.
    assert [(r.conversation_id, r.score) for r in results] == original_scores
    assert boosted is not results


def test_apply_click_boost_no_signal_returns_sorted(conn: sqlite3.Connection) -> None:
    results = [_result(1, 0.2), _result(2, 0.9)]
    boosted = apply_click_boost(results, "no matching clicks", conn)
    assert [r.conversation_id for r in boosted] == [2, 1]
    assert [r.score for r in boosted] == [0.9, 0.2]


def test_interaction_stats(conn: sqlite3.Connection) -> None:
    record_event(conn, InteractionEvent(event_type="search", query="a"))
    record_event(conn, InteractionEvent(event_type="search", query="b"))
    record_event(conn, InteractionEvent(event_type="open", query="a", conversation_id=1))
    record_event(conn, InteractionEvent(event_type="inspect", query="a", conversation_id=1))
    record_event(conn, InteractionEvent(event_type="ask", query="a"))
    conn.execute(
        "INSERT INTO learned_preferences(created_at, note) VALUES ('2020-01-01', 'likes sqlite')"
    )
    conn.commit()
    stats = interaction_stats(conn)
    assert stats == {
        "total": 5,
        "search": 2,
        "open": 1,
        "inspect": 1,
        "ask": 1,
        "distinct_queries": 2,
        "learned_preferences": 1,
    }


def test_clear_interactions(conn: sqlite3.Connection) -> None:
    record_event(conn, InteractionEvent(event_type="search", query="a"))
    record_event(conn, InteractionEvent(event_type="open", query="a", conversation_id=1))
    deleted = clear_interactions(conn)
    assert deleted == 2
    assert conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] == 0
    assert clear_interactions(conn) == 0
