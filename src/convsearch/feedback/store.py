from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

from convsearch.domain.models import ConversationResult
from convsearch.feedback.models import InteractionEvent

# Maximum boost any single conversation can receive from prior clicks.
_MAX_BOOST = 0.5
# Boost contributed per token-overlapping prior click, before capping.
_BOOST_PER_CLICK = 0.1

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _tokens(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric runs, drop empties."""
    return {tok for tok in _TOKEN_RE.split(text.lower()) if tok}


def record_event(conn: sqlite3.Connection, event: InteractionEvent) -> int:
    """Insert an interaction event, filling created_at with UTC ISO now when None.

    Returns the new event_id. Commits.
    """
    created_at = event.created_at if event.created_at is not None else _now_iso()
    cursor = conn.execute(
        "INSERT INTO interactions("
        "created_at, event_type, query, conversation_id, passage_id, segment_id, position"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            created_at,
            event.event_type,
            event.query,
            event.conversation_id,
            event.passage_id,
            event.segment_id,
            event.position,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def recent_queries(conn: sqlite3.Connection, limit: int = 10) -> list[str]:
    """Distinct non-empty search queries, most-recent first."""
    rows = conn.execute(
        "SELECT query, MAX(event_id) AS last_id FROM interactions "
        "WHERE event_type = 'search' AND query <> '' "
        "GROUP BY query ORDER BY last_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [row[0] for row in rows]


def popular_queries(conn: sqlite3.Connection, limit: int = 10) -> list[tuple[str, int]]:
    """(query, count) over 'search' events, most frequent first."""
    rows = conn.execute(
        "SELECT query, COUNT(*) AS n FROM interactions "
        "WHERE event_type = 'search' AND query <> '' "
        "GROUP BY query ORDER BY n DESC, query ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def click_boosts(conn: sqlite3.Connection, query: str) -> dict[int, float]:
    """Map conversation_id -> boost in [0, ~0.5] from prior click events.

    Considers prior 'open'/'inspect' events whose logged query shares at least one
    token with `query` (case-insensitive token overlap). Each overlapping prior click
    for a conversation adds `_BOOST_PER_CLICK`, capped at `_MAX_BOOST`. Deterministic;
    returns an empty dict when there is no signal.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return {}
    rows = conn.execute(
        "SELECT conversation_id, query FROM interactions "
        "WHERE event_type IN ('open', 'inspect') AND conversation_id IS NOT NULL"
    ).fetchall()
    counts: dict[int, int] = {}
    for row in rows:
        if not query_tokens & _tokens(row[1] or ""):
            continue
        conv_id = int(row[0])
        counts[conv_id] = counts.get(conv_id, 0) + 1
    return {conv_id: min(_MAX_BOOST, _BOOST_PER_CLICK * count) for conv_id, count in counts.items()}


def apply_click_boost(
    results: list[ConversationResult], query: str, conn: sqlite3.Connection
) -> list[ConversationResult]:
    """Return a NEW list with each result's score boosted by prior clicks, re-sorted desc.

    Never mutates inputs. When no boosts apply, returns a new list sorted by score.
    """
    boosts = click_boosts(conn, query)
    boosted = [
        replace(result, score=result.score + boosts.get(result.conversation_id, 0.0))
        for result in results
    ]
    boosted.sort(key=lambda result: result.score, reverse=True)
    return boosted


def interaction_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Counts across interaction event types plus distinct queries and learned prefs."""
    stats: dict[str, int] = {
        "total": 0,
        "search": 0,
        "open": 0,
        "inspect": 0,
        "ask": 0,
    }
    for row in conn.execute(
        "SELECT event_type, COUNT(*) AS n FROM interactions GROUP BY event_type"
    ):
        event_type = str(row[0])
        count = int(row[1])
        if event_type in stats:
            stats[event_type] = count
        stats["total"] += count
    stats["distinct_queries"] = int(
        conn.execute("SELECT COUNT(DISTINCT query) FROM interactions WHERE query <> ''").fetchone()[
            0
        ]
    )
    stats["learned_preferences"] = int(
        conn.execute("SELECT COUNT(*) FROM learned_preferences").fetchone()[0]
    )
    return stats


def clear_interactions(conn: sqlite3.Connection) -> int:
    """Delete all interactions (privacy/reset). Returns rows deleted. Commits."""
    cursor = conn.execute("DELETE FROM interactions")
    conn.commit()
    return int(cursor.rowcount)
