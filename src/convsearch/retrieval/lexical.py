from __future__ import annotations

import sqlite3

from convsearch.config.settings import Settings
from convsearch.domain.models import PassageHit
from convsearch.retrieval.query import build_fts_expressions, parse_query


def lexical_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    include_branches: bool = False,
    settings: Settings | None = None,
) -> list[PassageHit]:
    parsed = parse_query(query)
    minimum = settings.retrieval.lexical_fallback_min_results if settings else 5
    hits: dict[int, PassageHit] = {}
    for fallback_level, fts_query in build_fts_expressions(parsed):
        for hit in _passage_fts_search(
            conn, fts_query, limit, include_branches=include_branches, fallback_level=fallback_level
        ):
            hits.setdefault(hit.passage_id, hit)
        if len(hits) >= min(minimum, limit):
            break
    return list(hits.values())[:limit]


def title_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    include_branches: bool = False,
) -> list[PassageHit]:
    parsed = parse_query(query)
    terms = [*parsed.phrases, *parsed.required_terms, *parsed.identifiers, *parsed.optional_terms]
    if not terms:
        return []
    branch_filter = "" if include_branches else "AND m.is_primary_path = 1"
    predicates = " OR ".join("lower(c.title) LIKE ?" for _ in terms)
    params = [f"%{term.lower()}%" for term in terms]
    rows = conn.execute(
        f"""
        SELECT p.passage_id, p.conversation_id, p.message_id, c.title, m.role, p.text,
               c.created_at, m.is_primary_path, p.segment_id, s.title AS segment_title,
               ({_title_score_sql(len(terms))}) AS title_score
        FROM conversations c
        JOIN passages p ON p.conversation_id = c.conversation_id
        JOIN messages m ON m.message_id = p.message_id
        LEFT JOIN segments s ON s.segment_id = p.segment_id
        WHERE ({predicates})
        {branch_filter}
        GROUP BY c.conversation_id
        ORDER BY title_score DESC, c.updated_at DESC
        LIMIT ?
        """,
        [*params, *params, limit],
    ).fetchall()
    return [
        PassageHit(
            passage_id=int(row["passage_id"]),
            conversation_id=int(row["conversation_id"]),
            message_id=int(row["message_id"]),
            title=str(row["title"]),
            role=str(row["role"]),
            text=str(row["text"]),
            created_at=row["created_at"],
            is_primary_path=bool(row["is_primary_path"]),
            title_rank=index,
            title_score=float(row["title_score"]),
            segment_id=row["segment_id"],
            segment_title=row["segment_title"],
            channels=("title",),
        )
        for index, row in enumerate(rows, start=1)
    ]


def _title_score_sql(term_count: int) -> str:
    return " + ".join(
        "CASE WHEN lower(c.title) LIKE ? THEN 1.0 ELSE 0.0 END" for _ in range(term_count)
    )


def _passage_fts_search(
    conn: sqlite3.Connection,
    fts_query: str,
    limit: int,
    *,
    include_branches: bool,
    fallback_level: str,
) -> list[PassageHit]:
    branch_filter = "" if include_branches else "AND m.is_primary_path = 1"
    rows = conn.execute(
        f"""
        SELECT p.passage_id, p.conversation_id, p.message_id, c.title, m.role, p.text,
               c.created_at, m.is_primary_path, p.segment_id, s.title AS segment_title,
               bm25(passage_fts) AS rank_score
        FROM passage_fts
        JOIN passages p ON p.passage_id = passage_fts.rowid
        JOIN conversations c ON c.conversation_id = p.conversation_id
        JOIN messages m ON m.message_id = p.message_id
        LEFT JOIN segments s ON s.segment_id = p.segment_id
        WHERE passage_fts MATCH ?
        {branch_filter}
        ORDER BY rank_score
        LIMIT ?
        """,
        (fts_query, limit),
    ).fetchall()
    return [
        PassageHit(
            passage_id=int(row["passage_id"]),
            conversation_id=int(row["conversation_id"]),
            message_id=int(row["message_id"]),
            title=str(row["title"]),
            role=str(row["role"]),
            text=str(row["text"]),
            created_at=row["created_at"],
            is_primary_path=bool(row["is_primary_path"]),
            lexical_rank=index,
            lexical_score=float(row["rank_score"]),
            segment_id=row["segment_id"],
            segment_title=row["segment_title"],
            channels=("lexical", f"lexical:{fallback_level}"),
        )
        for index, row in enumerate(rows, start=1)
    ]
