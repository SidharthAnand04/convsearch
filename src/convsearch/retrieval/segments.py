from __future__ import annotations

import sqlite3

from convsearch.domain.models import PassageHit, SegmentResult
from convsearch.retrieval.query import build_fts_expressions, parse_query


def segment_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    include_branches: bool = False,
) -> list[SegmentResult]:
    parsed = parse_query(query)
    expressions = build_fts_expressions(parsed)
    results: dict[int, SegmentResult] = {}
    for _level, fts_query in expressions:
        branch_filter = "" if include_branches else "AND m.is_primary_path = 1"
        # bm25() cannot be used together with GROUP BY / aggregation — SQLite rejects it
        # ("unable to use function bm25 in the requested context"). Query flat (one row per
        # passage, like the working passage search) and dedup to one representative passage
        # per segment in Python, keeping the best-ranked row per segment.
        rows = conn.execute(
            f"""
            SELECT s.segment_id, s.conversation_id, c.title AS conversation_title,
                   s.title AS segment_title, bm25(segment_fts) AS rank_score,
                   p.passage_id, p.message_id, m.role, p.text, c.created_at, m.is_primary_path
            FROM segment_fts
            JOIN segments s ON s.segment_id = segment_fts.rowid
            JOIN conversations c ON c.conversation_id = s.conversation_id
            JOIN passages p ON p.segment_id = s.segment_id
            JOIN messages m ON m.message_id = p.message_id
            WHERE segment_fts MATCH ?
            {branch_filter}
            ORDER BY rank_score
            """,
            (fts_query,),
        ).fetchall()
        for row in rows:
            segment_id = int(row["segment_id"])
            if segment_id in results:
                continue
            rank = len(results) + 1
            score = 1.0 / (60.0 + rank)
            hit = PassageHit(
                passage_id=int(row["passage_id"]),
                conversation_id=int(row["conversation_id"]),
                message_id=int(row["message_id"]),
                title=str(row["conversation_title"]),
                role=str(row["role"]),
                text=str(row["text"]),
                created_at=row["created_at"],
                is_primary_path=bool(row["is_primary_path"]),
                lexical_rank=rank,
                lexical_score=float(row["rank_score"]),
                fused_score=score,
                segment_id=segment_id,
                segment_title=row["segment_title"],
                channels=("segment", "lexical"),
            )
            results[segment_id] = SegmentResult(
                segment_id=segment_id,
                conversation_id=int(row["conversation_id"]),
                conversation_title=str(row["conversation_title"]),
                title=row["segment_title"],
                score=score,
                best_passages=[hit],
            )
        if len(results) >= limit:
            break
    return sorted(results.values(), key=lambda result: result.score, reverse=True)[:limit]
