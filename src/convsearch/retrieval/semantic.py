from __future__ import annotations

import sqlite3
from pathlib import Path

from convsearch.config.settings import Settings
from convsearch.domain.models import PassageHit
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.indexes.vector import search_vector_index

SEMANTIC_OVERFETCH_FACTOR = 4


def semantic_search(
    conn: sqlite3.Connection,
    workspace: Path,
    query: str,
    settings: Settings,
    provider: EmbeddingProvider,
    limit: int,
    *,
    include_branches: bool = False,
    profile: str = "balanced",
) -> list[PassageHit]:
    query_vector = provider.encode_query(query)
    fetch_limit = limit if include_branches else limit * SEMANTIC_OVERFETCH_FACTOR
    vector_hits = search_vector_index(workspace, query_vector, fetch_limit)
    if not vector_hits:
        return []
    semantic_floor = settings.retrieval.semantic_floor(profile)
    placeholders = ",".join("?" for _ in vector_hits)
    rows = conn.execute(
        f"""
        SELECT p.passage_id, p.conversation_id, p.message_id, c.title, m.role, p.text,
               c.created_at, m.is_primary_path, p.segment_id, s.title AS segment_title
        FROM passages p
        JOIN conversations c ON c.conversation_id = p.conversation_id
        JOIN messages m ON m.message_id = p.message_id
        LEFT JOIN segments s ON s.segment_id = p.segment_id
        WHERE p.passage_id IN ({placeholders})
        """,
        [hit.passage_id for hit in vector_hits],
    ).fetchall()
    by_id = {int(row["passage_id"]): row for row in rows}
    hits: list[PassageHit] = []
    for vector_hit in vector_hits:
        if vector_hit.score < semantic_floor:
            continue
        row = by_id.get(vector_hit.passage_id)
        if row is None:
            continue
        is_primary_path = bool(row["is_primary_path"])
        if not include_branches and not is_primary_path:
            continue
        hits.append(
            PassageHit(
                passage_id=int(row["passage_id"]),
                conversation_id=int(row["conversation_id"]),
                message_id=int(row["message_id"]),
                title=str(row["title"]),
                role=str(row["role"]),
                text=str(row["text"]),
                created_at=row["created_at"],
                is_primary_path=is_primary_path,
                semantic_rank=vector_hit.rank,
                semantic_score=vector_hit.score,
                segment_id=row["segment_id"],
                segment_title=row["segment_title"],
                channels=("semantic",),
            )
        )
        if len(hits) >= limit:
            break
    return hits
