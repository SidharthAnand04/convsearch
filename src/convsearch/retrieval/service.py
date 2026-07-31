from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from convsearch.config.settings import Settings, faiss_index_path, vector_map_path
from convsearch.domain.models import ConversationResult, SegmentResult
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.retrieval.aggregation import aggregate_conversations
from convsearch.retrieval.fusion import reciprocal_rank_fusion
from convsearch.retrieval.lexical import lexical_search, title_search
from convsearch.retrieval.reranking import apply_reranking, make_reranker
from convsearch.retrieval.segments import segment_search
from convsearch.retrieval.semantic import semantic_search
from convsearch.storage.database import connection


def search_conversations(
    workspace: Path,
    query: str,
    settings: Settings,
    provider: EmbeddingProvider,
    *,
    limit: int,
    profile: str,
    show_passages: int,
    include_branches: bool = False,
    rerank: bool | None = None,
    test_reranker: bool = False,
    llm_query: bool | None = None,
) -> list[ConversationResult]:
    if not vector_map_path(workspace).exists():
        raise RuntimeError("Vector index is missing. Run `convsearch index` first.")
    if not (
        faiss_index_path(workspace).exists()
        or faiss_index_path(workspace).with_suffix(".npy").exists()
    ):
        raise RuntimeError("Vector index is missing. Run `convsearch index` first.")
    search_query = query
    use_llm_query = settings.llm.enabled if llm_query is None else llm_query
    if use_llm_query:
        try:
            from convsearch.llm.client import expand_query

            expanded = expand_query(
                query,
                settings.llm.model,
                max_terms=settings.llm.max_expansion_terms,
            )
            search_query = expanded.search_text or query
        except Exception:
            if settings.llm.failure_policy == "error":
                raise
    with connection(workspace) as conn:
        lexical_hits = lexical_search(
            conn,
            search_query,
            settings.lexical_candidate_limit,
            include_branches=include_branches,
            settings=settings,
        )
        title_hits = title_search(
            conn,
            query,
            settings.lexical_candidate_limit,
            include_branches=include_branches,
        )
        semantic_hits = semantic_search(
            conn,
            workspace,
            query,
            settings,
            provider,
            settings.semantic_candidate_limit,
            include_branches=include_branches,
            profile=profile,
        )
        fused = reciprocal_rank_fusion(
            lexical_hits,
            semantic_hits,
            weights=settings.profile_weights(profile),
            rrf_k=settings.rrf_k,
            title_hits=title_hits,
            title_weight=settings.retrieval.title_weight,
        )
        should_rerank = settings.reranking.enabled if rerank is None else rerank
        if should_rerank:
            try:
                reranker = make_reranker(settings.reranking, deterministic=test_reranker)
                fused = apply_reranking(query, fused, settings.reranking, reranker)
            except Exception:
                if settings.reranking.failure_policy == "error":
                    raise
        results = aggregate_conversations(
            fused,
            weights=settings.aggregation_weights,
            limit=limit,
            passages_per_conversation=show_passages,
        )
        return _attach_conversation_timestamps(conn, results)


def _attach_conversation_timestamps(
    conn: sqlite3.Connection, results: list[ConversationResult]
) -> list[ConversationResult]:
    """Fill in conversation-level timestamps.

    Aggregation only sees passage hits, which carry message timestamps rather than the
    conversation's own, so it leaves `updated_at` unset. Callers surface this date next to
    each result, so read the real value back from the conversations table.
    """
    if not results:
        return results
    ids = [result.conversation_id for result in results]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT conversation_id, created_at, updated_at FROM conversations "
        f"WHERE conversation_id IN ({placeholders})",
        ids,
    ).fetchall()
    timestamps = {
        int(row["conversation_id"]): (row["created_at"], row["updated_at"]) for row in rows
    }
    filled = []
    for result in results:
        created_at, updated_at = timestamps.get(result.conversation_id, (None, None))
        filled.append(
            replace(
                result,
                created_at=created_at or result.created_at,
                updated_at=updated_at,
            )
        )
    return filled


def search_segments(
    workspace: Path,
    query: str,
    settings: Settings,
    *,
    limit: int,
    include_branches: bool = False,
) -> list[SegmentResult]:
    with connection(workspace) as conn:
        return segment_search(conn, query, limit, include_branches=include_branches)
