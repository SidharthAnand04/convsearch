from __future__ import annotations

from convsearch.config.settings import SearchWeights
from convsearch.domain.models import PassageHit


def reciprocal_rank_fusion(
    lexical_hits: list[PassageHit],
    semantic_hits: list[PassageHit],
    *,
    weights: SearchWeights,
    rrf_k: int,
    title_hits: list[PassageHit] | None = None,
    title_weight: float = 0.0,
) -> list[PassageHit]:
    title_hits = title_hits or []
    merged: dict[int, PassageHit] = {}
    scores: dict[int, float] = {}
    lexical_ranks = {hit.passage_id: rank for rank, hit in enumerate(lexical_hits, start=1)}
    semantic_ranks = {hit.passage_id: rank for rank, hit in enumerate(semantic_hits, start=1)}
    title_ranks = {hit.passage_id: rank for rank, hit in enumerate(title_hits, start=1)}
    lexical_by_id = {hit.passage_id: hit for hit in lexical_hits}
    semantic_by_id = {hit.passage_id: hit for hit in semantic_hits}
    title_by_id = {hit.passage_id: hit for hit in title_hits}
    for hit in lexical_hits + semantic_hits + title_hits:
        merged.setdefault(hit.passage_id, hit)
    for passage_id in merged:
        score = 0.0
        channels: list[str] = []
        lexical_rank = lexical_ranks.get(passage_id)
        semantic_rank = semantic_ranks.get(passage_id)
        title_rank = title_ranks.get(passage_id)
        if lexical_rank is not None:
            score += weights.lexical / (rrf_k + lexical_rank)
            channels.append("lexical")
        if semantic_rank is not None:
            score += weights.semantic / (rrf_k + semantic_rank)
            channels.append("semantic")
        if title_rank is not None:
            score += title_weight / (rrf_k + title_rank)
            channels.append("title")
        scores[passage_id] = score
        base = merged[passage_id]
        merged[passage_id] = PassageHit(
            passage_id=base.passage_id,
            conversation_id=base.conversation_id,
            message_id=base.message_id,
            title=base.title,
            role=base.role,
            text=base.text,
            created_at=base.created_at,
            is_primary_path=base.is_primary_path,
            lexical_rank=lexical_rank,
            semantic_rank=semantic_rank,
            title_rank=title_rank,
            reranker_rank=base.reranker_rank,
            lexical_score=lexical_by_id.get(passage_id, base).lexical_score,
            semantic_score=semantic_by_id.get(passage_id, base).semantic_score,
            title_score=title_by_id.get(passage_id, base).title_score,
            reranker_score=base.reranker_score,
            fused_score=score,
            final_score=base.final_score,
            segment_id=base.segment_id,
            segment_title=base.segment_title,
            channels=tuple(channels),
        )
    return sorted(merged.values(), key=lambda hit: hit.fused_score, reverse=True)
