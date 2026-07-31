from __future__ import annotations

from convsearch.config.settings import RerankingSettings
from convsearch.domain.models import PassageHit
from convsearch.retrieval.reranking import DeterministicReranker, apply_reranking


def test_deterministic_reranker_reorders_candidates() -> None:
    hits = [
        PassageHit(1, 1, 1, "title", "assistant", "unrelated text", None, True, fused_score=0.1),
        PassageHit(
            2,
            1,
            2,
            "title",
            "assistant",
            "Use IndexFlatIP with conversations.json locally.",
            None,
            True,
            fused_score=0.095,
        ),
    ]
    reranked = apply_reranking(
        "IndexFlatIP conversations.json",
        hits,
        RerankingSettings(candidate_limit=2, weight=20.0),
        DeterministicReranker(),
    )
    assert reranked[0].passage_id == 2
    assert reranked[0].reranker_rank == 1
    assert "reranker" in reranked[0].channels
