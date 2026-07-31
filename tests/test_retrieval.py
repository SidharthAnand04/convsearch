from __future__ import annotations

from pathlib import Path

from convsearch.config.settings import SearchWeights, Settings
from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.retrieval.aggregation import aggregate_conversations
from convsearch.retrieval.fusion import reciprocal_rank_fusion
from convsearch.retrieval.lexical import lexical_search
from convsearch.retrieval.service import search_conversations
from convsearch.storage.database import connection
from tests.conftest import index_with_test_embeddings


def test_exact_lexical_match(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    with connection(workspace) as conn:
        hits = lexical_search(conn, "SQLite", 10)
    assert hits
    assert "SQLite" in hits[0].text


def test_semantic_match(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    results = search_conversations(
        workspace,
        "hybrid FAISS search",
        settings,
        DeterministicEmbeddingProvider(),
        limit=3,
        profile="semantic",
        show_passages=2,
        include_branches=False,
    )
    assert results


def test_hybrid_fusion_and_profile_weighting(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    with connection(workspace) as conn:
        lexical = lexical_search(conn, "SQLite", 10)
        semantic = lexical_search(conn, "FAISS", 10)
    exact = reciprocal_rank_fusion(
        lexical, semantic, weights=SearchWeights(lexical=1.5, semantic=0.5), rrf_k=60
    )
    semantic_weighted = reciprocal_rank_fusion(
        lexical, semantic, weights=SearchWeights(lexical=0.5, semantic=1.5), rrf_k=60
    )
    assert exact
    assert semantic_weighted
    exact_scores = {hit.passage_id: hit.fused_score for hit in exact}
    semantic_scores = {hit.passage_id: hit.fused_score for hit in semantic_weighted}
    shared_passage_id = next(iter(exact_scores.keys() & semantic_scores.keys()))
    assert exact_scores[shared_passage_id] != semantic_scores[shared_passage_id]


def test_conversation_aggregation_bias_control() -> None:
    from convsearch.config.settings import AggregationWeights
    from convsearch.domain.models import PassageHit

    hits = [
        PassageHit(1, 1, 1, "short", "user", "a", None, True, fused_score=0.10),
        PassageHit(2, 2, 2, "long", "user", "b", None, True, fused_score=0.08),
        PassageHit(3, 2, 3, "long", "user", "c", None, True, fused_score=0.08),
        PassageHit(4, 2, 4, "long", "user", "d", None, True, fused_score=0.08),
        PassageHit(5, 2, 5, "long", "user", "e", None, True, fused_score=0.08),
    ]
    results = aggregate_conversations(
        hits, weights=AggregationWeights(), limit=2, passages_per_conversation=2
    )
    assert results[0].title == "short"


def test_primary_path_search_excludes_alternate_branch(
    workspace: Path, settings: Settings, branch_export_zip: Path
) -> None:
    import_chatgpt_zip(branch_export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    results = search_conversations(
        workspace,
        "Pinecone Cloud Vector Enterprise",
        settings,
        DeterministicEmbeddingProvider(),
        limit=5,
        profile="balanced",
        show_passages=3,
    )
    texts = [hit.text for result in results for hit in result.best_passages]
    assert not any("Pinecone Cloud Vector Enterprise" in text for text in texts)

    branch_results = search_conversations(
        workspace,
        "Pinecone Cloud Vector Enterprise",
        settings,
        DeterministicEmbeddingProvider(),
        limit=5,
        profile="balanced",
        show_passages=3,
        include_branches=True,
    )
    branch_hits = [hit for result in branch_results for hit in result.best_passages]
    assert any("Pinecone Cloud Vector Enterprise" in hit.text for hit in branch_hits)
    assert any(not hit.is_primary_path for hit in branch_hits)


def test_semantic_branch_filter_overfetches_primary_result(
    workspace: Path, settings: Settings, branch_export_zip: Path
) -> None:
    import_chatgpt_zip(branch_export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    results = search_conversations(
        workspace,
        "Pinecone Cloud Vector Enterprise",
        settings,
        DeterministicEmbeddingProvider(),
        limit=1,
        profile="semantic",
        show_passages=1,
    )
    assert len(results) == 1
    assert results[0].best_passages
    assert results[0].best_passages[0].is_primary_path
    assert "Pinecone Cloud Vector Enterprise" not in results[0].best_passages[0].text


def test_fusion_preserves_branch_status() -> None:
    from convsearch.domain.models import PassageHit

    lexical = [PassageHit(1, 1, 1, "title", "assistant", "lex", None, False)]
    semantic = [
        PassageHit(1, 1, 1, "title", "assistant", "lex", None, False),
        PassageHit(2, 1, 2, "title", "assistant", "sem", None, True),
    ]
    fused = reciprocal_rank_fusion(
        lexical, semantic, weights=SearchWeights(lexical=1.0, semantic=1.0), rrf_k=60
    )
    by_id = {hit.passage_id: hit for hit in fused}
    assert by_id[1].is_primary_path is False
    assert by_id[2].is_primary_path is True
