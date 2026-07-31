from __future__ import annotations

import importlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Any, Protocol, cast

from convsearch.config.settings import RerankingSettings
from convsearch.domain.models import PassageHit
from convsearch.retrieval.query import parse_query


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class DeterministicReranker:
    model_id = "deterministic-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        parsed = parse_query(query)
        query_terms = {term.lower() for term in [*parsed.required_terms, *parsed.optional_terms]}
        query_terms.update(term.lower() for term in parsed.identifiers)
        phrase_terms = [phrase.lower() for phrase in parsed.phrases]
        scores: list[float] = []
        for passage in passages:
            text = passage.lower()
            tokens = set(re.findall(r"[a-z0-9_./:-]+", text))
            overlap = len(query_terms & tokens)
            phrase_bonus = sum(2 for phrase in phrase_terms if phrase in text)
            excluded_penalty = sum(2 for term in parsed.excluded_terms if term.lower() in text)
            scores.append(float(overlap + phrase_bonus - excluded_penalty))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `uv sync --extra ml --group dev`."
            ) from exc
        cross_encoder = cast(Any, sentence_transformers).CrossEncoder
        kwargs = {} if device == "auto" else {"device": device}
        self._model = cross_encoder(model_id, **kwargs)
        self.model_id = model_id

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


# Rerankers are memoised by configuration. `make_reranker` is called from inside
# `search_conversations`, i.e. once per query, and CrossEncoderReranker.__init__ loads a model
# from disk — the same mistake that was costing the auto-indexer 2.7s on every pass. Reranking
# is off by default, so this never showed up in the measured baseline; it would have made every
# single search pay a multi-second model load for whoever enabled it first.
_RERANKER_CACHE: dict[tuple[str, str, str], Reranker] = {}
_RERANKER_LOCK = threading.Lock()


def make_reranker(settings: RerankingSettings, *, deterministic: bool = False) -> Reranker:
    if deterministic:
        # Cheap and stateless; no point caching it.
        return DeterministicReranker()
    key = (settings.backend, settings.model, settings.device)
    with _RERANKER_LOCK:
        cached = _RERANKER_CACHE.get(key)
        if cached is not None:
            return cached
    # Built outside the lock: loading a model takes seconds and must not block other callers
    # that want a *different* reranker. A benign race just builds one twice on first use.
    built: Reranker
    if settings.backend == "llm":
        from convsearch.llm.client import LLMReranker

        built = LLMReranker(settings.llm_model)
    else:
        built = CrossEncoderReranker(settings.model, settings.device)
    with _RERANKER_LOCK:
        return _RERANKER_CACHE.setdefault(key, built)


def apply_reranking(
    query: str,
    hits: list[PassageHit],
    settings: RerankingSettings,
    reranker: Reranker,
) -> list[PassageHit]:
    candidates = hits[: settings.candidate_limit]
    remainder = hits[settings.candidate_limit :]
    raw_scores = reranker.score(query, [hit.text for hit in candidates])
    ranked_ids = {
        hit.passage_id: rank
        for rank, (hit, _score) in enumerate(_ranked_pairs(candidates, raw_scores), start=1)
    }
    score_by_id = {hit.passage_id: score for hit, score in zip(candidates, raw_scores, strict=True)}
    updated: list[PassageHit] = []
    for hit in candidates:
        rank = ranked_ids[hit.passage_id]
        final = hit.fused_score + settings.weight / (settings.rrf_k + rank)
        channels = tuple(dict.fromkeys((*hit.channels, "reranker")))
        updated.append(_replace_hit(hit, score_by_id[hit.passage_id], rank, final, channels))
    updated.sort(
        key=lambda hit: hit.final_score if hit.final_score is not None else -math.inf,
        reverse=True,
    )
    return updated + remainder


def _ranked_pairs(
    candidates: list[PassageHit], raw_scores: list[float]
) -> list[tuple[PassageHit, float]]:
    return sorted(
        zip(candidates, raw_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )


def _replace_hit(
    hit: PassageHit,
    reranker_score: float,
    reranker_rank: int,
    final_score: float,
    channels: tuple[str, ...],
) -> PassageHit:
    return PassageHit(
        passage_id=hit.passage_id,
        conversation_id=hit.conversation_id,
        message_id=hit.message_id,
        title=hit.title,
        role=hit.role,
        text=hit.text,
        created_at=hit.created_at,
        is_primary_path=hit.is_primary_path,
        lexical_rank=hit.lexical_rank,
        semantic_rank=hit.semantic_rank,
        title_rank=hit.title_rank,
        reranker_rank=reranker_rank,
        lexical_score=hit.lexical_score,
        semantic_score=hit.semantic_score,
        title_score=hit.title_score,
        reranker_score=reranker_score,
        fused_score=hit.fused_score,
        final_score=final_score,
        segment_id=hit.segment_id,
        segment_title=hit.segment_title,
        channels=channels,
    )
