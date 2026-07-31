from __future__ import annotations

from collections import defaultdict
from statistics import mean

from convsearch.config.settings import AggregationWeights
from convsearch.domain.models import ConversationResult, PassageHit


def aggregate_conversations(
    hits: list[PassageHit],
    *,
    weights: AggregationWeights,
    limit: int,
    passages_per_conversation: int,
) -> list[ConversationResult]:
    grouped: dict[int, list[PassageHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.conversation_id].append(hit)
    results: list[ConversationResult] = []
    for conversation_id, passage_hits in grouped.items():
        ranked = sorted(passage_hits, key=_score, reverse=True)
        top_scores = [_score(hit) for hit in ranked[:3]]
        distinct_messages = {hit.message_id for hit in ranked}
        distinct_segments = {hit.segment_id for hit in ranked if hit.segment_id is not None}
        channels = {channel for hit in ranked for channel in hit.channels}
        primary_ratio = sum(1 for hit in ranked if hit.is_primary_path) / max(len(ranked), 1)
        bonus = (min(len(distinct_messages), 3) / 3) * _score(ranked[0])
        channel_bonus = (min(len(channels), 4) / 4) * 0.02 * _score(ranked[0])
        length_penalty = 1.0 / (1.0 + max(len(ranked) - 6, 0) * 0.03)
        score = (
            weights.best_passage * _score(ranked[0])
            + weights.mean_top_three * mean(top_scores)
            + weights.distinct_message_bonus * bonus
            + channel_bonus
        ) * length_penalty
        first = ranked[0]
        features = {
            "best_passage_score": _score(ranked[0]),
            "mean_top_three_score": mean(top_scores),
            "distinct_message_count": float(len(distinct_messages)),
            "distinct_segment_count": float(len(distinct_segments)),
            "channel_diversity": float(len(channels)),
            "primary_path_ratio": primary_ratio,
            "conversation_length_penalty": length_penalty,
        }
        results.append(
            ConversationResult(
                conversation_id=conversation_id,
                title=first.title,
                created_at=first.created_at,
                updated_at=None,
                score=score,
                best_passages=ranked[:passages_per_conversation],
                distinct_message_count=len(distinct_messages),
                features=features,
            )
        )
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def _score(hit: PassageHit) -> float:
    return hit.final_score if hit.final_score is not None else hit.fused_score
