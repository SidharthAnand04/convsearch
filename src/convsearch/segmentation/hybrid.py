from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from convsearch.config.settings import SegmentationSettings
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.segmentation.models import ProposedSegment, SegmentableMessage
from convsearch.segmentation.rules import RuleBasedSegmentationProvider
from convsearch.segmentation.semantic import (
    SegmentDraft,
    SemanticShiftSegmentationProvider,
    cosine_similarity,
    embed_messages,
    finalize_segments,
    is_question_answer_pair,
    merge_undersized_drafts,
)

_DEFAULT_CONFIDENCE = 0.70

MeanVectorFn = Callable[[SegmentDraft], NDArray[np.float32]]


class HybridSegmentationProvider:
    """Unions rule-based and semantic-shift boundaries, then merges similar neighbors."""

    version = "hybrid-v1"

    def __init__(self, settings: SegmentationSettings, provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.provider = provider
        self._rules = RuleBasedSegmentationProvider(settings)
        self._semantic = SemanticShiftSegmentationProvider(settings, provider)

    def segment(self, messages: Sequence[SegmentableMessage]) -> list[ProposedSegment]:
        ordered = sorted(messages, key=lambda message: message.source_order)
        if not ordered:
            return []
        boundaries = self._union_boundaries(ordered)
        drafts = self._build_drafts(ordered, boundaries)
        drafts = merge_undersized_drafts(drafts, self.settings.minimum_segment_messages)
        vectors = embed_messages(self.provider, ordered)
        index_of = {message.message_id: index for index, message in enumerate(ordered)}
        drafts = self._merge_similar_drafts(drafts, vectors, index_of)
        return finalize_segments(self.version, drafts)

    def _union_boundaries(
        self, ordered: list[SegmentableMessage]
    ) -> dict[int, tuple[set[str], float]]:
        index_of = {message.message_id: index for index, message in enumerate(ordered)}
        boundaries: dict[int, tuple[set[str], float]] = {}

        def add(index: int, reasons: Sequence[str], confidence: float) -> None:
            if index <= 0:
                return
            previous, current = ordered[index - 1], ordered[index]
            splits_branch = previous.is_primary_path != current.is_primary_path
            if is_question_answer_pair(previous, current) and not splits_branch:
                return
            existing = boundaries.get(index)
            if existing is None:
                boundaries[index] = (set(reasons), confidence)
            else:
                existing[0].update(reasons)
                boundaries[index] = (existing[0], max(existing[1], confidence))

        for provider in (self._rules, self._semantic):
            for segment in provider.segment(ordered):
                add(
                    index_of[segment.start_message_id],
                    segment.reasons,
                    segment.boundary_confidence,
                )
        for index in range(1, len(ordered)):
            gap_minutes = _gap_minutes(ordered[index - 1].created_at, ordered[index].created_at)
            if gap_minutes is not None and gap_minutes >= self.settings.time_gap_minutes:
                add(index, (f"time_gap={gap_minutes:.0f}m",), _DEFAULT_CONFIDENCE)
        return boundaries

    def _build_drafts(
        self,
        ordered: list[SegmentableMessage],
        boundaries: dict[int, tuple[set[str], float]],
    ) -> list[SegmentDraft]:
        drafts = [SegmentDraft([ordered[0]], ("segment_start",), _DEFAULT_CONFIDENCE)]
        for index in range(1, len(ordered)):
            previous, current = ordered[index - 1], ordered[index]
            splits_branch = previous.is_primary_path != current.is_primary_path
            boundary = boundaries.get(index)
            oversized = len(
                drafts[-1].messages
            ) >= self.settings.maximum_segment_messages and not is_question_answer_pair(
                previous, current
            )
            if splits_branch:
                reasons = boundary[0] if boundary else {"branch_boundary"}
                reasons.add("branch_boundary")
                confidence = boundary[1] if boundary else _DEFAULT_CONFIDENCE
                drafts.append(SegmentDraft([current], tuple(sorted(reasons)), confidence))
            elif boundary is not None:
                drafts.append(SegmentDraft([current], tuple(sorted(boundary[0])), boundary[1]))
            elif oversized:
                drafts.append(
                    SegmentDraft([current], ("max_segment_messages",), _DEFAULT_CONFIDENCE)
                )
            else:
                drafts[-1].messages.append(current)
        return drafts

    def _merge_similar_drafts(
        self,
        drafts: list[SegmentDraft],
        vectors: NDArray[np.float32],
        index_of: dict[int, int],
    ) -> list[SegmentDraft]:
        def mean_vector(draft: SegmentDraft) -> NDArray[np.float32]:
            rows = [index_of[message.message_id] for message in draft.messages]
            return np.asarray(np.mean(vectors[rows], axis=0), dtype=np.float32)

        merged: list[SegmentDraft] = []
        for draft in drafts:
            if merged and self._should_merge(merged[-1], draft, mean_vector):
                merged[-1].messages.extend(draft.messages)
            else:
                merged.append(draft)
        return merged

    def _should_merge(
        self,
        left: SegmentDraft,
        right: SegmentDraft,
        mean_vector: MeanVectorFn,
    ) -> bool:
        if left.messages[-1].is_primary_path != right.messages[0].is_primary_path:
            return False
        if len(left.messages) + len(right.messages) > self.settings.maximum_segment_messages:
            return False
        similarity = cosine_similarity(mean_vector(left), mean_vector(right))
        return similarity > self.settings.merge_similarity_threshold


def _gap_minutes(previous: str | None, current: str | None) -> float | None:
    if not previous or not current:
        return None
    try:
        earlier = datetime.fromisoformat(previous)
        later = datetime.fromisoformat(current)
    except ValueError:
        return None
    return abs((later - earlier).total_seconds()) / 60.0
