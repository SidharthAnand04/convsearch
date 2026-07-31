from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from convsearch.config.settings import SegmentationSettings
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.segmentation.models import ProposedSegment, SegmentableMessage
from convsearch.utils import stable_hash

_DEFAULT_CONFIDENCE = 0.70
_EMBED_BATCH_SIZE = 32


@dataclass
class SegmentDraft:
    """A contiguous group of messages plus the boundary evidence that started it."""

    messages: list[SegmentableMessage]
    reasons: tuple[str, ...]
    confidence: float


def cosine_similarity(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right)) / denominator


def embed_messages(
    provider: EmbeddingProvider, messages: Sequence[SegmentableMessage]
) -> NDArray[np.float32]:
    texts = [message.text for message in messages]
    return provider.encode_documents(texts, batch_size=_EMBED_BATCH_SIZE)


def is_question_answer_pair(previous: SegmentableMessage, current: SegmentableMessage) -> bool:
    return previous.role == "user" and current.role == "assistant"


def split_into_branch_runs(
    messages: Sequence[SegmentableMessage],
) -> list[list[SegmentableMessage]]:
    runs: list[list[SegmentableMessage]] = []
    for message in messages:
        if runs and runs[-1][-1].is_primary_path == message.is_primary_path:
            runs[-1].append(message)
        else:
            runs.append([message])
    return runs


def segment_title(messages: Sequence[SegmentableMessage]) -> str:
    for message in messages:
        if message.role == "user" and message.text.strip():
            return " ".join(message.text.split())[:80]
    return " ".join(messages[0].text.split())[:80]


def merge_undersized_drafts(
    drafts: list[SegmentDraft], minimum_segment_messages: int
) -> list[SegmentDraft]:
    merged: list[SegmentDraft] = []
    for draft in drafts:
        if (
            merged
            and len(draft.messages) < minimum_segment_messages
            and merged[-1].messages[-1].is_primary_path == draft.messages[0].is_primary_path
        ):
            merged[-1].messages.extend(draft.messages)
        else:
            merged.append(draft)
    return merged


def finalize_segments(version: str, drafts: Sequence[SegmentDraft]) -> list[ProposedSegment]:
    segments: list[ProposedSegment] = []
    for segment_order, draft in enumerate(drafts):
        group = draft.messages
        text = "\n".join(f"{message.role}: {message.text}" for message in group)
        segments.append(
            ProposedSegment(
                conversation_id=group[0].conversation_id,
                segment_order=segment_order,
                start_message_id=group[0].message_id,
                end_message_id=group[-1].message_id,
                title=segment_title(group),
                summary=None,
                boundary_confidence=draft.confidence,
                reasons=draft.reasons,
                message_ids=tuple(message.message_id for message in group),
                content_hash=stable_hash(version, group[0].conversation_id, segment_order, text),
            )
        )
    return segments


class SemanticShiftSegmentationProvider:
    """Splits conversations where consecutive messages drift apart semantically."""

    version = "semantic-shift-v1"

    def __init__(self, settings: SegmentationSettings, provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.provider = provider

    def segment(self, messages: Sequence[SegmentableMessage]) -> list[ProposedSegment]:
        ordered = sorted(messages, key=lambda message: message.source_order)
        if not ordered:
            return []
        drafts: list[SegmentDraft] = []
        for run_index, run in enumerate(split_into_branch_runs(ordered)):
            drafts.extend(self._segment_run(run, is_first_run=run_index == 0))
        drafts = merge_undersized_drafts(drafts, self.settings.minimum_segment_messages)
        return finalize_segments(self.version, drafts)

    def _segment_run(
        self, run: list[SegmentableMessage], *, is_first_run: bool
    ) -> list[SegmentDraft]:
        run_start_reason = "segment_start" if is_first_run else "branch_boundary"
        vectors = embed_messages(self.provider, run)
        drafts = [SegmentDraft([run[0]], (run_start_reason,), _DEFAULT_CONFIDENCE)]
        for index in range(1, len(run)):
            boundary = self._propose_boundary(
                run[index - 1],
                run[index],
                vectors[index - 1],
                vectors[index],
                len(drafts[-1].messages),
            )
            if boundary is None:
                drafts[-1].messages.append(run[index])
            else:
                reasons, confidence = boundary
                drafts.append(SegmentDraft([run[index]], reasons, confidence))
        return drafts

    def _propose_boundary(
        self,
        previous: SegmentableMessage,
        current: SegmentableMessage,
        previous_vector: NDArray[np.float32],
        current_vector: NDArray[np.float32],
        current_group_size: int,
    ) -> tuple[tuple[str, ...], float] | None:
        if is_question_answer_pair(previous, current):
            return None
        if current_group_size >= self.settings.maximum_segment_messages:
            return ("max_segment_messages",), _DEFAULT_CONFIDENCE
        if current_group_size < self.settings.minimum_segment_messages:
            return None
        similarity = cosine_similarity(previous_vector, current_vector)
        if similarity < self.settings.semantic_shift_threshold:
            confidence = min(1.0, max(0.0, 1.0 - similarity))
            return ("semantic_shift", f"similarity={similarity:.3f}"), confidence
        return None
