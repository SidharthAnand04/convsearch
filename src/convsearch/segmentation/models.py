from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentableMessage:
    message_id: int
    conversation_id: int
    source_order: int
    role: str
    text: str
    created_at: str | None
    is_primary_path: bool


@dataclass(frozen=True)
class ProposedSegment:
    conversation_id: int
    segment_order: int
    start_message_id: int
    end_message_id: int
    title: str | None
    summary: str | None
    boundary_confidence: float
    reasons: tuple[str, ...]
    message_ids: tuple[int, ...]
    content_hash: str
