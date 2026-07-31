from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class ImportedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_node_id: str
    parent_source_node_id: str | None = None
    source_message_id: str
    role: str
    created_at: str | None = None
    source_order: int
    is_primary_path: bool
    text: str


class ImportedConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_conversation_id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ImportedMessage]


@dataclass(frozen=True)
class Passage:
    conversation_id: int
    message_id: int
    passage_order: int
    text: str
    start_offset: int
    end_offset: int
    word_count: int
    content_hash: str


@dataclass(frozen=True)
class PassageHit:
    passage_id: int
    conversation_id: int
    message_id: int
    title: str
    role: str
    text: str
    created_at: str | None
    is_primary_path: bool
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    title_rank: int | None = None
    reranker_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None
    title_score: float | None = None
    reranker_score: float | None = None
    fused_score: float = 0.0
    final_score: float | None = None
    segment_id: int | None = None
    segment_title: str | None = None
    channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationResult:
    conversation_id: int
    title: str
    created_at: str | None
    updated_at: str | None
    score: float
    best_passages: list[PassageHit]
    distinct_message_count: int
    features: dict[str, float] | None = None


@dataclass(frozen=True)
class SegmentResult:
    segment_id: int
    conversation_id: int
    conversation_title: str
    title: str | None
    score: float
    best_passages: list[PassageHit]
