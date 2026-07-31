from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InteractionEvent:
    event_type: str  # 'search'|'open'|'inspect'|'ask'
    query: str = ""
    conversation_id: int | None = None
    passage_id: int | None = None
    segment_id: int | None = None
    position: int | None = None
    created_at: str | None = None  # ISO-8601 UTC; store fills when None
