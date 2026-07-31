from __future__ import annotations

from dataclasses import dataclass, field

MEMORY_KINDS = (
    "decision",
    "task",
    "preference",
    "project_state",
    "risk",
    "constraint",
    "open_question",
)
MEMORY_STATUSES = (
    "proposed",
    "active",
    "contested",
    "superseded",
    "invalidated",
    "historical",
)
TASK_STATES = ("open", "completed")


@dataclass(frozen=True)
class ExtractedMemory:
    kind: str
    subject_key: str
    statement: str
    confidence: float
    project: str | None
    task_state: str | None
    conversation_id: int
    message_id: int
    created_at: str | None
    quote: str
    start_offset: int
    end_offset: int
    entities: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEvidence:
    evidence_id: int
    passage_id: int | None
    message_id: int
    quote: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class MemoryRelation:
    relation: str
    other_memory_id: int
    other_statement: str
    reason: str | None
    direction: str


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: int
    kind: str
    subject_key: str
    statement: str
    status: str
    confidence: float
    project: str | None
    task_state: str | None
    conversation_id: int
    conversation_title: str | None
    message_id: int
    created_at: str | None
    evidence: tuple[MemoryEvidence, ...] = ()
    relations: tuple[MemoryRelation, ...] = ()
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- created_at above is capture time, i.e. when convsearch
    # first saw it), or 'unknown' (neither is available). Callers should label the date
    # accordingly rather than presenting a capture time as a creation date.
    date_source: str = "unknown"
