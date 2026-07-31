from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from convsearch.memory.models import MemoryEvidence
from convsearch.memory.store import set_memory_status

# The extractor (extract.py) only ever emits two confidence tiers: 0.7 for a plain match and
# 0.9 when the sentence starts with a trigger phrase or names an identifier. The threshold
# sits between those two tiers so weakly-supported extractions (0.7) surface for review while
# strongly-supported ones (0.9) do not clutter the queue.
LOW_CONFIDENCE_THRESHOLD = 0.75


@dataclass(frozen=True)
class ConflictRef:
    memory_id: int
    statement: str
    status: str
    reason: str | None


@dataclass(frozen=True)
class SupersessionRef:
    memory_id: int
    statement: str


@dataclass(frozen=True)
class ReviewItem:
    memory_id: int
    kind: str
    statement: str
    status: str
    project: str | None
    confidence: float
    created_at: str | None
    pinned: bool
    reviewed_at: str | None
    conversation_id: int
    conversation_title: str | None
    evidence: tuple[MemoryEvidence, ...]
    conflicts: tuple[ConflictRef, ...]
    superseded_by: tuple[SupersessionRef, ...]
    review_reason: str


@dataclass(frozen=True)
class ReviewQueue:
    items: tuple[ReviewItem, ...]
    total_pending: int
    total_pinned: int
    total_contested: int
    total_invalidated: int


def _review_reason(status: str, has_conflict: bool, confidence: float) -> str | None:
    """Return the plain-language reason the highest-priority matching rule fired.

    Checked in priority order (highest need first); the first rule that matches wins.
    Returns None if no rule matches, meaning the memory does not belong in the queue.
    """
    if status == "contested":
        return "This memory conflicts with another memory and has not been resolved."
    if has_conflict:
        return "This memory is linked to a conflicting memory that needs a decision."
    if status == "proposed":
        return "This memory was extracted but has never been confirmed."
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return (
            f"This memory was extracted with low confidence "
            f"({confidence:.2f} < {LOW_CONFIDENCE_THRESHOLD:.2f})."
        )
    return None


def _rank(status: str, has_conflict: bool, confidence: float) -> int:
    if status == "contested":
        return 1
    if has_conflict:
        return 2
    if status == "proposed":
        return 3
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return 4
    return 99


def _load_evidence_batch(
    conn: sqlite3.Connection, memory_ids: list[int]
) -> dict[int, tuple[MemoryEvidence, ...]]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        SELECT memory_id, evidence_id, passage_id, message_id, quote, start_offset, end_offset
        FROM memory_evidence
        WHERE memory_id IN ({placeholders})
        ORDER BY memory_id, evidence_id
        """,
        memory_ids,
    ).fetchall()
    grouped: dict[int, list[MemoryEvidence]] = defaultdict(list)
    for row in rows:
        grouped[row["memory_id"]].append(
            MemoryEvidence(
                evidence_id=row["evidence_id"],
                passage_id=row["passage_id"],
                message_id=row["message_id"],
                quote=row["quote"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
            )
        )
    return {memory_id: tuple(items) for memory_id, items in grouped.items()}


def _load_conflicts_batch(
    conn: sqlite3.Connection, memory_ids: list[int]
) -> dict[int, tuple[ConflictRef, ...]]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        SELECT mr.from_memory_id, mr.to_memory_id, mr.reason,
               mf.statement AS from_statement, mf.status AS from_status,
               mt.statement AS to_statement, mt.status AS to_status
        FROM memory_relations mr
        JOIN memories mf ON mf.memory_id = mr.from_memory_id
        JOIN memories mt ON mt.memory_id = mr.to_memory_id
        WHERE mr.relation = 'conflicts_with'
          AND (mr.from_memory_id IN ({placeholders}) OR mr.to_memory_id IN ({placeholders}))
        """,
        [*memory_ids, *memory_ids],
    ).fetchall()
    grouped: dict[int, list[ConflictRef]] = defaultdict(list)
    id_set = set(memory_ids)
    for row in rows:
        from_id, to_id = row["from_memory_id"], row["to_memory_id"]
        if from_id in id_set:
            grouped[from_id].append(
                ConflictRef(
                    memory_id=to_id,
                    statement=row["to_statement"],
                    status=row["to_status"],
                    reason=row["reason"],
                )
            )
        if to_id in id_set:
            grouped[to_id].append(
                ConflictRef(
                    memory_id=from_id,
                    statement=row["from_statement"],
                    status=row["from_status"],
                    reason=row["reason"],
                )
            )
    return {memory_id: tuple(items) for memory_id, items in grouped.items()}


def _load_superseded_by_batch(
    conn: sqlite3.Connection, memory_ids: list[int]
) -> dict[int, tuple[SupersessionRef, ...]]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        SELECT mr.to_memory_id AS memory_id, m.memory_id AS other_id, m.statement AS other_statement
        FROM memory_relations mr
        JOIN memories m ON m.memory_id = mr.from_memory_id
        WHERE mr.relation = 'supersedes' AND mr.to_memory_id IN ({placeholders})
        """,
        memory_ids,
    ).fetchall()
    grouped: dict[int, list[SupersessionRef]] = defaultdict(list)
    for row in rows:
        grouped[row["memory_id"]].append(
            SupersessionRef(memory_id=row["other_id"], statement=row["other_statement"])
        )
    return {memory_id: tuple(items) for memory_id, items in grouped.items()}


def _conflict_ids(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute(
        "SELECT from_memory_id, to_memory_id FROM memory_relations "
        "WHERE relation = 'conflicts_with'"
    ).fetchall()
    ids: set[int] = set()
    for row in rows:
        ids.add(row["from_memory_id"])
        ids.add(row["to_memory_id"])
    return ids


def build_review_queue(
    conn: sqlite3.Connection,
    *,
    limit: int = 30,
    kind: str | None = None,
    project: str | None = None,
    include_reviewed: bool = False,
) -> ReviewQueue:
    """Build the human review queue, highest-need memories first.

    Priority order: contested status, then participation in a `conflicts_with` relation,
    then never-confirmed (`proposed`) status, then low confidence. Pinned memories and
    `invalidated` memories are excluded from the pending queue (settled or already curated);
    already-reviewed memories are excluded unless `include_reviewed=True`.
    """
    conditions: list[str] = ["m.status != 'invalidated'", "m.pinned = 0"]
    params: list[object] = []
    if not include_reviewed:
        conditions.append("m.reviewed_at IS NULL")
    if kind is not None:
        conditions.append("m.kind = ?")
        params.append(kind)
    if project is not None:
        conditions.append("m.project = ?")
        params.append(project)
    where_clause = " AND ".join(conditions)

    rows = conn.execute(
        f"""
        SELECT m.memory_id, m.kind, m.statement, m.status, m.project, m.confidence,
               m.created_at, m.pinned, m.reviewed_at, m.conversation_id,
               c.title AS conversation_title
        FROM memories m
        LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {where_clause}
        """,
        params,
    ).fetchall()

    conflict_ids = _conflict_ids(conn)

    candidates: list[tuple[int, sqlite3.Row, str]] = []
    for row in rows:
        has_conflict = row["memory_id"] in conflict_ids
        reason = _review_reason(row["status"], has_conflict, row["confidence"])
        if reason is None:
            continue
        rank = _rank(row["status"], has_conflict, row["confidence"])
        candidates.append((rank, row, reason))

    candidates.sort(key=lambda entry: (entry[0], entry[1]["memory_id"]))
    total_pending = len(candidates)
    selected = candidates[:limit]

    memory_ids = [entry[1]["memory_id"] for entry in selected]
    evidence_by_id = _load_evidence_batch(conn, memory_ids)
    conflicts_by_id = _load_conflicts_batch(conn, memory_ids)
    superseded_by_id = _load_superseded_by_batch(conn, memory_ids)

    items = tuple(
        ReviewItem(
            memory_id=row["memory_id"],
            kind=row["kind"],
            statement=row["statement"],
            status=row["status"],
            project=row["project"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            pinned=bool(row["pinned"]),
            reviewed_at=row["reviewed_at"],
            conversation_id=row["conversation_id"],
            conversation_title=row["conversation_title"],
            evidence=evidence_by_id.get(row["memory_id"], ()),
            conflicts=conflicts_by_id.get(row["memory_id"], ()),
            superseded_by=superseded_by_id.get(row["memory_id"], ()),
            review_reason=reason,
        )
        for _, row, reason in selected
    )

    stat_conditions: list[str] = []
    stat_params: list[object] = []
    if kind is not None:
        stat_conditions.append("kind = ?")
        stat_params.append(kind)
    if project is not None:
        stat_conditions.append("project = ?")
        stat_params.append(project)
    stat_where = (" AND " + " AND ".join(stat_conditions)) if stat_conditions else ""

    total_pinned = conn.execute(
        f"SELECT COUNT(*) AS n FROM memories WHERE pinned = 1{stat_where}", stat_params
    ).fetchone()["n"]
    total_contested = conn.execute(
        f"SELECT COUNT(*) AS n FROM memories WHERE status = 'contested'{stat_where}", stat_params
    ).fetchone()["n"]
    total_invalidated = conn.execute(
        f"SELECT COUNT(*) AS n FROM memories WHERE status = 'invalidated'{stat_where}", stat_params
    ).fetchone()["n"]

    return ReviewQueue(
        items=items,
        total_pending=total_pending,
        total_pinned=total_pinned,
        total_contested=total_contested,
        total_invalidated=total_invalidated,
    )


def _stamp_reviewed(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute(
        "UPDATE memories SET reviewed_at = CURRENT_TIMESTAMP WHERE memory_id = ?",
        (memory_id,),
    )


def confirm_memory(conn: sqlite3.Connection, memory_id: int, *, reason: str | None = None) -> None:
    """Mark a memory as confirmed: status -> 'active', reviewed_at stamped, one transaction."""
    set_memory_status(conn, memory_id, "active", reason=reason)
    _stamp_reviewed(conn, memory_id)
    conn.commit()


def invalidate_memory(
    conn: sqlite3.Connection, memory_id: int, *, reason: str | None = None
) -> None:
    """Mark a memory as invalidated: status -> 'invalidated', reviewed_at stamped, one txn."""
    set_memory_status(conn, memory_id, "invalidated", reason=reason)
    _stamp_reviewed(conn, memory_id)
    conn.commit()


def set_memory_pinned(
    conn: sqlite3.Connection, memory_id: int, pinned: bool, *, reason: str | None = None
) -> None:
    """Set (or clear) the pin flag on a memory. Pinning is orthogonal to status and does not
    stamp reviewed_at or write status history. `reason` is accepted for API symmetry with
    confirm_memory/invalidate_memory but is not currently persisted (there is no pin history
    table)."""
    del reason
    row = conn.execute(
        "SELECT memory_id FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Memory not found: {memory_id}")
    conn.execute(
        "UPDATE memories SET pinned = ? WHERE memory_id = ?",
        (1 if pinned else 0, memory_id),
    )
    conn.commit()
