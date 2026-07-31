from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from convsearch.memory.models import MemoryEvidence, MemoryRecord, MemoryRelation
from convsearch.retrieval.query import fts_quote, parse_query
from convsearch.utils import memory_effective_timestamp_source_sql, memory_effective_timestamp_sql


def _load_evidence(conn: sqlite3.Connection, memory_id: int) -> tuple[MemoryEvidence, ...]:
    rows = conn.execute(
        """
        SELECT evidence_id, passage_id, message_id, quote, start_offset, end_offset
        FROM memory_evidence
        WHERE memory_id = ?
        ORDER BY evidence_id
        """,
        (memory_id,),
    ).fetchall()
    return tuple(
        MemoryEvidence(
            evidence_id=row["evidence_id"],
            passage_id=row["passage_id"],
            message_id=row["message_id"],
            quote=row["quote"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
        )
        for row in rows
    )


def _load_relations(conn: sqlite3.Connection, memory_id: int) -> tuple[MemoryRelation, ...]:
    # Outgoing: from_memory_id = memory_id
    out_rows = conn.execute(
        """
        SELECT mr.relation, mr.to_memory_id AS other_id, mr.reason,
               m.statement AS other_statement
        FROM memory_relations mr
        JOIN memories m ON m.memory_id = mr.to_memory_id
        WHERE mr.from_memory_id = ?
        ORDER BY mr.relation_id
        """,
        (memory_id,),
    ).fetchall()

    # Incoming: to_memory_id = memory_id
    in_rows = conn.execute(
        """
        SELECT mr.relation, mr.from_memory_id AS other_id, mr.reason,
               m.statement AS other_statement
        FROM memory_relations mr
        JOIN memories m ON m.memory_id = mr.from_memory_id
        WHERE mr.to_memory_id = ?
        ORDER BY mr.relation_id
        """,
        (memory_id,),
    ).fetchall()

    relations: list[MemoryRelation] = []
    for row in out_rows:
        relations.append(
            MemoryRelation(
                relation=row["relation"],
                other_memory_id=row["other_id"],
                other_statement=row["other_statement"],
                reason=row["reason"],
                direction="outgoing",
            )
        )
    for row in in_rows:
        relations.append(
            MemoryRelation(
                relation=row["relation"],
                other_memory_id=row["other_id"],
                other_statement=row["other_statement"],
                reason=row["reason"],
                direction="incoming",
            )
        )
    return tuple(relations)


def _load_record(conn: sqlite3.Connection, row: sqlite3.Row) -> MemoryRecord:
    memory_id: int = row["memory_id"]
    evidence = _load_evidence(conn, memory_id)
    relations = _load_relations(conn, memory_id)
    return MemoryRecord(
        memory_id=memory_id,
        kind=row["kind"],
        subject_key=row["subject_key"],
        statement=row["statement"],
        status=row["status"],
        confidence=row["confidence"],
        project=row["project"],
        task_state=row["task_state"],
        conversation_id=row["conversation_id"],
        conversation_title=row["conversation_title"],
        message_id=row["message_id"],
        created_at=row["created_at"],
        evidence=evidence,
        relations=relations,
        date_source=row["date_source"],
    )


_MEMORY_SELECT = f"""
    SELECT m.memory_id, m.kind, m.subject_key, m.statement, m.status, m.confidence,
           m.project, m.task_state, m.conversation_id, m.message_id,
           {memory_effective_timestamp_sql("m", "c")} AS created_at,
           {memory_effective_timestamp_source_sql("m", "c")} AS date_source,
           c.title AS conversation_title
    FROM memories m
    LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
"""


def search_memories(
    conn: sqlite3.Connection,
    query: str,
    *,
    kinds: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    project: str | None = None,
    limit: int = 20,
) -> list[MemoryRecord]:
    """Search memories using FTS5 full-text search."""
    # Route through the identifier-aware parser used by lexical passage search so
    # identifier queries like `source_node_id`, `conversations.json`, or
    # `BAAI/bge-small-en-v1.5` are preserved as FTS phrases instead of being split apart.
    parsed = parse_query(query)
    positive = [*parsed.phrases, *parsed.required_terms, *parsed.optional_terms]
    if not positive:
        # Fall back to raw whitespace tokens (e.g. stopword-only queries) to preserve
        # the previous behavior of matching whatever the user typed.
        positive = query.split()
    if not positive:
        return []

    quoted = [fts_quote(t) for t in positive]

    # Non-FTS filters shared across attempts.
    filter_conditions: list[str] = []
    filter_params: list[object] = []
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        filter_conditions.append(f"m.kind IN ({placeholders})")
        filter_params.extend(kinds)
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        filter_conditions.append(f"m.status IN ({placeholders})")
        filter_params.extend(statuses)
    if project is not None:
        filter_conditions.append("m.project = ?")
        filter_params.append(project)

    def _run(fts_expr: str) -> list[MemoryRecord]:
        conditions = ["fts.rowid = m.memory_id", "memory_fts MATCH ?", *filter_conditions]
        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT m.memory_id, m.kind, m.subject_key, m.statement, m.status, m.confidence,
                   m.project, m.task_state, m.conversation_id, m.message_id,
                   {memory_effective_timestamp_sql("m", "c")} AS created_at,
                   {memory_effective_timestamp_source_sql("m", "c")} AS date_source,
                   c.title AS conversation_title
            FROM memory_fts fts, memories m
            LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE {where_clause}
            ORDER BY bm25(memory_fts)
            LIMIT ?
        """
        params: list[object] = [fts_expr, *filter_params, limit]
        rows = conn.execute(sql, params).fetchall()
        return [_load_record(conn, row) for row in rows]

    # Require all terms first (precision); if that finds nothing and there were multiple
    # terms, fall back to matching any term (recall) so multi-word questions like
    # "SQLite FAISS storage" still surface relevant memories — and the planner can reason.
    records = _run(" AND ".join(quoted))
    if not records and len(quoted) > 1:
        records = _run(" OR ".join(quoted))
    return records


def list_memories(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    status: str | None = None,
    project: str | None = None,
    limit: int = 50,
) -> list[MemoryRecord]:
    """List memories with optional filters."""
    conditions: list[str] = []
    params: list[object] = []

    if kind is not None:
        conditions.append("m.kind = ?")
        params.append(kind)

    if status is not None:
        conditions.append("m.status = ?")
        params.append(status)

    if project is not None:
        conditions.append("m.project = ?")
        params.append(project)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    sql = f"""
        {_MEMORY_SELECT}
        {where_clause}
        ORDER BY (m.created_at IS NULL), m.created_at, m.memory_id
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [_load_record(conn, row) for row in rows]


def get_memory(conn: sqlite3.Connection, memory_id: int) -> MemoryRecord | None:
    """Get a single memory by ID."""
    row = conn.execute(
        f"""
        {_MEMORY_SELECT}
        WHERE m.memory_id = ?
        """,
        (memory_id,),
    ).fetchone()

    if row is None:
        return None
    return _load_record(conn, row)


def decision_timeline(conn: sqlite3.Connection, subject: str) -> list[MemoryRecord]:
    """Get all decision memories for a subject, ordered chronologically."""
    rows = conn.execute(
        f"""
        {_MEMORY_SELECT}
        WHERE m.kind = 'decision'
          AND (m.subject_key LIKE ? COLLATE NOCASE OR m.statement LIKE ? COLLATE NOCASE)
        ORDER BY (m.created_at IS NULL), m.created_at, m.message_id
        """,
        (f"%{subject}%", f"%{subject}%"),
    ).fetchall()
    return [_load_record(conn, row) for row in rows]
