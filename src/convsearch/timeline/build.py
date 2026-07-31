from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from convsearch.memory.search import search_memories
from convsearch.utils import memory_effective_timestamp_source_sql, memory_effective_timestamp_sql


@dataclass(frozen=True)
class TimelineEvidence:
    quote: str
    conversation_id: int
    conversation_title: str | None
    message_id: int
    timestamp: str | None


@dataclass(frozen=True)
class TimelineNode:
    memory_id: int
    kind: str
    statement: str
    status: str
    project: str | None
    created_at: str | None
    confidence: float
    conversation_id: int
    conversation_title: str | None
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    evidence: tuple[TimelineEvidence, ...] = ()
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- created_at above is capture time, i.e. when convsearch first
    # saw it), or 'unknown' (neither is available). Lets a caller render "first captured 29
    # Jul" rather than implying an authorship date it does not have.
    date_source: str = "unknown"


@dataclass(frozen=True)
class Timeline:
    topic: str
    nodes: tuple[TimelineNode, ...]
    active: tuple[TimelineNode, ...]
    superseded: tuple[TimelineNode, ...]
    contested: tuple[TimelineNode, ...]
    rejected: tuple[TimelineNode, ...]
    first_seen: str | None
    last_seen: str | None
    matched_count: int
    # Provenance of first_seen/last_seen: see TimelineNode.date_source. None when the
    # corresponding *_seen value is itself None.
    first_seen_source: str | None = None
    last_seen_source: str | None = None


def _empty_timeline(topic: str) -> Timeline:
    return Timeline(
        topic=topic,
        nodes=(),
        active=(),
        superseded=(),
        contested=(),
        rejected=(),
        first_seen=None,
        last_seen=None,
        matched_count=0,
    )


def _sort_key(row: sqlite3.Row) -> tuple[bool, str, int]:
    """Oldest-first sort key on the effective timestamp: NULL sorts last, tie-break on
    memory_id."""
    created_at: str | None = row["created_at"]
    return (created_at is None, created_at or "", row["memory_id"])


def _fetch_memory_rows(
    conn: sqlite3.Connection,
    memory_ids: set[int],
    *,
    project: str | None,
) -> list[sqlite3.Row]:
    if not memory_ids:
        return []
    ids = sorted(memory_ids)
    placeholders = ",".join("?" for _ in ids)
    conditions = [f"m.memory_id IN ({placeholders})"]
    params: list[object] = list(ids)
    if project is not None:
        conditions.append("m.project = ?")
        params.append(project)
    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")
    sql = f"""
        SELECT m.memory_id, m.kind, m.statement, m.status, m.confidence,
               m.project, m.conversation_id, m.message_id,
               {ts} AS created_at, {ts_source} AS date_source,
               c.title AS conversation_title
        FROM memories m
        LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {" AND ".join(conditions)}
    """
    return conn.execute(sql, params).fetchall()


def _fetch_relation_rows(conn: sqlite3.Connection, memory_ids: set[int]) -> list[sqlite3.Row]:
    if not memory_ids:
        return []
    ids = sorted(memory_ids)
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT mr.relation_id, mr.from_memory_id, mr.to_memory_id, mr.relation, mr.reason,
               mf.statement AS from_statement, mt.statement AS to_statement
        FROM memory_relations mr
        JOIN memories mf ON mf.memory_id = mr.from_memory_id
        JOIN memories mt ON mt.memory_id = mr.to_memory_id
        WHERE mr.from_memory_id IN ({placeholders}) OR mr.to_memory_id IN ({placeholders})
        ORDER BY mr.relation_id
    """
    return conn.execute(sql, [*ids, *ids]).fetchall()


def _fetch_evidence(
    conn: sqlite3.Connection,
    memory_ids: set[int],
) -> dict[int, list[TimelineEvidence]]:
    if not memory_ids:
        return {}
    ids = sorted(memory_ids)
    placeholders = ",".join("?" for _ in ids)
    sql = f"""
        SELECT me.memory_id, me.quote, me.message_id,
               msg.conversation_id, msg.created_at AS message_created_at,
               c.title AS conversation_title
        FROM memory_evidence me
        JOIN messages msg ON msg.message_id = me.message_id
        JOIN conversations c ON c.conversation_id = msg.conversation_id
        WHERE me.memory_id IN ({placeholders})
        ORDER BY me.evidence_id
    """
    rows = conn.execute(sql, ids).fetchall()
    result: dict[int, list[TimelineEvidence]] = {mid: [] for mid in ids}
    for row in rows:
        result[row["memory_id"]].append(
            TimelineEvidence(
                quote=row["quote"],
                conversation_id=row["conversation_id"],
                conversation_title=row["conversation_title"],
                message_id=row["message_id"],
                timestamp=row["message_created_at"],
            )
        )
    return result


def build_timeline(
    conn: sqlite3.Connection,
    topic: str,
    *,
    project: str | None = None,
    limit: int = 40,
    include_evidence: bool = True,
) -> Timeline:
    """Build a topic-scoped decision timeline: how an idea changed over time.

    `topic` is resolved to a candidate memory set via the existing memory FTS search
    (``convsearch.memory.search.search_memories``). The candidate set is then expanded
    exactly one hop along ``memory_relations`` (relation = 'supersedes'): for every
    matched memory, both the memory it supersedes and the memory that supersedes it are
    pulled in. This ensures a superseded predecessor is included in the timeline even
    when its statement text no longer matches the query (e.g. the old approach's name
    was replaced along with the decision itself). Expansion only follows 'supersedes'
    edges, not 'conflicts_with'/'relates_to'/'depends_on' — those are surfaced as
    annotations on nodes already in the timeline rather than pulling in new nodes.

    Read-only: issues no writes, no LLM/network calls. All ordering is deterministic,
    tie-broken on memory_id. Returns an empty Timeline (matched_count=0) rather than
    raising when the topic matches nothing.
    """
    if not topic or not topic.strip():
        return _empty_timeline(topic)

    candidates = search_memories(conn, topic, project=project, limit=limit)
    if not candidates:
        return _empty_timeline(topic)

    matched_count = len(candidates)
    candidate_ids = {c.memory_id for c in candidates}

    # One-hop expansion along 'supersedes' relations (see docstring).
    expanded_ids = set(candidate_ids)
    hop_rows = _fetch_relation_rows(conn, candidate_ids)
    for row in hop_rows:
        if row["relation"] == "supersedes":
            expanded_ids.add(row["from_memory_id"])
            expanded_ids.add(row["to_memory_id"])

    memory_rows = _fetch_memory_rows(conn, expanded_ids, project=project)
    if not memory_rows:
        return _empty_timeline(topic)

    kept_ids = {row["memory_id"] for row in memory_rows}

    # Re-fetch relations over the final node set so supersedes/conflicts_with/reasons
    # cover every node actually in the timeline, not just the original candidates.
    relation_rows = _fetch_relation_rows(conn, kept_ids)

    supersedes: dict[int, list[str]] = {mid: [] for mid in kept_ids}
    superseded_by: dict[int, list[str]] = {mid: [] for mid in kept_ids}
    conflicts_with: dict[int, list[str]] = {mid: [] for mid in kept_ids}
    reasons: dict[int, list[str]] = {mid: [] for mid in kept_ids}
    rejected_ids: set[int] = set()

    for row in relation_rows:
        from_id: int = row["from_memory_id"]
        to_id: int = row["to_memory_id"]
        relation: str = row["relation"]
        reason: str | None = row["reason"]

        if relation == "supersedes":
            if from_id in kept_ids and row["to_statement"] not in supersedes[from_id]:
                supersedes[from_id].append(row["to_statement"])
            if to_id in kept_ids and row["from_statement"] not in superseded_by[to_id]:
                superseded_by[to_id].append(row["from_statement"])
            if to_id in kept_ids:
                rejected_ids.add(to_id)
            if reason:
                if from_id in kept_ids and reason not in reasons[from_id]:
                    reasons[from_id].append(reason)
                if to_id in kept_ids and reason not in reasons[to_id]:
                    reasons[to_id].append(reason)
        elif relation == "conflicts_with":
            if from_id in kept_ids and row["to_statement"] not in conflicts_with[from_id]:
                conflicts_with[from_id].append(row["to_statement"])
            if to_id in kept_ids and row["from_statement"] not in conflicts_with[to_id]:
                conflicts_with[to_id].append(row["from_statement"])

    evidence_map: dict[int, list[TimelineEvidence]] = (
        _fetch_evidence(conn, kept_ids) if include_evidence else {}
    )

    sorted_rows = sorted(memory_rows, key=_sort_key)

    nodes: list[TimelineNode] = []
    for row in sorted_rows:
        mid: int = row["memory_id"]
        nodes.append(
            TimelineNode(
                memory_id=mid,
                kind=row["kind"],
                statement=row["statement"],
                status=row["status"],
                project=row["project"],
                created_at=row["created_at"],
                confidence=row["confidence"],
                conversation_id=row["conversation_id"],
                conversation_title=row["conversation_title"],
                supersedes=tuple(supersedes[mid]),
                superseded_by=tuple(superseded_by[mid]),
                conflicts_with=tuple(conflicts_with[mid]),
                reasons=tuple(reasons[mid]),
                evidence=tuple(evidence_map.get(mid, [])),
                date_source=row["date_source"],
            )
        )

    nodes_t = tuple(nodes)
    active = tuple(n for n in nodes if n.status == "active")
    superseded_nodes = tuple(n for n in nodes if n.status == "superseded")
    contested = tuple(n for n in nodes if n.status == "contested")
    rejected = tuple(
        n
        for n in nodes
        if n.status in ("superseded", "invalidated") and n.memory_id in rejected_ids
    )

    dated_nodes = [n for n in nodes if n.created_at is not None]
    first_node = min(dated_nodes, key=lambda n: (n.created_at, n.memory_id), default=None)
    last_node = max(dated_nodes, key=lambda n: (n.created_at, n.memory_id), default=None)

    return Timeline(
        topic=topic,
        nodes=nodes_t,
        active=active,
        superseded=superseded_nodes,
        contested=contested,
        rejected=rejected,
        first_seen=first_node.created_at if first_node else None,
        last_seen=last_node.created_at if last_node else None,
        matched_count=matched_count,
        first_seen_source=first_node.date_source if first_node else None,
        last_seen_source=last_node.date_source if last_node else None,
    )
