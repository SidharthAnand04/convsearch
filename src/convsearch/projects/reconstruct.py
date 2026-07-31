from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from convsearch.utils import (
    memory_effective_timestamp_source_sql,
    memory_effective_timestamp_sql,
)


@dataclass(frozen=True)
class EvidenceRef:
    memory_id: int
    conversation_id: int
    conversation_title: str | None
    message_id: int
    passage_id: int | None
    quote: str


@dataclass(frozen=True)
class SupersededBy:
    """The memory that replaced a superseded decision, and why (when recorded).

    Mirrors the shape of convsearch.memory.review.SupersessionRef, extended with a `reason`
    field: the reviewer view only needs to say *what* replaced a memory, but a project report
    exists to answer "what replaced it, and why", so the reason is load-bearing here.
    """

    memory_id: int
    statement: str
    reason: str | None = None


@dataclass(frozen=True)
class ProjectItem:
    memory_id: int
    statement: str
    status: str
    created_at: str | None
    subject_key: str
    evidence: tuple[EvidenceRef, ...]
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- created_at above is capture time), or 'unknown' (neither is
    # available). Same fallback chain as TimelineEntry.date_source -- see convsearch.utils.
    date_source: str = "unknown"
    # Populated only for decisions with status in ('superseded', 'invalidated', 'historical');
    # None when no 'supersedes' relation targets this memory (nothing replaced it, or the link
    # was never recorded). See _fetch_superseded_by below for how this is populated.
    superseded_by: SupersededBy | None = None


@dataclass(frozen=True)
class TimelineEntry:
    created_at: str | None
    kind: str
    statement: str
    status: str
    memory_id: int
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- created_at above is capture time), or 'unknown' (neither
    # is available). See convsearch.utils for the shared fallback chain this is derived from.
    date_source: str = "unknown"


@dataclass(frozen=True)
class ProjectSummary:
    name: str
    memory_count: int
    conversation_count: int
    decision_count: int
    open_task_count: int
    last_activity: str | None
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- last_activity above is capture time), or 'unknown' (neither
    # is available). Same fallback chain as TimelineEntry.date_source -- see convsearch.utils.
    date_source: str = "unknown"


@dataclass(frozen=True)
class ProjectReport:
    name: str
    summary: str
    timeline: tuple[TimelineEntry, ...]
    architecture: tuple[ProjectItem, ...]
    decisions: tuple[ProjectItem, ...]
    superseded_decisions: tuple[ProjectItem, ...]
    rejected_alternatives: tuple[str, ...]
    open_tasks: tuple[ProjectItem, ...]
    completed_tasks: tuple[ProjectItem, ...]
    risks: tuple[ProjectItem, ...]
    conversations: tuple[tuple[int, str | None], ...]
    evidence_count: int
    # Additive fields. Stored as JSON-native dicts (same shape the server emits for a
    # ProjectItem) so the report serializes cleanly via json.dumps without a custom encoder.
    known_bugs: tuple[dict[str, object], ...] = ()
    next_milestones: tuple[dict[str, object], ...] = ()


def list_projects(conn: sqlite3.Connection) -> list[ProjectSummary]:
    """Return one ProjectSummary per distinct non-null/non-empty project name."""
    rows = conn.execute(
        """
        SELECT
            m.project AS project,
            COUNT(*) AS memory_count,
            COUNT(DISTINCT m.conversation_id) AS conversation_count,
            SUM(CASE WHEN m.kind = 'decision' THEN 1 ELSE 0 END) AS decision_count,
            SUM(
                CASE WHEN m.kind = 'task' AND m.task_state = 'open' THEN 1 ELSE 0 END
            ) AS open_task_count
        FROM memories m
        LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.project IS NOT NULL AND m.project != ''
        GROUP BY m.project
        ORDER BY m.project
        """
    ).fetchall()

    # `last_activity` uses the same fallback chain as the timeline/reconstruct queries below
    # (memory's own created_at, then its conversation's created_at, then capture time) --
    # otherwise this column is blank for every project whose memories all came from
    # live-captured conversations with no creation date of their own. A window function (not
    # a plain GROUP BY MAX) so the date_source reported is the source of the specific row
    # that determined last_activity, not an unrelated one -- same pattern as
    # digest.build._new_projects.
    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")
    latest_rows = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                m.project AS project,
                {ts} AS eff_created_at,
                {ts_source} AS date_source,
                ROW_NUMBER() OVER (
                    PARTITION BY m.project ORDER BY {ts} DESC, m.memory_id DESC
                ) AS rn
            FROM memories m
            LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE m.project IS NOT NULL AND m.project != '' AND {ts} IS NOT NULL
        )
        SELECT project, eff_created_at, date_source FROM ranked WHERE rn = 1
        """
    ).fetchall()
    latest_by_project = {
        row["project"]: (row["eff_created_at"], row["date_source"]) for row in latest_rows
    }

    result: list[ProjectSummary] = []
    for row in rows:
        last_activity, date_source = latest_by_project.get(row["project"], (None, "unknown"))
        result.append(
            ProjectSummary(
                name=row["project"],
                memory_count=row["memory_count"],
                conversation_count=row["conversation_count"],
                decision_count=row["decision_count"],
                open_task_count=row["open_task_count"],
                last_activity=last_activity,
                date_source=date_source,
            )
        )
    return result


def _fetch_evidence(
    conn: sqlite3.Connection,
    memory_ids: list[int],
) -> dict[int, list[EvidenceRef]]:
    """Fetch all memory_evidence rows for the given memory_ids, keyed by memory_id."""
    if not memory_ids:
        return {}

    placeholders = ",".join("?" * len(memory_ids))
    rows = conn.execute(
        f"""
        SELECT
            me.memory_id,
            me.passage_id,
            me.message_id,
            me.quote,
            msg.conversation_id,
            c.title AS conversation_title
        FROM memory_evidence me
        JOIN messages msg ON msg.message_id = me.message_id
        JOIN conversations c ON c.conversation_id = msg.conversation_id
        WHERE me.memory_id IN ({placeholders})
        ORDER BY me.evidence_id
        """,
        memory_ids,
    ).fetchall()

    result: dict[int, list[EvidenceRef]] = {mid: [] for mid in memory_ids}
    for row in rows:
        ref = EvidenceRef(
            memory_id=row["memory_id"],
            conversation_id=row["conversation_id"],
            conversation_title=row["conversation_title"],
            message_id=row["message_id"],
            passage_id=row["passage_id"],
            quote=row["quote"],
        )
        result[row["memory_id"]].append(ref)
    return result


def _to_project_items(
    rows: list[sqlite3.Row],
    evidence_map: dict[int, list[EvidenceRef]],
    superseded_by_map: dict[int, SupersededBy] | None = None,
) -> tuple[ProjectItem, ...]:
    items: list[ProjectItem] = []
    for row in rows:
        mid = row["memory_id"]
        items.append(
            ProjectItem(
                memory_id=mid,
                statement=row["statement"],
                status=row["status"],
                created_at=row["effective_created_at"],
                subject_key=row["subject_key"],
                evidence=tuple(evidence_map.get(mid, [])),
                date_source=row["date_source"],
                superseded_by=(superseded_by_map or {}).get(mid),
            )
        )
    return tuple(items)


def _fetch_superseded_by(
    conn: sqlite3.Connection,
    memory_ids: list[int],
) -> dict[int, SupersededBy]:
    """Batched lookup of the 'supersedes' relation targeting each of the given memory_ids.

    Mirrors the join shape used by convsearch.timeline.build._fetch_relation_rows: a
    memory_relations row with relation='supersedes' has from_memory_id as the replacement
    and to_memory_id as the memory it replaced, plus an optional `reason`. One query for all
    of `memory_ids` rather than one per decision. When more than one relation targets the same
    memory (not expected in practice), the earliest (lowest relation_id) wins, for
    determinism.
    """
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"""
        SELECT mr.relation_id, mr.to_memory_id, mr.reason,
               mf.memory_id AS from_memory_id, mf.statement AS from_statement
        FROM memory_relations mr
        JOIN memories mf ON mf.memory_id = mr.from_memory_id
        WHERE mr.relation = 'supersedes' AND mr.to_memory_id IN ({placeholders})
        ORDER BY mr.relation_id
        """,
        memory_ids,
    ).fetchall()
    result: dict[int, SupersededBy] = {}
    for row in rows:
        to_id: int = row["to_memory_id"]
        if to_id in result:
            continue
        result[to_id] = SupersededBy(
            memory_id=row["from_memory_id"],
            statement=row["from_statement"],
            reason=row["reason"],
        )
    return result


# Keyword signals used to derive the additive known_bugs / next_milestones buckets.
_BUG_KEYWORDS = ("bug", "broken", "fails", "regression", "error")
_MILESTONE_KEYWORDS = ("next", "milestone", "plan to", "will", "todo", "roadmap")


def _text_matches(row: sqlite3.Row, keywords: tuple[str, ...], *, include_metadata: bool) -> bool:
    """True when the row's statement (and optionally metadata_json) contains any keyword."""
    text = (row["statement"] or "").lower()
    if include_metadata:
        text = f"{text} {(row['metadata_json'] or '').lower()}"
    return any(kw in text for kw in keywords)


def _to_item_dicts(
    rows: list[sqlite3.Row],
    evidence_map: dict[int, list[EvidenceRef]],
) -> tuple[dict[str, object], ...]:
    """Build JSON-native item dicts (mirrors the server's ProjectItem payload shape)."""
    items: list[dict[str, object]] = []
    for row in rows:
        mid = row["memory_id"]
        items.append(
            {
                "memory_id": mid,
                "statement": row["statement"],
                "status": row["status"],
                "created_at": row["effective_created_at"],
                "date_source": row["date_source"],
                "subject_key": row["subject_key"],
                "evidence": [
                    {
                        "memory_id": ev.memory_id,
                        "conversation_id": ev.conversation_id,
                        "conversation_title": ev.conversation_title,
                        "message_id": ev.message_id,
                        "passage_id": ev.passage_id,
                        "quote": ev.quote,
                    }
                    for ev in evidence_map.get(mid, [])
                ],
            }
        )
    return tuple(items)


def _sort_key(row: sqlite3.Row) -> tuple[bool, str, int]:
    """Sort key for newest-first ordering of memory rows, using the effective timestamp so
    ordering matches what is actually displayed (not the possibly-NULL raw created_at)."""
    ts: str = row["effective_created_at"] if row["effective_created_at"] else ""
    return (row["effective_created_at"] is None, ts, row["memory_id"])


def reconstruct_project(conn: sqlite3.Connection, name: str) -> ProjectReport | None:
    """
    Reconstruct a ProjectReport for the given project name (case-insensitive).
    Returns None when no memories exist for that project.
    """
    # Fetch all memories for the project (case-insensitive match). Joined against
    # conversations so the timeline can fall back to capture time (see
    # memory_effective_timestamp_sql) instead of printing a blank date for live-captured
    # memories that have no creation timestamp of their own.
    all_rows = conn.execute(
        f"""
        SELECT m.memory_id, m.kind, m.subject_key, m.statement, m.status, m.task_state,
               m.created_at, m.metadata_json,
               {memory_effective_timestamp_sql("m", "c")} AS effective_created_at,
               {memory_effective_timestamp_source_sql("m", "c")} AS date_source
        FROM memories m
        LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE LOWER(m.project) = LOWER(?)
        ORDER BY (effective_created_at IS NULL), effective_created_at, m.memory_id
        """,
        (name,),
    ).fetchall()

    if not all_rows:
        return None

    # Collect all memory IDs for evidence lookup
    all_ids = [row["memory_id"] for row in all_rows]
    evidence_map = _fetch_evidence(conn, all_ids)

    # --- timeline: all memories ordered by (created_at IS NULL, created_at, memory_id) ---
    timeline = tuple(
        TimelineEntry(
            created_at=row["effective_created_at"],
            kind=row["kind"],
            statement=row["statement"],
            status=row["status"],
            memory_id=row["memory_id"],
            date_source=row["date_source"],
        )
        for row in all_rows
    )

    # --- architecture: kind='project_state', all statuses, newest first ---
    arch_rows = [r for r in all_rows if r["kind"] == "project_state"]
    arch_rows_sorted = sorted(arch_rows, key=_sort_key, reverse=True)
    architecture = _to_project_items(arch_rows_sorted, evidence_map)

    # --- decisions ---
    active_decision_rows = [
        r for r in all_rows if r["kind"] == "decision" and r["status"] in ("active", "contested")
    ]
    superseded_decision_rows = [
        r
        for r in all_rows
        if r["kind"] == "decision" and r["status"] in ("superseded", "invalidated", "historical")
    ]
    superseded_by_map = _fetch_superseded_by(
        conn, [r["memory_id"] for r in superseded_decision_rows]
    )
    decisions = _to_project_items(active_decision_rows, evidence_map)
    superseded_decisions = _to_project_items(
        superseded_decision_rows, evidence_map, superseded_by_map
    )

    # --- rejected_alternatives: from metadata_json of ALL decision rows ---
    rejected_alts: list[str] = []
    seen_alts: set[str] = set()
    for row in all_rows:
        if row["kind"] != "decision":
            continue
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        alt = meta.get("rejected_alternative")
        if isinstance(alt, str) and alt and alt not in seen_alts:
            rejected_alts.append(alt)
            seen_alts.add(alt)

    # --- tasks ---
    open_task_rows = [r for r in all_rows if r["kind"] == "task" and r["task_state"] == "open"]
    completed_task_rows = [
        r for r in all_rows if r["kind"] == "task" and r["task_state"] == "completed"
    ]
    open_tasks = _to_project_items(open_task_rows, evidence_map)
    completed_tasks = _to_project_items(completed_task_rows, evidence_map)

    # --- risks ---
    risk_rows = [r for r in all_rows if r["kind"] == "risk"]
    risks = _to_project_items(risk_rows, evidence_map)

    # --- known_bugs: any memory whose statement/metadata reads as a bug (risks included) ---
    bug_rows = [r for r in all_rows if _text_matches(r, _BUG_KEYWORDS, include_metadata=True)]
    known_bugs = _to_item_dicts(bug_rows, evidence_map)

    # --- next_milestones: open tasks / active project_state describing planned future work ---
    milestone_rows = [
        r
        for r in all_rows
        if (
            (r["kind"] == "task" and r["task_state"] == "open")
            or (r["kind"] == "project_state" and r["status"] == "active")
        )
        and _text_matches(r, _MILESTONE_KEYWORDS, include_metadata=False)
    ]
    next_milestones = _to_item_dicts(milestone_rows, evidence_map)

    # --- conversations: distinct (conversation_id, title) touched by the project ---
    conv_rows = conn.execute(
        """
        SELECT DISTINCT c.conversation_id, c.title
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE LOWER(m.project) = LOWER(?)
        ORDER BY c.conversation_id
        """,
        (name,),
    ).fetchall()
    conversations: tuple[tuple[int, str | None], ...] = tuple(
        (row["conversation_id"], row["title"]) for row in conv_rows
    )

    # --- summary ---
    summary_str: str
    # Find most recent active project_state memory
    active_state_rows = [r for r in arch_rows if r["status"] == "active"]
    if active_state_rows:
        newest_active = sorted(active_state_rows, key=_sort_key, reverse=True)[0]
        summary_str = newest_active["statement"]
    else:
        n_decisions = len(active_decision_rows) + len(superseded_decision_rows)
        n_open = len(open_task_rows)
        n_convs = len(conversations)
        summary_str = (
            f"{n_decisions} decision{'s' if n_decisions != 1 else ''}, "
            f"{n_open} open task{'s' if n_open != 1 else ''} "
            f"across {n_convs} conversation{'s' if n_convs != 1 else ''}"
        )

    # --- evidence_count ---
    total_evidence: int = sum(len(v) for v in evidence_map.values())

    return ProjectReport(
        name=name,
        summary=summary_str,
        timeline=timeline,
        architecture=architecture,
        decisions=decisions,
        superseded_decisions=superseded_decisions,
        rejected_alternatives=tuple(rejected_alts),
        open_tasks=open_tasks,
        completed_tasks=completed_tasks,
        risks=risks,
        conversations=conversations,
        evidence_count=total_evidence,
        known_bugs=known_bugs,
        next_milestones=next_milestones,
    )
