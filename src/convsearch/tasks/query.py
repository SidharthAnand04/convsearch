from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from convsearch.utils import memory_effective_timestamp_source_sql, memory_effective_timestamp_sql

# Statuses that should never surface as an "open"/"completed" task: an invalidated or
# superseded memory is no longer a live obligation, regardless of its task_state.
_EXCLUDED_STATUSES = ("invalidated", "superseded")
_EXCLUDED_STATUS_SQL = "status NOT IN ({})".format(",".join("?" * len(_EXCLUDED_STATUSES)))


@dataclass(frozen=True)
class TaskEvidence:
    quote: str
    conversation_id: int
    conversation_title: str | None
    message_id: int
    passage_id: int | None
    role: str
    timestamp: str | None


@dataclass(frozen=True)
class TaskItem:
    memory_id: int
    statement: str
    project: str | None
    task_state: str | None
    status: str
    confidence: float
    created_at: str | None
    conversation_id: int
    conversation_title: str | None
    # `evidence` and `evidence_count` answer two different questions and must not be
    # conflated:
    #   - `evidence_count` is the TRUTH: how many memory_evidence rows exist for this task,
    #     regardless of whether list_tasks was asked to fetch them. It is always populated.
    #   - `evidence` is what was actually LOADED this call: non-empty only when list_tasks
    #     was called with include_evidence=True. It can be `()` even when evidence_count > 0
    #     -- that means "evidence exists but wasn't fetched", not "no evidence".
    # `has_evidence` is derived from `evidence_count`, never from `len(evidence)`, precisely
    # so it stays truthful when include_evidence=False. Getting this backwards is the exact
    # bug this dataclass exists to prevent from being reintroduced.
    evidence: tuple[TaskEvidence, ...] = ()
    evidence_count: int = 0
    # 'created' (real creation date, possibly inherited from the conversation), 'captured'
    # (no creation date exists -- created_at above is capture time, i.e. when convsearch
    # first saw it), or 'unknown' (neither is available). Callers should label the date
    # accordingly rather than presenting a capture time as a creation date.
    date_source: str = "unknown"
    # Timestamp of the most recent set_task_state() transition (CLI `tasks complete/reopen`
    # or the /tasks/{id}/complete|reopen endpoints), or None if task_state has never been
    # changed that way -- see memory/store.py:set_task_state and migration 009.
    task_state_changed_at: str | None = None
    # 'user' if task_state_changed_at is set (a real, user-driven transition), 'heuristic' if
    # task_state is whatever the extractor produced (or nothing) and no one has acted on it.
    task_state_source: str = "heuristic"

    @property
    def has_evidence(self) -> bool:
        """True when evidence exists for this task, whether or not it was loaded.

        A task with no evidence is still returned by list_tasks (callers must not silently
        drop it), but the UI should flag it distinctly since it lacks a traceable quote.
        """
        return self.evidence_count > 0


@dataclass(frozen=True)
class TaskList:
    items: tuple[TaskItem, ...]
    total_open: int
    total_completed: int
    projects: tuple[str, ...]


def _fetch_evidence(
    conn: sqlite3.Connection,
    memory_ids: list[int],
) -> dict[int, list[TaskEvidence]]:
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
            msg.role,
            msg.created_at AS message_created_at,
            c.title AS conversation_title
        FROM memory_evidence me
        JOIN messages msg ON msg.message_id = me.message_id
        JOIN conversations c ON c.conversation_id = msg.conversation_id
        WHERE me.memory_id IN ({placeholders})
        ORDER BY me.evidence_id
        """,
        memory_ids,
    ).fetchall()

    result: dict[int, list[TaskEvidence]] = {mid: [] for mid in memory_ids}
    for row in rows:
        evidence = TaskEvidence(
            quote=row["quote"],
            conversation_id=row["conversation_id"],
            conversation_title=row["conversation_title"],
            message_id=row["message_id"],
            passage_id=row["passage_id"],
            role=row["role"],
            timestamp=row["message_created_at"],
        )
        result[row["memory_id"]].append(evidence)
    return result


def _evidence_counts(conn: sqlite3.Connection, memory_ids: list[int]) -> dict[int, int]:
    """Cheap COUNT of memory_evidence rows per memory_id, without fetching quotes.

    Used whenever we need to know evidence *existence* but not its content -- notably when
    include_evidence=False skipped the full fetch, in which case this is the only thing
    that can make `TaskItem.has_evidence` truthful (one grouped COUNT, no joins to messages
    or conversations).
    """
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    rows = conn.execute(
        f"SELECT memory_id, COUNT(*) AS n FROM memory_evidence "
        f"WHERE memory_id IN ({placeholders}) GROUP BY memory_id",
        memory_ids,
    ).fetchall()
    return {row["memory_id"]: row["n"] for row in rows}


def list_tasks(
    conn: sqlite3.Connection,
    *,
    state: str = "open",
    project: str | None = None,
    limit: int = 50,
    since: datetime | None = None,
    include_evidence: bool = True,
) -> TaskList:
    """Return a task inbox: memories of kind='task', newest-first.

    `state` selects "open", "completed", or "all" task_state values. `since` filters on
    memories.created_at (inclusive). Memories with status 'invalidated' or 'superseded' are
    excluded by default, since an invalidated task is not an open task.

    A task with no memory_evidence rows is still returned, with an empty `evidence` tuple —
    callers decide what to do with it, but a UI surfacing task inboxes should visually flag
    evidence-less tasks (see TaskItem.has_evidence) rather than presenting them as equally
    trustworthy.

    Fetching is two queries total regardless of result size: one for the task rows, one
    batched lookup for all their evidence (no N+1 per task).
    """
    if state not in ("open", "completed", "all"):
        raise ValueError(f"invalid state: {state!r}")

    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")

    clauses = ["kind = 'task'", _EXCLUDED_STATUS_SQL]
    params: list[object] = list(_EXCLUDED_STATUSES)

    if state != "all":
        clauses.append("task_state = ?")
        params.append(state)

    if project is not None:
        clauses.append("LOWER(project) = LOWER(?)")
        params.append(project)

    if since is not None:
        clauses.append(f"{ts} >= ?")
        params.append(since.isoformat())

    where_sql = " AND ".join(clauses)
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT
            m.memory_id,
            m.statement,
            m.project,
            m.task_state,
            m.status,
            m.confidence,
            {ts} AS created_at,
            {ts_source} AS date_source,
            m.conversation_id,
            c.title AS conversation_title,
            m.task_state_changed_at
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {where_sql}
        ORDER BY ({ts} IS NULL), {ts} DESC, m.memory_id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    memory_ids = [row["memory_id"] for row in rows]
    evidence_map = _fetch_evidence(conn, memory_ids) if include_evidence else {}
    # Evidence existence must be reported truthfully regardless of include_evidence, so
    # count it separately rather than reusing len(evidence_map[...]) which would be empty
    # (and wrong) whenever include_evidence=False. When evidence was already fetched in
    # full, its length IS the count, so no extra query is needed.
    counts_map = (
        {mid: len(ev) for mid, ev in evidence_map.items()}
        if include_evidence
        else _evidence_counts(conn, memory_ids)
    )

    items = tuple(
        TaskItem(
            memory_id=row["memory_id"],
            statement=row["statement"],
            project=row["project"],
            task_state=row["task_state"],
            status=row["status"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            conversation_id=row["conversation_id"],
            conversation_title=row["conversation_title"],
            evidence=tuple(evidence_map.get(row["memory_id"], [])),
            evidence_count=counts_map.get(row["memory_id"], 0),
            date_source=row["date_source"],
            task_state_changed_at=row["task_state_changed_at"],
            task_state_source="user" if row["task_state_changed_at"] else "heuristic",
        )
        for row in rows
    )

    # --- aggregate counts, independent of the paginated `items` above ---
    count_clauses = ["m.kind = 'task'", _EXCLUDED_STATUS_SQL.replace("status", "m.status")]
    count_params: list[object] = list(_EXCLUDED_STATUSES)
    if project is not None:
        count_clauses.append("LOWER(m.project) = LOWER(?)")
        count_params.append(project)
    if since is not None:
        count_clauses.append(f"{ts} >= ?")
        count_params.append(since.isoformat())
    count_where_sql = " AND ".join(count_clauses)

    count_row = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN m.task_state = 'open' THEN 1 ELSE 0 END) AS total_open,
            SUM(CASE WHEN m.task_state = 'completed' THEN 1 ELSE 0 END) AS total_completed
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {count_where_sql}
        """,
        count_params,
    ).fetchone()

    project_rows = conn.execute(
        f"""
        SELECT DISTINCT m.project AS project
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE {count_where_sql} AND m.project IS NOT NULL AND m.project != ''
        ORDER BY project
        """,
        count_params,
    ).fetchall()

    return TaskList(
        items=items,
        total_open=count_row["total_open"] or 0,
        total_completed=count_row["total_completed"] or 0,
        projects=tuple(row["project"] for row in project_rows),
    )


def get_task(conn: sqlite3.Connection, memory_id: int) -> TaskItem | None:
    """Re-read a single task in the TaskItem shape, e.g. after set_task_state.

    Returns None if the memory doesn't exist or isn't kind='task' -- callers (the HTTP
    handlers) turn that into 404 rather than fabricating a task-shaped response for a
    non-task memory. Unlike `list_tasks`, this does not exclude invalidated/superseded
    memories: the caller just mutated this exact row and needs to see its true state.
    """
    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")
    row = conn.execute(
        f"""
        SELECT
            m.memory_id,
            m.statement,
            m.project,
            m.task_state,
            m.status,
            m.confidence,
            {ts} AS created_at,
            {ts_source} AS date_source,
            m.conversation_id,
            c.title AS conversation_title,
            m.task_state_changed_at
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.memory_id = ? AND m.kind = 'task'
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        return None

    evidence = _fetch_evidence(conn, [memory_id]).get(memory_id, [])
    return TaskItem(
        memory_id=row["memory_id"],
        statement=row["statement"],
        project=row["project"],
        task_state=row["task_state"],
        status=row["status"],
        confidence=row["confidence"],
        created_at=row["created_at"],
        conversation_id=row["conversation_id"],
        conversation_title=row["conversation_title"],
        evidence=tuple(evidence),
        evidence_count=len(evidence),
        date_source=row["date_source"],
        task_state_changed_at=row["task_state_changed_at"],
        task_state_source="user" if row["task_state_changed_at"] else "heuristic",
    )
