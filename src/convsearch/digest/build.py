"""User-facing digest: "what changed in my memory system recently?"

Fully deterministic and readable with no LLM involved -- that is the primary path, not a
fallback. Every section is derived from real timestamps (`memories.created_at` for newly
created rows, `memory_status_history.changed_at` for status transitions) so a decision that
was superseded this week is reported as a change this week even if the memory itself is old.

Read-only: no writes, no LLM, no network. Stdlib + sqlite3 only.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from convsearch.capture.inventory import list_captures
from convsearch.capture.state import CAPTURE_SOURCE_HASH
from convsearch.utils import (
    count_missing_effective_timestamps,
    count_missing_timestamps,
    effective_timestamp_source_sql,
    effective_timestamp_sql,
    format_dated,
    memory_effective_timestamp_source_sql,
    memory_effective_timestamp_sql,
    pluralize,
)

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_DURATION_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_duration(text: str) -> timedelta:
    """Parse a short duration string like "7d", "24h", "30m", "2w" into a timedelta.

    Raises ValueError for anything else (empty string, unknown unit, non-numeric amount).
    """
    match = _DURATION_RE.match(text)
    if not match:
        raise ValueError(
            f"invalid duration: {text!r} (expected a number followed by m/h/d/w, e.g. '7d')"
        )
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return timedelta(seconds=amount * _DURATION_UNIT_SECONDS[unit])


@dataclass(frozen=True)
class DigestSection:
    key: str
    title: str
    items: tuple[str, ...]
    count: int
    detail: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class Digest:
    since: str
    until: str
    window_label: str
    sections: tuple[DigestSection, ...]
    is_empty: bool
    headline: str
    unreliable_window: bool = False
    caveat: str = ""


def _window_label(since: datetime, until: datetime) -> str:
    """Deterministic human label for the window, e.g. "last 7 days".

    Rounds to the nearest whole second before picking a unit. Two independent
    `datetime.now()` calls (one for `--since`, one for the digest's own `until`
    default) can leave the raw delta a few microseconds off an otherwise-whole
    span -- e.g. 3650 days plus 40us -- which used to fall through to the
    fractional-day fallback and print "3650.0 days". Rounding first keeps the
    label whole-unit and pluralised whenever the span actually is whole.
    """
    total_seconds = round((until - since).total_seconds())
    if total_seconds <= 0:
        return "an empty window"

    days, day_remainder = divmod(total_seconds, 86400)
    if days >= 7 and day_remainder == 0 and days % 7 == 0:
        weeks = days // 7
        return f"last {weeks} week" + ("s" if weeks != 1 else "")
    if days >= 1 and day_remainder == 0:
        return f"last {days} day" + ("s" if days != 1 else "")

    hours, hour_remainder = divmod(total_seconds, 3600)
    if hours >= 1 and hour_remainder == 0:
        return f"last {hours} hour" + ("s" if hours != 1 else "")

    if days >= 1:
        # Not a whole number of days or hours: report the nearest whole day
        # rather than a fractional count.
        rounded_days = max(1, round(total_seconds / 86400))
        return f"last {rounded_days} day" + ("s" if rounded_days != 1 else "")

    minutes = max(1, round(total_seconds / 60))
    return f"last {minutes} minute" + ("s" if minutes != 1 else "")


def _section(
    key: str,
    title: str,
    items: list[str],
    detail: list[dict[str, object]],
    limit: int,
) -> DigestSection | None:
    """Build a section, or None if it has no items (never render an empty section)."""
    if not items:
        return None
    return DigestSection(
        key=key,
        title=title,
        items=tuple(items[:limit]),
        count=len(items),
        detail=tuple(detail[:limit]),
    )


def _pluralize(count: int, noun: str) -> str:
    """Render "N noun"/"N nouns" -- never the ungrammatical "N noun(s)".

    Thin wrapper over the shared helper so the digest's many call sites stay terse.
    """
    return pluralize(count, noun)


def _new_captures(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    ts = effective_timestamp_sql("c")
    ts_source = effective_timestamp_source_sql("c")
    rows = conn.execute(
        f"""
        SELECT
            c.conversation_id AS conversation_id,
            c.title AS title,
            {ts} AS effective_created_at,
            {ts_source} AS date_source,
            i.source_hash AS source_hash,
            (SELECT COUNT(*) FROM messages msg WHERE msg.conversation_id = c.conversation_id)
                AS message_count
        FROM conversations c
        JOIN imports i ON i.import_id = c.import_id
        WHERE {ts} IS NOT NULL AND {ts} BETWEEN ? AND ?
        ORDER BY {ts}, c.conversation_id
        """,
        (since_iso, until_iso),
    ).fetchall()

    items: list[str] = []
    detail: list[dict[str, object]] = []
    for row in rows:
        source = "live-capture" if row["source_hash"] == CAPTURE_SOURCE_HASH else "export-import"
        date_source: str = row["date_source"]
        # No per-row "[dated by capture time]" note here: it is appended in a second pass by
        # `_mark_mixed_provenance` in `build_digest`, and only when the digest's rows actually
        # mix "created" and "captured" provenance. When every dated row shares one provenance
        # the caveat line beneath the headline already says so once; repeating it per row is
        # noise, not information.
        items.append(f'"{row["title"]}" ({row["message_count"]} messages, {source})')
        detail.append(
            {
                "conversation_id": int(row["conversation_id"]),
                "title": row["title"],
                "created_at": row["effective_created_at"],
                "date_source": date_source,
                "source": source,
                "message_count": int(row["message_count"]),
            }
        )
    return _section("new_captures", "Conversations captured", items, detail, limit)


def _new_projects(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")
    # Window functions (not a plain GROUP BY MIN) so the date_source reported for a project
    # is the source of the specific row that determined its first_seen, not an unrelated one.
    rows = conn.execute(
        f"""
        WITH ranked AS (
            SELECT
                m.project AS project,
                {ts} AS eff_created_at,
                {ts_source} AS date_source,
                ROW_NUMBER() OVER (PARTITION BY m.project ORDER BY {ts}, m.memory_id) AS rn,
                COUNT(*) OVER (PARTITION BY m.project) AS memory_count
            FROM memories m
            JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE m.project IS NOT NULL AND m.project != '' AND {ts} IS NOT NULL
        )
        SELECT project, eff_created_at AS first_seen, date_source, memory_count
        FROM ranked
        WHERE rn = 1 AND eff_created_at BETWEEN ? AND ?
        ORDER BY first_seen, project
        """,
        (since_iso, until_iso),
    ).fetchall()

    items: list[str] = []
    detail: list[dict[str, object]] = []
    for row in rows:
        date_source: str = row["date_source"]
        label = "first captured" if date_source == "captured" else "first seen"
        # `format_dated` gives the same compact rendering used everywhere else in the CLI
        # ("30 Jul") instead of a raw ISO timestamp. Provenance is passed as "created" here
        # (regardless of the row's actual date_source) so format_dated never appends its own
        # "(captured)" suffix -- the `label` word immediately before it already says that.
        items.append(f'"{row["project"]}" ({label} {format_dated(row["first_seen"], "created")})')
        detail.append(
            {
                "project": row["project"],
                "first_seen": row["first_seen"],
                "date_source": date_source,
                "memory_count": int(row["memory_count"]),
            }
        )
    return _section("new_projects", "New projects", items, detail, limit)


def _memories_by_kind(
    conn: sqlite3.Connection,
    kind: str,
    since_iso: str,
    until_iso: str,
    extra_where: str = "",
    extra_params: tuple[object, ...] = (),
) -> list[sqlite3.Row]:
    ts = memory_effective_timestamp_sql("m", "c")
    ts_source = memory_effective_timestamp_source_sql("m", "c")
    return conn.execute(
        f"""
        SELECT
            m.memory_id, m.statement, m.project, m.status,
            {ts} AS created_at, {ts_source} AS date_source,
            m.conversation_id, c.title AS conversation_title
        FROM memories m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.kind = ? AND {ts} IS NOT NULL AND {ts} BETWEEN ? AND ?
          {extra_where}
        ORDER BY {ts}, m.memory_id
        """,
        (kind, since_iso, until_iso, *extra_params),
    ).fetchall()


def _new_decisions(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    rows = _memories_by_kind(conn, "decision", since_iso, until_iso)
    items = [f"{row['statement']} ({row['project'] or 'no project'})" for row in rows]
    detail = [
        {
            "memory_id": int(row["memory_id"]),
            "statement": row["statement"],
            "project": row["project"],
            "status": row["status"],
            "created_at": row["created_at"],
            "date_source": row["date_source"],
            "conversation_id": int(row["conversation_id"]),
            "conversation_title": row["conversation_title"],
        }
        for row in rows
    ]
    return _section("new_decisions", "New decisions", items, detail, limit)


def _superseded_decisions(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    """Decisions whose status became 'superseded' during the window.

    Uses memory_status_history.changed_at, not the memory's created_at: the memory itself
    may predate the window by a long way, but the transition is what makes it a "change".
    """
    rows = conn.execute(
        """
        SELECT
            h.history_id, h.memory_id, h.old_status, h.new_status, h.changed_at, h.reason,
            m.statement, m.project, m.conversation_id, c.title AS conversation_title
        FROM memory_status_history h
        JOIN memories m ON m.memory_id = h.memory_id
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.kind = 'decision' AND h.new_status = 'superseded'
          AND h.changed_at BETWEEN ? AND ?
        ORDER BY h.changed_at, h.history_id
        """,
        (since_iso, until_iso),
    ).fetchall()

    items = [
        f"{row['statement']} -- superseded {row['changed_at']} ({row['project'] or 'no project'})"
        for row in rows
    ]
    detail = [
        {
            "memory_id": int(row["memory_id"]),
            "statement": row["statement"],
            "project": row["project"],
            "old_status": row["old_status"],
            "new_status": row["new_status"],
            "changed_at": row["changed_at"],
            "reason": row["reason"],
            "conversation_id": int(row["conversation_id"]),
            "conversation_title": row["conversation_title"],
        }
        for row in rows
    ]
    return _section("superseded_decisions", "Decisions superseded", items, detail, limit)


def _new_open_tasks(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    rows = _memories_by_kind(
        conn, "task", since_iso, until_iso, extra_where="AND m.task_state = 'open'"
    )
    items = [f"{row['statement']} ({row['project'] or 'no project'})" for row in rows]
    detail = [
        {
            "memory_id": int(row["memory_id"]),
            "statement": row["statement"],
            "project": row["project"],
            "created_at": row["created_at"],
            "date_source": row["date_source"],
            "conversation_id": int(row["conversation_id"]),
            "conversation_title": row["conversation_title"],
        }
        for row in rows
    ]
    return _section("new_open_tasks", "New open tasks", items, detail, limit)


def _completed_tasks(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    """Tasks whose task_state transitioned to 'completed' during the window.

    Uses task_state_history.changed_at, not the memory's created_at -- mirroring how
    `_superseded_decisions` uses memory_status_history.changed_at above. A task that was
    *born* completed by the extraction heuristic (task_state='completed' with no
    task_state_history row -- see memory/store.py:set_task_state) is deliberately NOT
    reported here: it was never actually completed *during* this window, the extractor just
    guessed its state at extraction time. Only a real set_task_state() transition counts.
    """
    rows = conn.execute(
        """
        SELECT
            h.history_id, h.memory_id, h.old_state, h.new_state, h.changed_at, h.reason,
            m.statement, m.project, m.conversation_id, c.title AS conversation_title
        FROM task_state_history h
        JOIN memories m ON m.memory_id = h.memory_id
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.kind = 'task' AND h.new_state = 'completed'
          AND h.changed_at BETWEEN ? AND ?
        ORDER BY h.changed_at, h.history_id
        """,
        (since_iso, until_iso),
    ).fetchall()

    items = [
        f"{row['statement']} -- completed {row['changed_at']} ({row['project'] or 'no project'})"
        for row in rows
    ]
    detail = [
        {
            "memory_id": int(row["memory_id"]),
            "statement": row["statement"],
            "project": row["project"],
            "old_state": row["old_state"],
            "new_state": row["new_state"],
            "changed_at": row["changed_at"],
            "reason": row["reason"],
            "conversation_id": int(row["conversation_id"]),
            "conversation_title": row["conversation_title"],
        }
        for row in rows
    ]
    return _section("completed_tasks", "Tasks completed", items, detail, limit)


def _new_preferences(
    conn: sqlite3.Connection, since_iso: str, until_iso: str, limit: int
) -> DigestSection | None:
    rows = conn.execute(
        """
        SELECT pref_id, note, weight, source, created_at
        FROM learned_preferences
        WHERE created_at BETWEEN ? AND ?
        ORDER BY created_at, pref_id
        """,
        (since_iso, until_iso),
    ).fetchall()

    items = [row["note"] for row in rows]
    detail = [
        {
            "pref_id": int(row["pref_id"]),
            "note": row["note"],
            "weight": float(row["weight"]),
            "source": row["source"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return _section("new_preferences", "Preferences learned", items, detail, limit)


def _capture_problems(conn: sqlite3.Connection, limit: int) -> DigestSection | None:
    """Current capture/indexing problems.

    Unlike the other sections this is not windowed: a stuck index or an unindexed
    conversation is an ongoing state, not something that "happened" in a time range.
    """
    inventory = list_captures(conn, only_problems=True, limit=max(limit, 1) * 1000)

    items: list[str] = []
    detail: list[dict[str, object]] = []
    for item in inventory.items:
        items.append(f'"{item.title}": {", ".join(item.warnings)}')
        detail.append(
            {
                "conversation_id": item.conversation_id,
                "title": item.title,
                "warnings": list(item.warnings),
            }
        )
    if inventory.stale_index:
        items.append("the vector index is stale and needs a rebuild")
        detail.append({"stale_index": True})

    return _section("capture_problems", "Capture/indexing problems", items, detail, limit)


# A table where at least half the rows lack `created_at` is treated as having
# unreliable windowing: an empty section there is not evidence that "nothing
# happened", it is evidence that the timestamps needed to place rows in a
# window are missing.
_UNRELIABLE_TIMESTAMP_RATIO = 0.5


# `section.title` (e.g. "New projects", "Conversations captured") is always plural -- it
# reads fine as a section header ("New projects (1):") where the count sits alongside it,
# but the headline inlines it as "1 new projects", which is wrong at count 1. This maps each
# section key to (singular, plural) headline phrasing so the headline can pick the right one
# instead of reusing the (always-plural) section title.
_HEADLINE_NOUNS: dict[str, tuple[str, str]] = {
    "new_captures": ("conversation captured", "conversations captured"),
    "new_projects": ("new project", "new projects"),
    "new_decisions": ("new decision", "new decisions"),
    "superseded_decisions": ("decision superseded", "decisions superseded"),
    "new_open_tasks": ("new open task", "new open tasks"),
    "completed_tasks": ("task completed", "tasks completed"),
    "new_preferences": ("preference learned", "preferences learned"),
    "capture_problems": ("capture/indexing problem", "capture/indexing problems"),
}


def _headline_noun(section: DigestSection) -> str:
    singular, plural = _HEADLINE_NOUNS[section.key]
    return singular if section.count == 1 else plural


def _headline(
    sections: tuple[DigestSection, ...],
    window_label: str,
    *,
    unreliable_window: bool,
) -> str:
    """The one-line, count-derived summary.

    Provenance/reliability caveats are deliberately NOT embedded here -- they are stated
    once, separately, in `Digest.caveat`. An unreliable window still must never read as
    "nothing changed": it reads as "cannot tell", which is a different (truthful) claim.
    """
    if not sections:
        if unreliable_window:
            return f"Cannot tell what happened in the {window_label}."
        return f"Nothing changed in the {window_label}."
    parts = [f"{section.count} {_headline_noun(section)}" for section in sections]
    prefix = f"In the {window_label}"
    if unreliable_window:
        prefix += " (may be incomplete)"
    return f"{prefix}: " + "; ".join(parts) + "."


def _mark_mixed_provenance(sections: tuple[DigestSection, ...]) -> tuple[DigestSection, ...]:
    """Append a per-row "[dated by capture time]" note to `new_captures`, but only if this
    digest's dated rows actually mix "created" and "captured" provenance.

    The digest-level caveat already states once, beneath the headline, that some records are
    dated by capture time rather than creation time. Repeating that on every row when it is
    true of *every* row is the same noise stated twice; it only carries information when rows
    genuinely differ in provenance, i.e. a reader could otherwise mistake a capture-time date
    on one row for a creation-time date because a neighboring row's date happens to be a real
    creation date.
    """
    known_sources = {
        detail["date_source"]
        for section in sections
        for detail in section.detail
        if detail.get("date_source") in ("created", "captured")
    }
    if len(known_sources) <= 1:
        return sections

    marked: list[DigestSection] = []
    for section in sections:
        if section.key != "new_captures":
            marked.append(section)
            continue
        new_items = tuple(
            item + (" [dated by capture time]" if detail["date_source"] == "captured" else "")
            for item, detail in zip(section.items, section.detail, strict=True)
        )
        marked.append(replace(section, items=new_items))
    return tuple(marked)


def build_digest(
    conn: sqlite3.Connection,
    *,
    since: datetime,
    until: datetime | None = None,
    limit_per_section: int = 5,
) -> Digest:
    """Build a deterministic digest of memory-system changes between `since` and `until`.

    `until` defaults to now (in the same tzinfo-awareness as `since`). All window
    comparisons are done as string comparisons against ISO timestamps, matching how the
    rest of the codebase filters on `created_at` (see tasks.query.list_tasks), so `since`
    and `until` must use the same naive/aware convention as the timestamps stored in the
    database. Section order and item order are both deterministic (chronological, tied
    on id) so two calls over the same data produce identical output.
    """
    if until is None:
        until = datetime.now(since.tzinfo)

    since_iso = since.isoformat()
    until_iso = until.isoformat()

    builders = (
        _new_captures(conn, since_iso, until_iso, limit_per_section),
        _new_projects(conn, since_iso, until_iso, limit_per_section),
        _new_decisions(conn, since_iso, until_iso, limit_per_section),
        _superseded_decisions(conn, since_iso, until_iso, limit_per_section),
        _new_open_tasks(conn, since_iso, until_iso, limit_per_section),
        _completed_tasks(conn, since_iso, until_iso, limit_per_section),
        _new_preferences(conn, since_iso, until_iso, limit_per_section),
        _capture_problems(conn, limit_per_section),
    )
    sections = tuple(section for section in builders if section is not None)
    sections = _mark_mixed_provenance(sections)

    total_conversations = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
    total_memories = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # `missing_created` counts rows lacking a real creation date; `missing_effective` counts
    # rows where even the capture-time fallback can't resolve one. The gap between the two is
    # how many rows the capture-time fallback rescued -- those are dated, just not by creation
    # time, so they make the window's caveat milder rather than "unreliable".
    missing_created = count_missing_timestamps(conn)
    missing_effective = count_missing_effective_timestamps(conn)
    unreliable_window = (
        total_conversations > 0
        and missing_effective["conversations"] / total_conversations >= _UNRELIABLE_TIMESTAMP_RATIO
    ) or (
        total_memories > 0
        and missing_effective["memories"] / total_memories >= _UNRELIABLE_TIMESTAMP_RATIO
    )
    missing_total = missing_effective["conversations"] + missing_effective["memories"]
    captured_fallback_total = (
        missing_created["conversations"] - missing_effective["conversations"]
    ) + (missing_created["memories"] - missing_effective["memories"])
    if unreliable_window:
        caveat = (
            f"{_pluralize(missing_total, 'record')} (conversations + memories) have no "
            "creation date or capture date, so windowed sections may be missing entries "
            "that actually belong in this window."
        )
    elif captured_fallback_total > 0:
        caveat = (
            f"{_pluralize(captured_fallback_total, 'record')} here are dated by capture "
            "time (when convsearch first saw them), not creation time."
        )
    else:
        caveat = ""

    window_label = _window_label(since, until)
    return Digest(
        since=since_iso,
        until=until_iso,
        window_label=window_label,
        sections=sections,
        is_empty=len(sections) == 0 and not unreliable_window,
        headline=_headline(sections, window_label, unreliable_window=unreliable_window),
        unreliable_window=unreliable_window,
        caveat=caveat,
    )
