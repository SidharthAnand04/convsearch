from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

# Tables with a `created_at` column whose reliability the timeline/digest depend on.
_TIMESTAMPED_TABLES: tuple[str, ...] = ("conversations", "messages", "memories")


def count_missing_timestamps(conn: sqlite3.Connection) -> dict[str, int]:
    """Count rows with a NULL `created_at` per table.

    Lets a caller distinguish "nothing happened in this window" from "N records have no
    timestamp so the window is unreliable" without silently treating unknown as absent.
    """
    counts: dict[str, int] = {}
    for table in _TIMESTAMPED_TABLES:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table} WHERE created_at IS NULL")
        counts[table] = int(row.fetchone()["count"])
    return counts


# ---------------------------------------------------------------------------
# Effective timestamp: fallback chain from creation time to capture time
# ---------------------------------------------------------------------------
#
# Live-captured conversations (see extension/capture.js) carry `created_at = NULL`
# because ChatGPT's DOM exposes no per-message timestamp -- there is no creation date to
# derive one from. `conversations.updated_at` is populated for every row, but it means
# something different: it is capture wall-clock time, i.e. when convsearch first saw the
# conversation, not when the conversation itself was created. A conversation from months
# ago captured yesterday must never be presented as "created yesterday".
#
# The helpers below build a fallback chain (creation time first, capture time only as a
# last resort) alongside a matching "which source did this come from" expression, so every
# caller can label a row correctly instead of silently coalescing away the distinction.
# `date_source` is one of "created" (a real creation date, possibly inherited from the
# parent conversation), "captured" (no creation date exists -- this is capture time), or
# "unknown" (neither is available).


def effective_timestamp_sql(alias: str) -> str:
    """SQL expression: best-available timestamp for a `conversations`-shaped row.

    `alias` is the table alias used in the query, e.g. `effective_timestamp_sql("c")` for
    `... FROM conversations c ...`. Pair with `effective_timestamp_source_sql` so callers
    can tell whether the value is a real creation date or a capture-time fallback.
    """
    return f"COALESCE({alias}.created_at, {alias}.updated_at)"


def effective_timestamp_source_sql(alias: str) -> str:
    """Companion to `effective_timestamp_sql`: yields 'created' | 'captured' | 'unknown'."""
    return (
        f"(CASE WHEN {alias}.created_at IS NOT NULL THEN 'created' "
        f"WHEN {alias}.updated_at IS NOT NULL THEN 'captured' ELSE 'unknown' END)"
    )


def memory_effective_timestamp_sql(memory_alias: str, conversation_alias: str) -> str:
    """SQL expression: best-available timestamp for a `memories`-shaped row.

    Fallback chain: the memory's own `created_at`, then its conversation's `created_at`
    (still a genuine creation date -- just the conversation's, not the memory's own), then
    the conversation's `updated_at` (capture time, last resort). The query must join
    `conversations` under `conversation_alias` on `memory_alias.conversation_id`.
    """
    return (
        f"COALESCE({memory_alias}.created_at, {conversation_alias}.created_at, "
        f"{conversation_alias}.updated_at)"
    )


def memory_effective_timestamp_source_sql(memory_alias: str, conversation_alias: str) -> str:
    """Companion to `memory_effective_timestamp_sql`: 'created' | 'captured' | 'unknown'."""
    return (
        f"(CASE WHEN {memory_alias}.created_at IS NOT NULL THEN 'created' "
        f"WHEN {conversation_alias}.created_at IS NOT NULL THEN 'created' "
        f"WHEN {conversation_alias}.updated_at IS NOT NULL THEN 'captured' "
        f"ELSE 'unknown' END)"
    )


def count_missing_effective_timestamps(conn: sqlite3.Connection) -> dict[str, int]:
    """Count rows where even the capture-time fallback can't resolve a date.

    Unlike `count_missing_timestamps`, a conversation with NULL `created_at` but a real
    `updated_at` does NOT count here -- it has *a* date, just not a creation date. Callers
    use this (rather than `count_missing_timestamps`) to decide whether a time window is
    actually unreliable now that capture time is available as a fallback.
    """
    conversations = int(
        conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE created_at IS NULL AND updated_at IS NULL"
        ).fetchone()[0]
    )
    memories = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM memories m
            JOIN conversations c ON c.conversation_id = m.conversation_id
            WHERE m.created_at IS NULL AND c.created_at IS NULL AND c.updated_at IS NULL
            """
        ).fetchone()[0]
    )
    return {"conversations": conversations, "memories": memories}


def format_dated(value: str | None, date_source: str = "unknown") -> str:
    """Render a date with its provenance so a capture date is never read as a creation date.

    'created' renders bare ("29 Jul"); 'captured' and 'imported' are annotated ("29 Jul
    (captured)" / "29 Jul (imported)") since those dates are when convsearch first saw or
    imported the item, not when it was created; a missing value renders as "(unknown date)"
    regardless of source. Shared by the CLI and the digest so every rendered date in the
    product uses the same compact, provenance-aware presentation -- never a raw ISO string.
    """
    if not value:
        return "(unknown date)"
    try:
        day = datetime.fromisoformat(value[:19]).strftime("%d %b")
    except ValueError:
        day = value[:10]
    if date_source in ("captured", "imported"):
        return f"{day} ({date_source})"
    return day


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Render "1 memory" / "2 memories" -- never the ungrammatical "N memory(s)".

    Pass `plural` explicitly for irregular nouns; naive "+s" would yield "memorys".
    """
    if count == 1:
        return f"{count} {singular}"
    return f"{count} {plural if plural is not None else singular + 's'}"
