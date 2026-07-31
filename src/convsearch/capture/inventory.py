"""Inventory of what the browser extension has actually captured.

`POST /capture` and the export importer both write into the same `conversations` table,
distinguished only by which `imports` row owns them (see `capture.state`). This module
answers the question a user asks when the popup looks wrong: which conversations came from
live capture versus an export, and how far each one got through the pipeline (segmented,
indexed, memories extracted). It is read-only and does not touch the vector index or run
any model.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from convsearch.capture.state import CAPTURE_SOURCE_HASH, read_index_stale
from convsearch.utils import effective_timestamp_sql

SOURCE_LIVE = "live-capture"
SOURCE_IMPORT = "export-import"

# ChatGPT conversation ids are UUIDs. Anything else (a synthetic id the extension made up,
# a placeholder, an empty string) is not a real conversation and must not become a URL.
_SOURCE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


@dataclass(frozen=True)
class CaptureItem:
    conversation_id: str
    source_conversation_id: str | None
    title: str
    captured_at: str | None
    updated_at: str | None
    message_count: int
    source: str
    indexed: bool
    segmented: bool
    memories_extracted: bool
    passage_count: int
    memory_count: int
    source_url: str | None
    warnings: tuple[str, ...]
    # 'created' (a real creation date), 'captured' (no creation date -- captured_at above is
    # conversations.updated_at, i.e. capture wall-clock), 'imported' (neither of the above
    # exists -- captured_at falls back to imports.imported_at, i.e. when the batch containing
    # this conversation was imported, a coarser and less precise time than per-conversation
    # capture time), or 'unknown' (no timestamp at all). See convsearch.utils for the
    # created/captured half of this chain; the imported fallback is specific to captures
    # because only this surface joins against `imports`.
    date_source: str = "unknown"


@dataclass(frozen=True)
class CaptureInventory:
    items: tuple[CaptureItem, ...]
    total: int
    live_captured: int
    imported: int
    not_indexed: int
    not_segmented: int
    stale_index: bool


def _source_url(source_conversation_id: str | None) -> str | None:
    """Reconstruct the chatgpt.com URL, never guessing at ids that are not real UUIDs."""
    if source_conversation_id is None:
        return None
    if not _SOURCE_ID_RE.match(source_conversation_id):
        return None
    return f"https://chatgpt.com/c/{source_conversation_id}"


def _warnings(
    *,
    message_count: int,
    indexed: bool,
    segmented: bool,
    memories_extracted: bool,
    passage_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if message_count == 0:
        warnings.append("no messages")
    if passage_count == 0 and message_count > 0:
        warnings.append("no passages")
    if passage_count > 0 and not indexed:
        warnings.append("not indexed")
    if message_count > 0 and not segmented:
        warnings.append("not segmented")
    if message_count > 0 and not memories_extracted:
        warnings.append("no memories extracted")
    return tuple(warnings)


def list_captures(
    conn: sqlite3.Connection,
    *,
    source: str = "all",
    limit: int = 50,
    only_problems: bool = False,
) -> CaptureInventory:
    """Build the inventory of captured conversations and how far each was processed.

    `source` filters to "all", "live" (captured by the extension), or "import" (from an
    export ZIP). `only_problems` keeps only conversations with at least one warning. Counts
    are aggregated with one `GROUP BY` query per table rather than one query per
    conversation.
    """
    if source not in ("all", "live", "import"):
        raise ValueError(f"unknown source filter: {source!r}")

    conversation_rows = conn.execute(
        f"""
        SELECT
            c.conversation_id AS conversation_id,
            c.source_conversation_id AS source_conversation_id,
            c.title AS title,
            c.updated_at AS updated_at,
            i.source_hash AS source_hash,
            COALESCE({effective_timestamp_sql("c")}, i.imported_at) AS captured_at,
            (CASE
                WHEN c.created_at IS NOT NULL THEN 'created'
                WHEN c.updated_at IS NOT NULL THEN 'captured'
                WHEN i.imported_at IS NOT NULL THEN 'imported'
                ELSE 'unknown'
            END) AS date_source
        FROM conversations c
        JOIN imports i ON i.import_id = c.import_id
        """
    ).fetchall()

    message_counts = _grouped_counts(
        conn, "SELECT conversation_id, COUNT(*) AS n FROM messages GROUP BY conversation_id"
    )
    passage_counts = _grouped_counts(
        conn, "SELECT conversation_id, COUNT(*) AS n FROM passages GROUP BY conversation_id"
    )
    segment_counts = _grouped_counts(
        conn, "SELECT conversation_id, COUNT(*) AS n FROM segments GROUP BY conversation_id"
    )
    memory_counts = _grouped_counts(
        conn, "SELECT conversation_id, COUNT(*) AS n FROM memories GROUP BY conversation_id"
    )
    indexed_conversations = _indexed_conversation_ids(conn)

    items: list[CaptureItem] = []
    live_captured = 0
    imported = 0
    for row in conversation_rows:
        conversation_id = int(row["conversation_id"])
        item_source = SOURCE_LIVE if row["source_hash"] == CAPTURE_SOURCE_HASH else SOURCE_IMPORT
        if item_source == SOURCE_LIVE:
            live_captured += 1
        else:
            imported += 1
        if source == "live" and item_source != SOURCE_LIVE:
            continue
        if source == "import" and item_source != SOURCE_IMPORT:
            continue

        message_count = message_counts.get(conversation_id, 0)
        passage_count = passage_counts.get(conversation_id, 0)
        segmented = conversation_id in segment_counts
        indexed = conversation_id in indexed_conversations
        memory_count = memory_counts.get(conversation_id, 0)
        memories_extracted = memory_count > 0

        warnings = _warnings(
            message_count=message_count,
            indexed=indexed,
            segmented=segmented,
            memories_extracted=memories_extracted,
            passage_count=passage_count,
        )
        if only_problems and not warnings:
            continue

        source_conversation_id = row["source_conversation_id"]
        items.append(
            CaptureItem(
                conversation_id=str(conversation_id),
                source_conversation_id=source_conversation_id,
                title=str(row["title"]),
                captured_at=row["captured_at"],
                updated_at=row["updated_at"],
                message_count=message_count,
                source=item_source,
                indexed=indexed,
                segmented=segmented,
                memories_extracted=memories_extracted,
                passage_count=passage_count,
                memory_count=memory_count,
                source_url=_source_url(source_conversation_id),
                warnings=warnings,
                date_source=str(row["date_source"]),
            )
        )

    # `captured_at` is already the effective (fallback-resolved) timestamp, so ordering by it
    # directly keeps sort order consistent with what is displayed.
    items.sort(
        key=lambda item: (item.captured_at or "", int(item.conversation_id)),
        reverse=True,
    )
    limited = tuple(items[:limit])

    with_index_stale = read_index_stale(conn)
    return CaptureInventory(
        items=limited,
        total=len(items),
        live_captured=live_captured,
        imported=imported,
        not_indexed=sum(1 for item in items if item.passage_count > 0 and not item.indexed),
        not_segmented=sum(1 for item in items if item.message_count > 0 and not item.segmented),
        stale_index=with_index_stale,
    )


def _grouped_counts(conn: sqlite3.Connection, query: str) -> dict[int, int]:
    return {int(row["conversation_id"]): int(row["n"]) for row in conn.execute(query).fetchall()}


def _indexed_conversation_ids(conn: sqlite3.Connection) -> set[int]:
    """Conversation ids with at least one passage that has a current embedding record."""
    rows = conn.execute(
        """
        SELECT DISTINCT p.conversation_id AS conversation_id
        FROM passages p
        JOIN embedding_records e ON e.passage_id = p.passage_id
        WHERE e.content_hash = p.content_hash
        """
    ).fetchall()
    return {int(row["conversation_id"]) for row in rows}
