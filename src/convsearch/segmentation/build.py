from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from convsearch.config.settings import Settings
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.segmentation.base import SegmentationProvider
from convsearch.segmentation.hybrid import HybridSegmentationProvider
from convsearch.segmentation.models import SegmentableMessage
from convsearch.segmentation.rules import RuleBasedSegmentationProvider
from convsearch.segmentation.semantic import SemanticShiftSegmentationProvider
from convsearch.storage.database import connection


def make_segmentation_provider(
    settings: Settings, provider: EmbeddingProvider | None = None
) -> SegmentationProvider:
    strategy = settings.segmentation.strategy
    if strategy in ("semantic", "hybrid"):
        if provider is None:
            raise RuntimeError(
                f"Segmentation strategy '{strategy}' requires an embedding provider. "
                "Pass provider=<EmbeddingProvider> to rebuild_segments()."
            )
        if strategy == "semantic":
            return SemanticShiftSegmentationProvider(settings.segmentation, provider)
        return HybridSegmentationProvider(settings.segmentation, provider)
    return RuleBasedSegmentationProvider(settings.segmentation)


def rebuild_segments(
    workspace: Path, settings: Settings, *, provider: EmbeddingProvider | None = None
) -> int:
    """Full rebuild: re-segment every conversation in the workspace.

    O(entire corpus). Safe to call from `build_indexes` (already a full pass) but far too
    costly to call from an incremental indexing pass once the corpus is more than a handful
    of conversations - see `rebuild_segments_for_conversations` for the scoped variant used
    there.
    """
    segmentation_provider = make_segmentation_provider(settings, provider)
    with connection(workspace) as conn, conn:
        conn.execute("DELETE FROM segment_fts")
        conn.execute("DELETE FROM segments")
        total = _segment_conversations(conn, _conversation_ids(conn), segmentation_provider)
        _write_segment_state(conn, segmentation_provider.version)
        return total


def rebuild_segments_for_conversations(
    workspace: Path,
    settings: Settings,
    conversation_ids: list[int],
    *,
    provider: EmbeddingProvider | None = None,
) -> int:
    """Incremental rebuild: re-segment only the given conversations.

    Cost is O(len(conversation_ids)), not O(corpus) - safe to run on every incremental
    indexing pass alongside newly-embedded passages. Existing segments for conversations NOT
    in `conversation_ids` are left untouched.
    """
    if not conversation_ids:
        return 0
    segmentation_provider = make_segmentation_provider(settings, provider)
    with connection(workspace) as conn, conn:
        placeholders = ",".join("?" for _ in conversation_ids)
        stale_segment_ids = [
            int(row["segment_id"])
            for row in conn.execute(
                f"SELECT segment_id FROM segments WHERE conversation_id IN ({placeholders})",
                conversation_ids,
            )
        ]
        if stale_segment_ids:
            fts_placeholders = ",".join("?" for _ in stale_segment_ids)
            conn.execute(
                f"DELETE FROM segment_fts WHERE rowid IN ({fts_placeholders})",
                stale_segment_ids,
            )
            conn.execute(
                f"DELETE FROM segments WHERE conversation_id IN ({placeholders})",
                conversation_ids,
            )
        total = _segment_conversations(conn, conversation_ids, segmentation_provider)
        _write_segment_state(conn, segmentation_provider.version)
        return total


def _segment_conversations(
    conn: sqlite3.Connection,
    conversation_ids: list[int],
    segmentation_provider: SegmentationProvider,
) -> int:
    total = 0
    for conversation_id in conversation_ids:
        messages = _messages_for_conversation(conn, conversation_id)
        for segment in segmentation_provider.segment(messages):
            cursor = conn.execute(
                """
                INSERT INTO segments(
                    conversation_id, segment_order, start_message_id, end_message_id,
                    title, summary, boundary_confidence, segmentation_version,
                    content_hash, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.conversation_id,
                    segment.segment_order,
                    segment.start_message_id,
                    segment.end_message_id,
                    segment.title,
                    segment.summary,
                    segment.boundary_confidence,
                    segmentation_provider.version,
                    segment.content_hash,
                    json.dumps({"reasons": segment.reasons}),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Segment insert did not return a segment_id")
            segment_id = int(cursor.lastrowid)
            placeholders = ",".join("?" for _ in segment.message_ids)
            conn.execute(
                f"UPDATE passages SET segment_id = ? WHERE message_id IN ({placeholders})",
                (segment_id, *segment.message_ids),
            )
            _insert_segment_fts(conn, segment_id)
            total += 1
    return total


def _write_segment_state(conn: sqlite3.Connection, version: str) -> None:
    conn.execute(
        """
        INSERT INTO index_metadata(key, value, updated_at)
        VALUES ('segment_state', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """,
        (json.dumps({"state": "current", "version": version}),),
    )


def refresh_segment_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM segment_fts")
    for row in conn.execute("SELECT segment_id FROM segments ORDER BY segment_id"):
        _insert_segment_fts(conn, int(row["segment_id"]))


def _conversation_ids(conn: sqlite3.Connection) -> list[int]:
    return [
        int(row["conversation_id"])
        for row in conn.execute(
            "SELECT conversation_id FROM conversations ORDER BY conversation_id"
        )
    ]


def _messages_for_conversation(
    conn: sqlite3.Connection, conversation_id: int
) -> list[SegmentableMessage]:
    rows = conn.execute(
        """
        SELECT message_id, conversation_id, source_order, role, text, created_at, is_primary_path
        FROM messages
        WHERE conversation_id = ?
        ORDER BY source_order, message_id
        """,
        (conversation_id,),
    ).fetchall()
    return [
        SegmentableMessage(
            message_id=int(row["message_id"]),
            conversation_id=int(row["conversation_id"]),
            source_order=int(row["source_order"]),
            role=str(row["role"]),
            text=str(row["text"]),
            created_at=row["created_at"],
            is_primary_path=bool(row["is_primary_path"]),
        )
        for row in rows
    ]


def _insert_segment_fts(conn: sqlite3.Connection, segment_id: int) -> None:
    row = conn.execute(
        """
        SELECT s.title, c.title AS conversation_title,
               group_concat(m.role || ': ' || m.text, char(10)) AS text
        FROM segments s
        JOIN conversations c ON c.conversation_id = s.conversation_id
        JOIN messages m
          ON m.conversation_id = s.conversation_id
         AND m.source_order BETWEEN
             (SELECT source_order FROM messages WHERE message_id = s.start_message_id)
             AND (SELECT source_order FROM messages WHERE message_id = s.end_message_id)
         AND m.is_primary_path = (
             SELECT is_primary_path FROM messages WHERE message_id = s.start_message_id
         )
        WHERE s.segment_id = ?
        GROUP BY s.segment_id
        """,
        (segment_id,),
    ).fetchone()
    if row is None:
        return
    conn.execute(
        "INSERT INTO segment_fts(rowid, text, title, conversation_title) VALUES (?, ?, ?, ?)",
        (segment_id, row["text"] or "", row["title"], row["conversation_title"]),
    )
