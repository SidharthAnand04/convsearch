from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from convsearch.capture.state import set_index_stale
from convsearch.config.settings import Settings, faiss_index_path
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.indexes.lexical import refresh_fts, sync_fts
from convsearch.indexes.locking import MUTATE_LOCK, index_lock
from convsearch.indexes.vector import (
    append_vector_index,
    read_vector_map,
    vector_index_size,
    write_vector_index,
)
from convsearch.segmentation.build import rebuild_segments, rebuild_segments_for_conversations
from convsearch.storage.database import connection

# An index pass may sit behind another process's pass (a CLI `convsearch index` on a big
# corpus). Waiting is correct; waiting forever is not — after this the caller gets an
# IndexLockTimeout naming the other process instead of hanging.
MUTATE_LOCK_TIMEOUT = 900.0


@dataclass(frozen=True)
class IndexUpdate:
    """Outcome of `update_indexes`. `encoded` is how many passages were embedded.

    `pending` is how many passages still lack a current embedding when the pass finished —
    normally 0, but non-zero when a capture committed while this pass was encoding. The
    caller must schedule another pass, or those passages stay unsearchable.
    """

    mode: str  # "full" | "incremental" | "noop"
    encoded: int
    total: int
    reason: str | None = None
    pending: int = 0


def _pending_passage_count(conn: sqlite3.Connection, model_id: str) -> int:
    """Passages with no current embedding for `model_id`: never embedded, or edited since."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM passages p
        LEFT JOIN embedding_records e ON e.passage_id = p.passage_id
        WHERE e.passage_id IS NULL
           OR e.content_hash <> p.content_hash
           OR e.model_id <> ?
        """,
        (model_id,),
    ).fetchone()
    return int(row["total"]) if row is not None else 0


def _sync_stale_flag(conn: sqlite3.Connection, workspace: Path, model_id: str) -> int:
    """Set `stale_index` from what is actually embedded, and report what is still missing.

    Writing `False` unconditionally at the end of a pass is a lost update: a capture that
    commits while the pass is encoding sets the flag to True, and the pass's later False
    overwrites it even though the new passages are not embedded. `/health` then reports a
    current index over data that is not indexed, and if the process dies before the next
    capture arrives nothing ever schedules the follow-up pass — those conversations become
    permanently unsearchable. Deriving the flag from `embedding_records` cannot lose an
    update, because the count is taken inside the same write transaction that stores it.
    """
    pending = _pending_passage_count(conn, model_id)
    # Backstop for the one loss the row-level check cannot see. If another process overwrote
    # this pass's index file, its `embedding_records` rows still say "embedded" while the
    # published map never lists them, so those passages are unfindable and nothing above
    # notices. The cross-process `mutate` lock is what prevents that; this is what detects it
    # if the lock is ever unavailable, so the workspace stays marked stale and gets retried.
    mapped_count = len(read_vector_map(workspace))
    row = conn.execute("SELECT COUNT(*) AS total FROM embedding_records").fetchone()
    record_count = int(row["total"]) if row is not None else 0
    if record_count != mapped_count:
        pending = max(pending, abs(record_count - mapped_count))
    set_index_stale(conn, pending > 0)
    return pending


def build_indexes(workspace: Path, settings: Settings, provider: EmbeddingProvider) -> int:
    with index_lock(workspace, MUTATE_LOCK, timeout=MUTATE_LOCK_TIMEOUT):
        return _build_indexes(workspace, settings, provider)


def _build_indexes(workspace: Path, settings: Settings, provider: EmbeddingProvider) -> int:
    if settings.segmentation.enabled:
        rebuild_segments(workspace, settings, provider=provider)
    with connection(workspace) as conn:
        with conn:
            refresh_fts(conn)
        rows = conn.execute(
            """
            SELECT passage_id, text, content_hash
            FROM passages
            ORDER BY passage_id
            """
        ).fetchall()
        passage_ids = [int(row["passage_id"]) for row in rows]
        texts = [str(row["text"]) for row in rows]
        content_hashes = {int(row["passage_id"]): str(row["content_hash"]) for row in rows}
    vectors = (
        provider.encode_documents(texts, batch_size=settings.embedding_batch_size)
        if texts
        else np.zeros((0, 0), dtype=np.float32)
    )
    backend, dimension = write_vector_index(workspace, vectors, passage_ids)
    now = datetime.now(UTC).isoformat()
    with connection(workspace) as conn, conn:
        conn.execute("DELETE FROM embedding_records")
        conn.executemany(
            """
            INSERT INTO embedding_records(
                passage_id, vector_id, model_id, embedding_dimension, content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    passage_id,
                    vector_id,
                    provider.model_id,
                    dimension,
                    content_hashes[passage_id],
                )
                for vector_id, passage_id in enumerate(passage_ids)
            ],
        )
        metadata = {
            "model_id": provider.model_id,
            "embedding_dimension": dimension,
            "backend": backend,
            "index_type": "IndexFlatIP" if backend == "faiss" else "numpy-exact-ip",
            "built_at": now,
            "passage_count": len(passage_ids),
        }
        for key, value in metadata.items():
            conn.execute(
                """
                INSERT INTO index_metadata(key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (key, json.dumps(value)),
            )
        # Every rebuild refreshes the live-capture staleness marker, so the CLI, the server's
        # /reindex route and the evaluation runner cannot disagree about it. Derived rather
        # than forced to False: a capture that landed while this rebuild was encoding must
        # leave the workspace stale, not falsely current.
        _sync_stale_flag(conn, workspace, provider.model_id)
    return len(passage_ids)


def _read_metadata(conn: sqlite3.Connection) -> dict[str, Any]:
    values: dict[str, Any] = {}
    try:
        rows = conn.execute("SELECT key, value FROM index_metadata").fetchall()
    except sqlite3.DatabaseError:
        return values
    for row in rows:
        try:
            values[str(row["key"])] = json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            continue
    return values


def _full_rebuild_reason(
    workspace: Path,
    provider: EmbeddingProvider,
    mapped: list[int],
    current: dict[int, str],
    embedded: dict[int, str],
    metadata: dict[str, Any],
    max_vector_id: int | None = None,
) -> str | None:
    """Why an append would be unsafe, or None when incremental is provably fine.

    Appending assumes every vector already in the index still corresponds to the passage it
    was built from. Each check below is a way that assumption breaks.
    """
    index_exists = (
        faiss_index_path(workspace).exists()
        or faiss_index_path(workspace).with_suffix(".npy").exists()
    )
    if not index_exists or not mapped:
        return "no existing index"
    on_disk = vector_index_size(workspace)
    if on_disk is None:
        # Truncated or corrupt: the file exists but cannot be opened. A rebuild is the only
        # way back, and doing it here is what makes a corrupt index self-healing instead of
        # an error on every search until someone presses a button.
        return "existing index file is unreadable"
    if on_disk != len(mapped):
        # The index and the map are two files replaced one after the other, so a crash
        # between them desyncs the pair. Left alone, position N of the map no longer names
        # vector N and every future search answers with the wrong passage.
        return f"index holds {on_disk} vectors but the map has {len(mapped)} ids"
    if metadata.get("model_id") != provider.model_id:
        # Inner product across two different embedding spaces is meaningless.
        return "embedding model changed"
    if len(mapped) != len(set(mapped)):
        return "vector map contains duplicate passage ids"
    missing = [passage_id for passage_id in mapped if passage_id not in current]
    if missing:
        # Capture upserts by deleting a conversation's messages; passages cascade. Left in
        # place, those vectors return hits for text that no longer exists.
        return f"{len(missing)} indexed passage(s) no longer exist"
    changed = [
        passage_id
        for passage_id in mapped
        if embedded.get(passage_id) is not None and embedded[passage_id] != current[passage_id]
    ]
    if changed:
        return f"{len(changed)} indexed passage(s) were edited"
    if max_vector_id is not None and max_vector_id >= len(mapped):
        # embedding_records is a THIRD piece of state alongside the index file and the map, so
        # it can drift independently of both. An incremental append starts numbering at
        # len(mapped); if a record already holds that vector_id, the insert violates
        # UNIQUE(vector_id) and the pass dies with "IntegrityError: constraint failed" — and
        # because nothing repairs it, every later pass fails the same way and indexing stops
        # for good. Observed in the wild after two servers indexed one workspace at once.
        return (
            f"embedding_records holds vector_id {max_vector_id} but the map has {len(mapped)} ids"
        )
    return None


def update_indexes(workspace: Path, settings: Settings, provider: EmbeddingProvider) -> IndexUpdate:
    """Bring the index up to date, encoding only what is new when that is safe.

    Falls back to a full rebuild whenever appending could produce wrong results. A slow
    rebuild is always preferable to a phantom search hit.

    Holds the workspace's cross-process `mutate` lock for the whole pass. Without it a
    concurrent `convsearch index` (or a second `serve`) could read the same map, encode, and
    write last — silently dropping this pass's vectors while its `embedding_records` rows
    stay committed, so those passages would look indexed and never be found.
    """
    with index_lock(workspace, MUTATE_LOCK, timeout=MUTATE_LOCK_TIMEOUT):
        return _update_indexes(workspace, settings, provider)


def _update_indexes(
    workspace: Path, settings: Settings, provider: EmbeddingProvider
) -> IndexUpdate:
    with connection(workspace) as conn:
        current = {
            int(row["passage_id"]): str(row["content_hash"])
            for row in conn.execute("SELECT passage_id, content_hash FROM passages")
        }
        embedded = {
            int(row["passage_id"]): str(row["content_hash"])
            for row in conn.execute("SELECT passage_id, content_hash FROM embedding_records")
        }
        metadata = _read_metadata(conn)
        row = conn.execute("SELECT MAX(vector_id) AS top FROM embedding_records").fetchone()
        max_vector_id = None if row is None or row["top"] is None else int(row["top"])
    mapped = read_vector_map(workspace)

    reason = _full_rebuild_reason(
        workspace, provider, mapped, current, embedded, metadata, max_vector_id
    )
    if reason is not None:
        total = _build_indexes(workspace, settings, provider)
        with connection(workspace) as conn, conn:
            pending = _sync_stale_flag(conn, workspace, provider.model_id)
        return IndexUpdate(mode="full", encoded=total, total=total, reason=reason, pending=pending)

    indexed = set(mapped)
    new_ids = sorted(passage_id for passage_id in current if passage_id not in indexed)
    if not new_ids:
        with connection(workspace) as conn, conn:
            pending = _sync_stale_flag(conn, workspace, provider.model_id)
        return IndexUpdate(mode="noop", encoded=0, total=len(mapped), pending=pending)

    with connection(workspace) as conn:
        with conn:
            # New passages must become lexically searchable too, not just semantically.
            # Scoped to the new ids: a full refresh_fts here would make adding one
            # conversation cost O(entire archive) on every pass.
            sync_fts(conn, new_ids)
        placeholders = ",".join("?" * len(new_ids))
        rows = conn.execute(
            f"SELECT passage_id, text, conversation_id FROM passages "
            f"WHERE passage_id IN ({placeholders}) ORDER BY passage_id",
            new_ids,
        ).fetchall()
    texts = [str(row["text"]) for row in rows]
    passage_ids = [int(row["passage_id"]) for row in rows]
    if settings.segmentation.enabled:
        # Scoped to the conversations that gained new passages this pass - a full
        # rebuild_segments() here would make adding one conversation cost O(entire archive),
        # exactly the incremental-indexing regression this function exists to avoid.
        new_conversation_ids = sorted({int(row["conversation_id"]) for row in rows})
        rebuild_segments_for_conversations(
            workspace, settings, new_conversation_ids, provider=provider
        )
    vectors = provider.encode_documents(texts, batch_size=settings.embedding_batch_size)
    backend, dimension = append_vector_index(workspace, vectors, passage_ids)

    first_vector_id = len(mapped)
    with connection(workspace) as conn, conn:
        conn.executemany(
            """
            INSERT INTO embedding_records(
                passage_id, vector_id, model_id, embedding_dimension, content_hash
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(passage_id) DO UPDATE SET
                vector_id=excluded.vector_id,
                model_id=excluded.model_id,
                embedding_dimension=excluded.embedding_dimension,
                content_hash=excluded.content_hash
            """,
            [
                (
                    passage_id,
                    first_vector_id + offset,
                    provider.model_id,
                    dimension,
                    current[passage_id],
                )
                for offset, passage_id in enumerate(passage_ids)
            ],
        )
        total = first_vector_id + len(passage_ids)
        for key, value in {
            "backend": backend,
            "built_at": datetime.now(UTC).isoformat(),
            "passage_count": total,
        }.items():
            conn.execute(
                """
                INSERT INTO index_metadata(key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (key, json.dumps(value)),
            )
        pending = _sync_stale_flag(conn, workspace, provider.model_id)
    return IndexUpdate(mode="incremental", encoded=len(passage_ids), total=total, pending=pending)
