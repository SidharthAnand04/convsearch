from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from convsearch.memory.extract import extract_from_message
from convsearch.memory.models import MEMORY_STATUSES, TASK_STATES, ExtractedMemory
from convsearch.utils import memory_effective_timestamp_sql, stable_hash


@dataclass(frozen=True)
class MemoryExtractionSummary:
    extracted: int
    inserted: int
    superseded: int
    contested: int
    entities: int
    # Additive: how many candidates the quality filter (quality.is_usable_statement)
    # dropped before insertion, and why. Defaults keep existing call sites/tests working
    # without passing these. A silent filter is hard to trust or tune, so a caller (e.g.
    # the CLI) can report "extracted N, filtered M" from these instead of debug logs.
    rejected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryPurgeSummary:
    """Result of `clear_memories`: how many memories were removed vs. kept because the

    user had curated them (see `_CURATION_PREDICATE` for the full list of signals).
    """

    deleted: int
    preserved: int


def extract_and_store_memories(
    conn: sqlite3.Connection,
    *,
    extraction_version: str = "rules-v2",
) -> MemoryExtractionSummary:
    """Extract memories from all primary-path messages and store them."""
    # Fallback chain: the message's own created_at, then its conversation's created_at
    # (still a real creation date), then the conversation's updated_at (capture time --
    # when convsearch first saw it -- as a last resort for live-captured conversations that
    # carry no per-message timestamp at all).
    ts = memory_effective_timestamp_sql("m", "c")
    rows = conn.execute(
        f"""
        SELECT m.message_id, m.conversation_id, m.text,
               {ts} AS created_at,
               c.title AS conversation_title
        FROM messages m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.is_primary_path = 1 AND m.text != ''
        ORDER BY m.conversation_id, m.source_order
        """
    ).fetchall()

    total_extracted = 0
    total_inserted = 0
    total_entities = 0
    reject_counts: dict[str, int] = {}

    for row in rows:
        message_id: int = row["message_id"]
        conversation_id: int = row["conversation_id"]
        text: str = row["text"]
        created_at: str | None = row["created_at"]
        conversation_title: str | None = row["conversation_title"]

        memories = extract_from_message(
            text,
            conversation_id=conversation_id,
            message_id=message_id,
            created_at=created_at,
            default_project=conversation_title,
            reject_counts=reject_counts,
        )
        total_extracted += len(memories)

        for mem in memories:
            inserted, entities = _insert_memory(conn, mem, extraction_version)
            total_inserted += inserted
            total_entities += entities

    superseded_count, contested_count = reconcile(conn)
    conn.commit()

    return MemoryExtractionSummary(
        extracted=total_extracted,
        inserted=total_inserted,
        superseded=superseded_count,
        contested=contested_count,
        entities=total_entities,
        rejected=sum(reject_counts.values()),
        rejected_by_reason=reject_counts,
    )


def _insert_memory(
    conn: sqlite3.Connection,
    mem: ExtractedMemory,
    extraction_version: str,
) -> tuple[int, int]:
    """Insert one extracted memory, deduped by content_hash. Returns (inserted, entities).

    ``inserted`` is 1 for a brand-new memory, 0 if an identical one already existed (never
    overwritten). ``entities`` counts entity mentions newly recorded. This is the single
    insertion path shared by the rules extractor (``extract_and_store_memories``) and the
    opt-in LLM path (``store_extracted_memories``), so both get identical dedup, evidence, and
    entity behaviour. It deliberately does NOT call ``reconcile()`` or ``commit()`` -- those
    run once per batch in the caller, so supersession sees every new memory at once.
    """
    content_hash = stable_hash(mem.kind, mem.subject_key, mem.statement, mem.message_id)

    conn.execute(
        """
        INSERT OR IGNORE INTO memories
          (kind, subject_key, statement, status, confidence, project,
           task_state, conversation_id, message_id, created_at,
           extraction_version, content_hash, metadata_json)
        VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mem.kind,
            mem.subject_key,
            mem.statement,
            mem.confidence,
            mem.project,
            mem.task_state,
            mem.conversation_id,
            mem.message_id,
            mem.created_at,
            extraction_version,
            content_hash,
            json.dumps(mem.metadata),
        ),
    )

    memory_row = conn.execute(
        "SELECT memory_id FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if memory_row is None:
        return 0, 0
    memory_id: int = memory_row["memory_id"]

    # Check if this was newly inserted by verifying evidence doesn't exist yet
    existing_evidence = conn.execute(
        "SELECT evidence_id FROM memory_evidence WHERE memory_id = ? LIMIT 1",
        (memory_id,),
    ).fetchone()

    if existing_evidence is not None:
        return 0, 0

    # Newly inserted - add to FTS and create evidence
    entities_added = 0

    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) "
        "VALUES (?, ?, ?, ?, 'active')",
        (memory_id, mem.statement, mem.kind, mem.project or ""),
    )

    # Find passage_id
    passage_row = conn.execute(
        """
        SELECT passage_id FROM passages
        WHERE message_id = ? AND start_offset <= ? AND end_offset >= ?
        LIMIT 1
        """,
        (mem.message_id, mem.start_offset, mem.end_offset),
    ).fetchone()

    if passage_row is None:
        passage_row = conn.execute(
            "SELECT passage_id FROM passages WHERE message_id = ? LIMIT 1",
            (mem.message_id,),
        ).fetchone()

    passage_id: int | None = passage_row["passage_id"] if passage_row else None

    conn.execute(
        """
        INSERT INTO memory_evidence
          (memory_id, passage_id, message_id, quote, start_offset, end_offset)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            passage_id,
            mem.message_id,
            mem.quote,
            mem.start_offset,
            mem.end_offset,
        ),
    )

    # Upsert entities
    for entity_name in mem.entities:
        conn.execute(
            "INSERT OR IGNORE INTO entities(name, normalized_name) VALUES (?, lower(?))",
            (entity_name, entity_name),
        )
        entity_row = conn.execute(
            "SELECT entity_id FROM entities WHERE normalized_name = lower(?)",
            (entity_name,),
        ).fetchone()
        if entity_row is not None:
            entity_id: int = entity_row["entity_id"]
            conn.execute(
                """
                INSERT OR IGNORE INTO entity_mentions
                  (entity_id, conversation_id, message_id, memory_id)
                VALUES (?, ?, ?, ?)
                """,
                (entity_id, mem.conversation_id, mem.message_id, memory_id),
            )
            entities_added += 1

    return 1, entities_added


def store_extracted_memories(
    conn: sqlite3.Connection,
    memories: list[ExtractedMemory] | tuple[ExtractedMemory, ...],
    *,
    extraction_version: str = "llm-v1",
) -> MemoryExtractionSummary:
    """Persist already-extracted ``ExtractedMemory`` objects through the rules store path.

    Exists so the opt-in LLM extractor (``memory.llm_extract.propose_memories``) can commit
    its accepted proposals with exactly the same dedup, evidence, entity, and
    supersession/contest behaviour the rules path uses -- it reuses ``_insert_memory`` and
    the shared ``reconcile()``, never a parallel insertion route. ``extracted`` mirrors the
    number of proposals handed in; ``inserted`` counts the ones that were new.
    """
    total_inserted = 0
    total_entities = 0
    for mem in memories:
        inserted, entities = _insert_memory(conn, mem, extraction_version)
        total_inserted += inserted
        total_entities += entities

    superseded_count, contested_count = reconcile(conn)
    conn.commit()

    return MemoryExtractionSummary(
        extracted=len(memories),
        inserted=total_inserted,
        superseded=superseded_count,
        contested=contested_count,
        entities=total_entities,
    )


# Every signal that means "a user deliberately curated this memory", and therefore that a
# purge must never delete it. This is the single source of truth for "curated" -- if you
# add a new user-facing mutation that lets someone mark up a memory (a new flag, a new
# history table, anything a person does on purpose rather than the extractor/reconciler
# doing it automatically), add its signal to this list. Do NOT add a second, parallel
# curation check elsewhere; that duplication is exactly how this predicate went stale
# before (task_state_history shipped in migration 009 without being added here).
#
# Current signals:
#   - `pinned = 1` -- explicit pin (set_memory_pinned)
#   - `reviewed_at IS NOT NULL` -- explicit confirm/invalidate (confirm_memory/invalidate_memory)
#   - a `memory_status_history` row with a non-NULL `reason` -- a manual status change via
#     `set_memory_status(..., reason=...)`. `reconcile()`'s automatic supersede/contest
#     bookkeeping always writes `reason IS NULL`, so this does not catch ordinary reconciliation.
#   - any `task_state_history` row -- a task completion/reopen via `set_task_state` (CLI/API).
#     Unlike status history, ANY row counts (not just ones with a reason): every row in this
#     table is already the result of a deliberate user action, there is no automatic writer.
_CURATION_PREDICATE = """(
    pinned = 1
    OR reviewed_at IS NOT NULL
    OR memory_id IN (SELECT memory_id FROM memory_status_history WHERE reason IS NOT NULL)
    OR memory_id IN (SELECT memory_id FROM task_state_history)
)"""


def _purge_candidates(
    conn: sqlite3.Connection,
    *,
    extraction_version: str | None,
) -> tuple[int, list[int]]:
    """Shared scoping logic for `clear_memories`/`preview_purge`: total rows in scope and

    the ids among them that are NOT curated (i.e. the ones a purge would delete). See
    `_CURATION_PREDICATE` for what "curated" means.
    """
    scope_where: list[str] = []
    scope_params: list[object] = []
    if extraction_version is not None:
        scope_where.append("extraction_version = ?")
        scope_params.append(extraction_version)
    scope_sql = f" WHERE {' AND '.join(scope_where)}" if scope_where else ""

    total_in_scope: int = conn.execute(
        f"SELECT COUNT(*) FROM memories{scope_sql}", scope_params
    ).fetchone()[0]

    if total_in_scope == 0:
        return 0, []

    delete_where = [*scope_where, f"NOT {_CURATION_PREDICATE}"]
    rows = conn.execute(
        f"SELECT memory_id FROM memories WHERE {' AND '.join(delete_where)}",
        scope_params,
    ).fetchall()
    memory_ids: list[int] = [row["memory_id"] for row in rows]
    return total_in_scope, memory_ids


def preview_purge(
    conn: sqlite3.Connection,
    *,
    extraction_version: str | None = None,
) -> MemoryPurgeSummary:
    """Read-only preview of what `clear_memories(conn, extraction_version=...)` would do.

    Same scoping and preservation rules, no writes -- for confirmation prompts that need
    to state concrete counts before the user commits to a destructive purge.
    """
    total_in_scope, memory_ids = _purge_candidates(conn, extraction_version=extraction_version)
    return MemoryPurgeSummary(deleted=len(memory_ids), preserved=total_in_scope - len(memory_ids))


def clear_memories(
    conn: sqlite3.Connection,
    *,
    extraction_version: str | None = None,
) -> MemoryPurgeSummary:
    """Purge memories so they can be re-extracted from scratch, e.g. after a change to the

    extraction rules or quality filter that an older `extraction_version` predates.

    Scope: all memories, or only those with the given `extraction_version` if provided.

    Preserved (never deleted), even within scope: any memory the user has curated. See
    `_CURATION_PREDICATE` for the exhaustive, single-source-of-truth list of what counts
    (pinned, reviewed, manually status-changed, or task-state-changed). Losing curated
    work to a re-extraction would be worse than leaving stale rows in place.

    Dependent rows (`memory_evidence`, `memory_status_history`, `task_state_history`,
    `memory_fts`) are deleted for each purged memory. `memory_relations` rows are deleted
    if EITHER endpoint is purged -- a relation naming a memory that no longer exists is not
    a relation worth keeping, even if its other endpoint happens to survive.
    `entity_mentions.memory_id` is cleared (not deleted -- the mention of the entity in the
    message is still real) for purged memories.

    Returns the count deleted and the count preserved (within scope, if scoped).
    """
    total_in_scope, memory_ids = _purge_candidates(conn, extraction_version=extraction_version)

    if total_in_scope == 0:
        return MemoryPurgeSummary(deleted=0, preserved=0)

    if not memory_ids:
        return MemoryPurgeSummary(deleted=0, preserved=total_in_scope)

    placeholders = ",".join("?" for _ in memory_ids)
    conn.execute(f"DELETE FROM memory_fts WHERE rowid IN ({placeholders})", memory_ids)
    conn.execute(f"DELETE FROM memory_evidence WHERE memory_id IN ({placeholders})", memory_ids)
    conn.execute(
        f"DELETE FROM memory_status_history WHERE memory_id IN ({placeholders})", memory_ids
    )
    conn.execute(f"DELETE FROM task_state_history WHERE memory_id IN ({placeholders})", memory_ids)
    conn.execute(
        f"UPDATE entity_mentions SET memory_id = NULL WHERE memory_id IN ({placeholders})",
        memory_ids,
    )
    conn.execute(
        f"DELETE FROM memory_relations WHERE from_memory_id IN ({placeholders}) "
        f"OR to_memory_id IN ({placeholders})",
        memory_ids + memory_ids,
    )
    conn.execute(f"DELETE FROM memories WHERE memory_id IN ({placeholders})", memory_ids)
    conn.commit()

    return MemoryPurgeSummary(
        deleted=len(memory_ids),
        preserved=total_in_scope - len(memory_ids),
    )


def reconcile(conn: sqlite3.Connection) -> tuple[int, int]:
    """Reconcile decision and preference memories: mark superseded/contested."""
    superseded_count = 0
    contested_count = 0

    groups = conn.execute(
        """
        SELECT kind, subject_key, project
        FROM memories
        WHERE kind IN ('decision', 'preference')
        GROUP BY kind, subject_key, project
        HAVING COUNT(*) > 1
        """
    ).fetchall()

    for group in groups:
        kind: str = group["kind"]
        subject_key: str = group["subject_key"]
        project: str | None = group["project"]

        members = conn.execute(
            """
            SELECT memory_id, status, created_at, message_id
            FROM memories
            WHERE kind = ? AND subject_key = ? AND (project = ? OR (project IS NULL AND ? IS NULL))
            ORDER BY (created_at IS NULL), created_at, message_id
            """,
            (kind, subject_key, project, project),
        ).fetchall()

        if len(members) < 2:
            continue

        # Last one is 'active' (newest by ordering), earlier ones are superseded
        # Unless they share the same created_at (both NULL or same value) -> contested
        newest = members[-1]
        newest_date_key = (newest["created_at"] is None, newest["created_at"])

        for earlier in members[:-1]:
            earlier_date_key = (
                earlier["created_at"] is None,
                earlier["created_at"],
            )

            if earlier_date_key == newest_date_key:
                # contested
                for mem_id in (earlier["memory_id"], newest["memory_id"]):
                    mem_row = conn.execute(
                        "SELECT status FROM memories WHERE memory_id = ?", (mem_id,)
                    ).fetchone()
                    if mem_row and mem_row["status"] != "contested":
                        _write_status_history(conn, mem_id, mem_row["status"], "contested", None)
                        conn.execute(
                            "UPDATE memories SET status = 'contested' WHERE memory_id = ?",
                            (mem_id,),
                        )
                        _update_memory_fts_status(conn, mem_id, "contested")
                        contested_count += 1

                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_relations
                      (from_memory_id, to_memory_id, relation, reason)
                    VALUES (?, ?, 'conflicts_with', NULL)
                    """,
                    (earlier["memory_id"], newest["memory_id"]),
                )
            else:
                # newer supersedes older
                earlier_row = conn.execute(
                    "SELECT status FROM memories WHERE memory_id = ?",
                    (earlier["memory_id"],),
                ).fetchone()
                if earlier_row and earlier_row["status"] not in ("superseded", "contested"):
                    _write_status_history(
                        conn, earlier["memory_id"], earlier_row["status"], "superseded", None
                    )
                    conn.execute(
                        "UPDATE memories SET status = 'superseded' WHERE memory_id = ?",
                        (earlier["memory_id"],),
                    )
                    _update_memory_fts_status(conn, earlier["memory_id"], "superseded")
                    superseded_count += 1

                # ensure newest is active (if not contested)
                newest_row = conn.execute(
                    "SELECT status FROM memories WHERE memory_id = ?",
                    (newest["memory_id"],),
                ).fetchone()
                if newest_row and newest_row["status"] not in ("active", "contested"):
                    _write_status_history(
                        conn, newest["memory_id"], newest_row["status"], "active", None
                    )
                    conn.execute(
                        "UPDATE memories SET status = 'active' WHERE memory_id = ?",
                        (newest["memory_id"],),
                    )
                    _update_memory_fts_status(conn, newest["memory_id"], "active")

                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_relations
                      (from_memory_id, to_memory_id, relation, reason)
                    VALUES (?, ?, 'supersedes', NULL)
                    """,
                    (newest["memory_id"], earlier["memory_id"]),
                )

    return superseded_count, contested_count


def _write_status_history(
    conn: sqlite3.Connection,
    memory_id: int,
    old_status: str,
    new_status: str,
    reason: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO memory_status_history (memory_id, old_status, new_status, reason)
        VALUES (?, ?, ?, ?)
        """,
        (memory_id, old_status, new_status, reason),
    )


def _update_memory_fts_status(
    conn: sqlite3.Connection,
    memory_id: int,
    new_status: str,
) -> None:
    # FTS5 doesn't support UPDATE in place; delete and reinsert
    mem_row = conn.execute(
        "SELECT statement, kind, project FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if mem_row is None:
        return
    conn.execute("DELETE FROM memory_fts WHERE rowid = ?", (memory_id,))
    conn.execute(
        "INSERT INTO memory_fts(rowid, statement, kind, project, status) VALUES (?, ?, ?, ?, ?)",
        (memory_id, mem_row["statement"], mem_row["kind"], mem_row["project"] or "", new_status),
    )


def set_memory_status(
    conn: sqlite3.Connection,
    memory_id: int,
    new_status: str,
    *,
    reason: str | None = None,
) -> None:
    """Update the status of a memory, recording history."""
    if new_status not in MEMORY_STATUSES:
        raise ValueError(f"Invalid memory status: {new_status!r}")

    row = conn.execute("SELECT status FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
    if row is None:
        raise ValueError(f"Memory not found: {memory_id}")

    old_status: str = row["status"]
    _write_status_history(conn, memory_id, old_status, new_status, reason)
    conn.execute(
        "UPDATE memories SET status = ? WHERE memory_id = ?",
        (new_status, memory_id),
    )
    _update_memory_fts_status(conn, memory_id, new_status)


def _write_task_state_history(
    conn: sqlite3.Connection,
    memory_id: int,
    old_state: str | None,
    new_state: str,
    reason: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO task_state_history (memory_id, old_state, new_state, reason)
        VALUES (?, ?, ?, ?)
        """,
        (memory_id, old_state, new_state, reason),
    )


def set_task_state(
    conn: sqlite3.Connection,
    memory_id: int,
    new_state: str,
    *,
    reason: str | None = None,
) -> None:
    """Update the task_state of a task memory, recording history.

    Mirrors `set_memory_status` above, with two extra invariants specific to tasks:
    - the memory must be `kind='task'` -- silently letting a decision or preference grow a
      task_state would be a data-integrity bug, not a feature.
    - a no-op (new_state == current task_state) returns early without writing a history row
      or touching `task_state_changed_at`, so re-completing an already-completed task does
      not pollute the audit trail with a fake transition.

    Writes the transition history row and updates the memory row in the same transaction
    (the caller commits).
    """
    if new_state not in TASK_STATES:
        raise ValueError(f"Invalid task state: {new_state!r}")

    row = conn.execute(
        "SELECT kind, task_state FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Memory not found: {memory_id}")
    if row["kind"] != "task":
        raise ValueError(f"Memory {memory_id} is kind={row['kind']!r}, not a task")

    old_state: str | None = row["task_state"]
    if old_state == new_state:
        return

    _write_task_state_history(conn, memory_id, old_state, new_state, reason)
    conn.execute(
        "UPDATE memories SET task_state = ?, task_state_changed_at = CURRENT_TIMESTAMP "
        "WHERE memory_id = ?",
        (new_state, memory_id),
    )
