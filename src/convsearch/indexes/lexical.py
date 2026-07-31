from __future__ import annotations

import sqlite3


def refresh_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM passage_fts")
    conn.execute(
        """
        INSERT INTO passage_fts(rowid, text, role, title)
        SELECT p.passage_id, p.text, m.role, c.title
        FROM passages p
        JOIN messages m ON m.message_id = p.message_id
        JOIN conversations c ON c.conversation_id = p.conversation_id
        """
    )


def sync_fts(conn: sqlite3.Connection, passage_ids: list[int]) -> int:
    """Index just these passages, leaving the rest of the FTS table untouched.

    `refresh_fts` deletes and re-inserts every row, which is O(corpus) — measured at ~30ms
    for 1200 passages. Auto-indexing runs a pass every few seconds while the user browses, so
    paying that on each pass makes the cost of adding one conversation grow with the size of
    the archive. This touches only what changed.

    Rows are deleted before insert so a re-captured (edited) passage cannot end up indexed
    twice. Returns the number of rows written.
    """
    if not passage_ids:
        return 0
    placeholders = ",".join("?" * len(passage_ids))
    conn.execute(f"DELETE FROM passage_fts WHERE rowid IN ({placeholders})", passage_ids)
    cursor = conn.execute(
        f"""
        INSERT INTO passage_fts(rowid, text, role, title)
        SELECT p.passage_id, p.text, m.role, c.title
        FROM passages p
        JOIN messages m ON m.message_id = p.message_id
        JOIN conversations c ON c.conversation_id = p.conversation_id
        WHERE p.passage_id IN ({placeholders})
        """,
        passage_ids,
    )
    return int(cursor.rowcount)


def fts_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) AS count FROM passage_fts").fetchone()["count"])
