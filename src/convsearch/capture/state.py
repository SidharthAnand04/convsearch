"""Durable bookkeeping for live capture.

Both facts the extension needs to see across server restarts live in SQLite so they
survive a process restart without an extra file to keep in sync:

* whether conversations have been captured since the last index build (`stale_index`), and
* how many conversations arrived through live capture rather than an export ZIP.

Capture rows are attributed to a single synthetic `imports` row (the `conversations`
table requires an `import_id`), which doubles as the marker for "this came from the
browser".
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from convsearch.storage.database import connection

CAPTURE_SOURCE_HASH = "live-capture"
CAPTURE_SOURCE_PATH = "live-capture"
STALE_INDEX_KEY = "capture_stale_index"


def ensure_capture_import(conn: sqlite3.Connection) -> int:
    """Return the id of the synthetic import row that owns all captured conversations."""
    conn.execute(
        """
        INSERT INTO imports(source_path, source_hash, status, warning_count, metadata_json)
        VALUES (?, ?, 'complete', 0, ?)
        ON CONFLICT(source_hash) DO NOTHING
        """,
        (
            CAPTURE_SOURCE_PATH,
            CAPTURE_SOURCE_HASH,
            json.dumps({"parser": "chrome-extension-live-capture"}),
        ),
    )
    row = conn.execute(
        "SELECT import_id FROM imports WHERE source_hash = ?", (CAPTURE_SOURCE_HASH,)
    ).fetchone()
    if row is None:  # pragma: no cover - the insert above guarantees a row
        raise RuntimeError("Could not create the live-capture import row")
    return int(row["import_id"])


def set_index_stale(conn: sqlite3.Connection, stale: bool) -> None:
    conn.execute(
        """
        INSERT INTO index_metadata(key, value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """,
        (STALE_INDEX_KEY, json.dumps(stale)),
    )


def read_index_stale(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (STALE_INDEX_KEY,)
        ).fetchone()
    except sqlite3.DatabaseError:
        return False
    if row is None:
        return False
    try:
        return bool(json.loads(str(row["value"])))
    except json.JSONDecodeError:
        return False


def count_captured_conversations(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM conversations c
            JOIN imports i ON i.import_id = c.import_id
            WHERE i.source_hash = ?
            """,
            (CAPTURE_SOURCE_HASH,),
        ).fetchone()
    except sqlite3.DatabaseError:
        return 0
    return int(row["total"]) if row is not None else 0


def clear_index_stale(workspace: Path) -> None:
    """Record that the vector index is now current. Called after every index build."""
    with connection(workspace) as conn, conn:
        set_index_stale(conn, False)
