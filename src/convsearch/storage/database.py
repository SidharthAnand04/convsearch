from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from convsearch.config.settings import database_path
from convsearch.storage.migrations import migration_files


class DatabaseError(RuntimeError):
    pass


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection(workspace: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(database_path(workspace))
    try:
        yield conn
    finally:
        conn.close()


def verify_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(value)")
        conn.execute("DROP TABLE temp.fts5_probe")
    except sqlite3.DatabaseError as exc:
        raise DatabaseError("SQLite FTS5 is not available in this Python build") from exc


def initialize_database(workspace: Path) -> None:
    apply_pending_migrations(workspace)


def apply_pending_migrations(workspace: Path) -> list[str]:
    """Apply any migrations not yet recorded in `schema_migrations`, idempotently.

    This is the ONLY place that mutates schema. It backs both a brand-new workspace
    (`convsearch init`, via `initialize_database`) and an existing one that predates newer
    migrations (`convsearch migrate`, `convsearch serve` on startup) -- there is no separate
    "upgrade" code path to fall out of sync with this one. Returns the versions that were
    actually applied, in order; an empty list means the workspace was already current. Never
    called from read-only commands (`search`, `tasks list`, `digest`, ...) -- a read must
    never silently rewrite schema out from under the user.
    """
    db_path = database_path(workspace)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    applied_now: list[str] = []
    # `with sqlite3.connect(...)` commits but does NOT close: the connection used to leak
    # until garbage collection. On Windows a live handle keeps a lock on the file, so the
    # workspace could not be moved or deleted afterwards. The nested `with conn` is what
    # commits; the outer contextlib.closing is what lets go of the file.
    with closing(connect(db_path)) as conn, conn:
        verify_fts5(conn)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
        for migration in migration_files():
            version = migration.stem
            if version in applied:
                continue
            conn.executescript(migration.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
            applied_now.append(version)
    return applied_now


def current_migrations(conn: sqlite3.Connection) -> set[str]:
    try:
        return {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
    except sqlite3.DatabaseError:
        return set()


def pending_migrations(conn: sqlite3.Connection) -> list[str]:
    """Migration versions not yet applied to `conn`, in the order they'd be applied."""
    applied = current_migrations(conn)
    return [path.stem for path in migration_files() if path.stem not in applied]
