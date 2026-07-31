from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from convsearch.config.settings import Settings, database_path, faiss_index_path, vector_map_path
from convsearch.indexes.lexical import fts_count
from convsearch.storage.database import (
    connection,
    current_migrations,
    pending_migrations,
    verify_fts5,
)
from convsearch.storage.migrations import migration_files


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(workspace: Path, settings: Settings, load_model: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("workspace", workspace.exists(), str(workspace)))
    db_path = database_path(workspace)
    checks.append(Check("database", db_path.exists(), str(db_path)))
    if db_path.exists():
        try:
            with connection(workspace) as conn:
                verify_fts5(conn)
                checks.append(Check("sqlite_fts5", True, "available"))
                applied = current_migrations(conn)
                expected = {path.stem for path in migration_files()}
                pending = pending_migrations(conn)
                detail = f"{len(applied)}/{len(expected)} applied"
                if pending:
                    detail += (
                        f"; pending: {', '.join(pending)} -- run: convsearch migrate -w {workspace}"
                    )
                checks.append(Check("migrations", not pending, detail))
                passage_count = int(
                    conn.execute("SELECT count(*) AS count FROM passages").fetchone()["count"]
                )
                embedded_count = int(
                    conn.execute("SELECT count(*) AS count FROM embedding_records").fetchone()[
                        "count"
                    ]
                )
                checks.append(
                    Check(
                        "passage_embedding_count",
                        passage_count == embedded_count,
                        f"{embedded_count}/{passage_count}",
                    )
                )
                checks.append(
                    Check("fts_count", fts_count(conn) == passage_count, str(fts_count(conn)))
                )
                segment_count = (
                    scalar_count(conn, "segments") if _table_exists(conn, "segments") else 0
                )
                unsegmented = (
                    int(
                        conn.execute(
                            "SELECT count(*) AS count FROM passages WHERE segment_id IS NULL"
                        ).fetchone()["count"]
                    )
                    if _column_exists(conn, "passages", "segment_id")
                    else passage_count
                )
                state = "current" if passage_count == 0 or unsegmented == 0 else "missing"
                checks.append(Check("segment_state", state == "current", state))
                checks.append(Check("segment_count", segment_count >= 0, str(segment_count)))

                # --- memory checks ---
                memory_tables = {"memories", "memory_evidence", "memory_relations", "entities"}
                missing_tables = [t for t in sorted(memory_tables) if not _table_exists(conn, t)]
                checks.append(
                    Check(
                        "memory_tables",
                        len(missing_tables) == 0,
                        "all present"
                        if not missing_tables
                        else f"missing: {', '.join(missing_tables)}",
                    )
                )

                if _table_exists(conn, "memories"):
                    status_rows = conn.execute(
                        "SELECT status, count(*) AS n FROM memories GROUP BY status ORDER BY status"
                    ).fetchall()
                    if status_rows:
                        summary_parts = [f"{r['status']}={r['n']}" for r in status_rows]
                        memory_summary = ", ".join(summary_parts)
                    else:
                        memory_summary = "no memories extracted yet"
                    checks.append(Check("memory_counts", True, memory_summary))

                    # memory FTS integrity
                    if _table_exists(conn, "memory_fts"):
                        fts_count_val = int(
                            conn.execute("SELECT count(*) AS n FROM memory_fts").fetchone()["n"]
                        )
                        mem_count_val = int(
                            conn.execute("SELECT count(*) AS n FROM memories").fetchone()["n"]
                        )
                        checks.append(
                            Check(
                                "memory_fts_integrity",
                                fts_count_val == mem_count_val,
                                f"fts={fts_count_val} memories={mem_count_val}",
                            )
                        )
                    else:
                        checks.append(
                            Check("memory_fts_integrity", True, "memory_fts table not present")
                        )

                    # orphan evidence
                    if _table_exists(conn, "memory_evidence"):
                        orphan_count = int(
                            conn.execute(
                                """
                                SELECT count(*) AS n FROM memory_evidence
                                WHERE memory_id NOT IN (SELECT memory_id FROM memories)
                                """
                            ).fetchone()["n"]
                        )
                        checks.append(
                            Check(
                                "orphan_evidence",
                                orphan_count == 0,
                                f"{orphan_count} orphan row(s)" if orphan_count else "none",
                            )
                        )

                # stale vector index
                meta_row = conn.execute(
                    "SELECT value FROM index_metadata WHERE key = 'passage_count'"
                ).fetchone()
                if meta_row is None:
                    checks.append(Check("stale_vector_index", True, "index not built"))
                else:
                    import json as _json

                    indexed_count = int(_json.loads(meta_row["value"]))
                    current_count = int(
                        conn.execute("SELECT count(*) AS n FROM passages").fetchone()["n"]
                    )
                    checks.append(
                        Check(
                            "stale_vector_index",
                            indexed_count == current_count,
                            f"indexed={indexed_count} current={current_count}",
                        )
                    )

        except Exception as exc:
            checks.append(Check("database_checks", False, str(exc)))
    checks.append(Check("faiss_import", importlib.util.find_spec("faiss") is not None, "faiss-cpu"))
    checks.append(
        Check(
            "vector_files",
            vector_map_path(workspace).exists()
            and (
                faiss_index_path(workspace).exists()
                or faiss_index_path(workspace).with_suffix(".npy").exists()
            ),
            str(faiss_index_path(workspace)),
        )
    )
    metadata_model = None
    if db_path.exists():
        with connection(workspace) as conn:
            row = conn.execute("SELECT value FROM index_metadata WHERE key = 'model_id'").fetchone()
            metadata_model = json.loads(row["value"]) if row else None
    checks.append(
        Check(
            "embedding_model_metadata",
            metadata_model in (None, settings.embedding_model),
            f"configured={settings.embedding_model}; indexed={metadata_model}",
        )
    )
    if load_model:
        try:
            from convsearch.embeddings.sentence_transformers import (
                SentenceTransformerEmbeddingProvider,
            )

            SentenceTransformerEmbeddingProvider(
                settings.embedding_model, settings.embedding_device
            )
            checks.append(Check("embedding_model_load", True, settings.embedding_model))
        except Exception as exc:
            checks.append(Check("embedding_model_load", False, str(exc)))
    return checks


def stats(workspace: Path) -> dict[str, object]:
    db_path = database_path(workspace)
    with connection(workspace) as conn:
        values: dict[str, object] = {
            "imports": scalar_count(conn, "imports"),
            "conversations": scalar_count(conn, "conversations"),
            "messages": scalar_count(conn, "messages"),
            "passages": scalar_count(conn, "passages"),
            "segments": scalar_count(conn, "segments") if _table_exists(conn, "segments") else 0,
            "embedded_passages": scalar_count(conn, "embedding_records"),
            "fts_passages": fts_count(conn),
            "database_size": db_path.stat().st_size if db_path.exists() else 0,
            "faiss_index_size": faiss_index_path(workspace).stat().st_size
            if faiss_index_path(workspace).exists()
            else 0,
        }
        for row in conn.execute("SELECT key, value FROM index_metadata"):
            values[str(row["key"])] = json.loads(row["value"])
    return values


def scalar_count(conn, table: str) -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute(f"SELECT count(*) AS count FROM {table}").fetchone()["count"])


def _table_exists(conn, table: str) -> bool:  # type: ignore[no-untyped-def]
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(conn, table: str, column: str) -> bool:  # type: ignore[no-untyped-def]
    return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})"))
