from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from convsearch.config.settings import vector_map_path
from convsearch.evaluation.models import EvaluationCheck
from convsearch.indexes.lexical import fts_count
from convsearch.storage.database import current_migrations, verify_fts5
from convsearch.storage.migrations import migration_files

EXPECTED_TABLES = {
    "imports",
    "conversations",
    "messages",
    "passages",
    "embedding_records",
    "index_metadata",
    "passage_fts",
}
ROUTING_COLUMNS = {"source_node_id", "parent_source_node_id", "resolved_parent_message_id"}


def inspect_database(
    workspace: Path, conn: sqlite3.Connection, manifest: dict[str, object]
) -> tuple[list[EvaluationCheck], dict[str, int]]:
    checks: list[EvaluationCheck] = []
    counts = collect_counts(workspace, conn)

    expected_migrations = {path.stem for path in migration_files()}
    applied = current_migrations(conn)
    checks.append(check_set("migrations_applied", "schema", expected_migrations, applied))

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        )
    }
    checks.append(check_set("expected_tables", "schema", EXPECTED_TABLES, tables))

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
    checks.append(check_set("routing_columns", "schema", ROUTING_COLUMNS, columns))

    try:
        verify_fts5(conn)
        checks.append(pass_check("fts5_available", "schema", "available", "available"))
    except Exception as exc:
        checks.append(fail_check("fts5_available", "schema", "available", str(exc)))

    checks.extend(count_checks(counts, manifest))
    checks.extend(referential_checks(conn))
    checks.extend(routing_checks(conn, manifest))
    checks.extend(index_checks(workspace, conn, counts))
    return checks, counts


def collect_counts(workspace: Path, conn: sqlite3.Connection) -> dict[str, int]:
    vector_count = 0
    if vector_map_path(workspace).exists():
        vector_count = len(
            json.loads(vector_map_path(workspace).read_text(encoding="utf-8")).get(
                "passage_ids", []
            )
        )
    return {
        "imports": safe_count(conn, "imports"),
        "conversations": safe_count(conn, "conversations"),
        "messages": safe_count(conn, "messages"),
        "passages": safe_count(conn, "passages"),
        "fts_records": safe_fts_count(conn),
        "embedding_records": safe_count(conn, "embedding_records"),
        "vector_map_records": vector_count,
        "primary_messages": safe_scalar(
            conn, "SELECT count(*) FROM messages WHERE is_primary_path = 1"
        ),
        "alternate_messages": safe_scalar(
            conn, "SELECT count(*) FROM messages WHERE is_primary_path = 0"
        ),
    }


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def scalar(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def safe_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return count(conn, table)
    except sqlite3.DatabaseError:
        return -1


def safe_scalar(conn: sqlite3.Connection, sql: str) -> int:
    try:
        return scalar(conn, sql)
    except sqlite3.DatabaseError:
        return -1


def safe_fts_count(conn: sqlite3.Connection) -> int:
    try:
        return fts_count(conn)
    except sqlite3.DatabaseError:
        return -1


def count_checks(counts: dict[str, int], manifest: dict[str, object]) -> list[EvaluationCheck]:
    checks = [
        compare_int(
            "conversation_count",
            "counts",
            manifest_int(manifest, "conversation_count"),
            counts["conversations"],
        ),
        compare_min(
            "minimum_message_count",
            "counts",
            manifest_int(manifest, "minimum_message_count"),
            counts["messages"],
        ),
        compare_int(
            "primary_message_count",
            "counts",
            manifest_int(manifest, "expected_primary_message_count"),
            counts["primary_messages"],
        ),
        compare_int(
            "alternate_message_count",
            "counts",
            manifest_int(manifest, "expected_alternate_message_count"),
            counts["alternate_messages"],
        ),
    ]
    checks.append(
        compare_int("fts_matches_passages", "index", counts["passages"], counts["fts_records"])
    )
    checks.append(
        compare_int(
            "embedding_matches_passages", "index", counts["passages"], counts["embedding_records"]
        )
    )
    checks.append(
        compare_int(
            "vector_map_matches_embeddings",
            "index",
            counts["embedding_records"],
            counts["vector_map_records"],
        )
    )
    return checks


def referential_checks(conn: sqlite3.Connection) -> list[EvaluationCheck]:
    checks: list[EvaluationCheck] = []
    fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    checks.append(compare_int("foreign_key_check", "referential", 0, len(fk_rows)))
    checks.append(
        compare_int(
            "orphan_messages",
            "referential",
            0,
            safe_scalar(
                conn,
                """
                SELECT count(*)
                FROM messages m
                LEFT JOIN conversations c ON c.conversation_id = m.conversation_id
                WHERE c.conversation_id IS NULL
                """,
            ),
        )
    )
    checks.append(
        compare_int(
            "orphan_passages",
            "referential",
            0,
            safe_scalar(
                conn,
                """
                SELECT count(*)
                FROM passages p
                LEFT JOIN messages m ON m.message_id = p.message_id
                WHERE m.message_id IS NULL
                """,
            ),
        )
    )
    checks.append(
        compare_int(
            "orphan_embeddings",
            "referential",
            0,
            safe_scalar(
                conn,
                """
                SELECT count(*)
                FROM embedding_records e
                LEFT JOIN passages p ON p.passage_id = e.passage_id
                WHERE p.passage_id IS NULL
                """,
            ),
        )
    )
    checks.append(
        compare_int(
            "missing_resolved_parents",
            "referential",
            0,
            safe_scalar(
                conn,
                """
                SELECT count(*)
                FROM messages child
                LEFT JOIN messages parent
                    ON parent.message_id = child.resolved_parent_message_id
                WHERE child.resolved_parent_message_id IS NOT NULL
                    AND parent.message_id IS NULL
                """,
            ),
        )
    )
    checks.append(
        compare_int(
            "duplicate_source_nodes",
            "referential",
            0,
            safe_scalar(
                conn,
                """
                SELECT count(*)
                FROM (
                    SELECT conversation_id, source_node_id
                    FROM messages
                    GROUP BY conversation_id, source_node_id
                    HAVING count(*) > 1
                )
                """,
            ),
        )
    )
    return checks


def routing_checks(conn: sqlite3.Connection, manifest: dict[str, object]) -> list[EvaluationCheck]:
    distinct = safe_scalar(
        conn, "SELECT count(*) FROM messages WHERE source_node_id != source_message_id"
    )
    resolved = safe_scalar(
        conn, "SELECT count(*) FROM messages WHERE resolved_parent_message_id IS NOT NULL"
    )
    alternate = safe_scalar(conn, "SELECT count(*) FROM messages WHERE is_primary_path = 0")
    primary = safe_scalar(conn, "SELECT count(*) FROM messages WHERE is_primary_path = 1")
    return [
        compare_bool(
            "distinct_node_and_message_ids",
            "routing",
            bool(manifest["has_distinct_node_and_message_ids"]),
            distinct > 0,
        ),
        compare_bool(
            "has_alternate_branches",
            "routing",
            bool(manifest["has_alternate_branches"]),
            alternate > 0,
        ),
        compare_min("resolved_parent_relationships", "routing", 1, resolved),
        compare_min("valid_primary_path", "routing", 1, primary),
    ]


def index_checks(
    workspace: Path, conn: sqlite3.Connection, counts: dict[str, int]
) -> list[EvaluationCheck]:
    checks = [
        compare_int(
            "unique_vector_ids",
            "index",
            counts["embedding_records"],
            safe_scalar(conn, "SELECT count(DISTINCT vector_id) FROM embedding_records"),
        ),
    ]
    try:
        metadata = {
            row["key"]: json.loads(row["value"])
            for row in conn.execute("SELECT key, value FROM index_metadata").fetchall()
        }
    except sqlite3.DatabaseError:
        metadata = {}
    checks.append(
        compare_int(
            "metadata_passage_count",
            "index",
            counts["passages"],
            int(metadata.get("passage_count", -1)),
        )
    )
    checks.append(compare_bool("metadata_model_id", "index", True, bool(metadata.get("model_id"))))
    checks.append(
        compare_min(
            "embedding_dimension",
            "index",
            1 if counts["passages"] else 0,
            int(metadata.get("embedding_dimension", 0)),
        )
    )
    checks.append(
        compare_bool("vector_map_exists", "index", True, vector_map_path(workspace).exists())
    )
    return checks


def pass_check(name: str, category: str, expected: object, actual: object) -> EvaluationCheck:
    return EvaluationCheck(
        check_name=name,
        category=category,
        status="passed",
        expected_value=str(expected),
        actual_value=str(actual),
    )


def fail_check(name: str, category: str, expected: object, actual: object) -> EvaluationCheck:
    return EvaluationCheck(
        check_name=name,
        category=category,
        status="failed",
        expected_value=str(expected),
        actual_value=str(actual),
    )


def compare_int(name: str, category: str, expected: int, actual: int) -> EvaluationCheck:
    return (
        pass_check(name, category, expected, actual)
        if expected == actual
        else fail_check(name, category, expected, actual)
    )


def compare_min(name: str, category: str, expected_min: int, actual: int) -> EvaluationCheck:
    return (
        pass_check(name, category, f">={expected_min}", actual)
        if actual >= expected_min
        else fail_check(name, category, f">={expected_min}", actual)
    )


def compare_bool(name: str, category: str, expected: bool, actual: bool) -> EvaluationCheck:
    return (
        pass_check(name, category, expected, actual)
        if expected == actual
        else fail_check(name, category, expected, actual)
    )


def check_set(name: str, category: str, expected: set[str], actual: set[str]) -> EvaluationCheck:
    missing = sorted(expected - actual)
    return EvaluationCheck(
        check_name=name,
        category=category,
        status="passed" if not missing else "failed",
        expected_value=",".join(sorted(expected)),
        actual_value=",".join(sorted(actual)),
        details={"missing": missing},
    )


def manifest_int(manifest: dict[str, object], key: str) -> int:
    value = manifest[key]
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Manifest value is not an integer: {key}")
