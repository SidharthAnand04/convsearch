from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from convsearch.evaluation.models import (
    EvaluationCheck,
    EvaluationMetrics,
    EvaluationQueryCase,
    EvaluationQueryResult,
    EvaluationRun,
    EvaluationStatus,
)


def connect_run_store(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_run_store(path: Path) -> None:
    with connect_run_store(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                is_ephemeral INTEGER NOT NULL,
                data_directory TEXT NOT NULL,
                data_manifest_hash TEXT NOT NULL,
                embedding_provider TEXT NOT NULL,
                error_message TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS evaluation_checks (
                check_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                check_name TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                expected_value TEXT,
                actual_value TEXT,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS evaluation_queries (
                query_result_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                case_id TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                expected_conversation_ids_json TEXT NOT NULL,
                returned_conversation_ids_json TEXT NOT NULL DEFAULT '[]',
                expected_rank INTEGER,
                actual_rank INTEGER,
                reciprocal_rank REAL NOT NULL DEFAULT 0,
                latency_ms REAL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )


def create_run(
    path: Path, run: EvaluationRun, checks: list[EvaluationCheck], cases: list[EvaluationQueryCase]
) -> None:
    with connect_run_store(path) as conn, conn:
        conn.execute(
            """
            INSERT INTO evaluation_runs(
                run_id, started_at, status, is_ephemeral, data_directory,
                data_manifest_hash, embedding_provider
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.started_at,
                run.status,
                int(run.is_ephemeral),
                run.data_directory,
                run.data_manifest_hash,
                run.embedding_provider,
            ),
        )
        conn.executemany(
            """
            INSERT INTO evaluation_checks(
                run_id, check_name, category, status, expected_value, actual_value, details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.run_id,
                    check.check_name,
                    check.category,
                    check.status,
                    check.expected_value,
                    check.actual_value,
                    json.dumps(check.details),
                )
                for check in checks
            ],
        )
        conn.executemany(
            """
            INSERT INTO evaluation_queries(
                run_id, case_id, query, status, expected_conversation_ids_json, expected_rank
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.run_id,
                    case.case_id,
                    case.query,
                    "pending",
                    json.dumps(case.expected_conversation_ids),
                    case.minimum_expected_rank,
                )
                for case in cases
            ],
        )


def update_check(path: Path, run_id: str, check: EvaluationCheck) -> None:
    with connect_run_store(path) as conn, conn:
        conn.execute(
            """
            UPDATE evaluation_checks
            SET status = ?, expected_value = ?, actual_value = ?, details_json = ?
            WHERE run_id = ? AND check_name = ?
            """,
            (
                check.status,
                check.expected_value,
                check.actual_value,
                json.dumps(check.details),
                run_id,
                check.check_name,
            ),
        )


def update_query(path: Path, run_id: str, result: EvaluationQueryResult) -> None:
    with connect_run_store(path) as conn, conn:
        conn.execute(
            """
            UPDATE evaluation_queries
            SET status = ?, returned_conversation_ids_json = ?, actual_rank = ?,
                reciprocal_rank = ?, latency_ms = ?, details_json = ?
            WHERE run_id = ? AND case_id = ?
            """,
            (
                result.status,
                json.dumps(result.returned_conversation_ids),
                result.actual_rank,
                result.reciprocal_rank,
                result.latency_ms,
                json.dumps(result.details),
                run_id,
                result.case_id,
            ),
        )


def finalize_run(
    path: Path,
    run_id: str,
    status: EvaluationStatus,
    finished_at: str,
    metrics: EvaluationMetrics,
    error_message: str | None = None,
) -> None:
    with connect_run_store(path) as conn, conn:
        conn.execute(
            """
            UPDATE evaluation_runs
            SET status = ?, finished_at = ?, metrics_json = ?, error_message = ?
            WHERE run_id = ?
            """,
            (status, finished_at, metrics.model_dump_json(), error_message, run_id),
        )
