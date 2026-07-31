from __future__ import annotations

import gc
import json
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from convsearch.config.settings import Settings, database_path
from convsearch.domain.models import PassageHit
from convsearch.embeddings.sentence_transformers import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from convsearch.evaluation.database_inspector import inspect_database
from convsearch.evaluation.metrics import calculate_metrics
from convsearch.evaluation.models import (
    EvaluationCheck,
    EvaluationMetrics,
    EvaluationOptions,
    EvaluationQueryCase,
    EvaluationQueryResult,
    EvaluationReport,
    EvaluationRun,
    EvaluationStatus,
)
from convsearch.evaluation.reporting import report_to_json
from convsearch.evaluation.run_store import (
    create_run,
    finalize_run,
    initialize_run_store,
    update_check,
    update_query,
)
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.indexes.build import build_indexes
from convsearch.retrieval.service import search_conversations
from convsearch.storage.database import connection, initialize_database
from convsearch.utils import sha256_file

BASE_CHECKS = [
    ("fixture_files", "fixture"),
    ("manifest_hashes", "fixture"),
    ("workspace_init", "execution"),
    ("import_idempotency", "execution"),
    ("index_build", "execution"),
    ("memory_extraction", "memory"),
]


def run_synthetic_evaluation(options: EvaluationOptions) -> tuple[EvaluationReport, bool]:
    run_id = uuid.uuid4().hex[:12]
    run_root = options.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / f"convsearch-eval-{run_id}"
    workspace = run_dir / "workspace"
    run_store_path = run_dir / "run.sqlite3"
    run_dir.mkdir(parents=True)
    started = now()
    manifest_hash = (
        sha256_file(options.data_dir / "manifest.json")
        if (options.data_dir / "manifest.json").exists()
        else ""
    )
    provider: EmbeddingProvider = (
        SentenceTransformerEmbeddingProvider(
            Settings.default().embedding_model, Settings.default().embedding_device
        )
        if options.real_model
        else DeterministicEmbeddingProvider()
    )
    run = EvaluationRun(
        run_id=run_id,
        started_at=started,
        status="running",
        is_ephemeral=not options.keep_run,
        data_directory=str(options.data_dir),
        data_manifest_hash=manifest_hash,
        embedding_provider=provider.model_id,
        run_dir=str(run_dir),
    )
    manifest: dict[str, object] = {}
    cases: list[EvaluationQueryCase] = []
    memory_cases: list[dict[str, object]] = []
    segment_cases: list[dict[str, object]] = []
    checks = [
        EvaluationCheck(check_name=name, category=category, status="pending")
        for name, category in BASE_CHECKS
    ]
    try:
        manifest, cases, memory_cases, segment_cases = load_fixture(options.data_dir)
        checks.extend(
            EvaluationCheck(
                check_name=f"query:{case.case_id}", category="retrieval", status="pending"
            )
            for case in cases
        )
        for mc in memory_cases:
            checks.append(
                EvaluationCheck(
                    check_name=f"memory:{mc['case_id']}",
                    category="memory",
                    status="pending",
                )
            )
        for sc in segment_cases:
            checks.append(
                EvaluationCheck(
                    check_name=f"segment:{sc['case_id']}",
                    category="segment",
                    status="pending",
                )
            )
        initialize_run_store(run_store_path)
        create_run(run_store_path, run, checks, cases)
        checks = []
        checks.extend(validate_fixture(options.data_dir, manifest))
        for check in checks:
            update_check(run_store_path, run_id, check)

        initialize_workspace(workspace)
        check = EvaluationCheck(
            check_name="workspace_init",
            category="execution",
            status="passed",
            expected_value="created",
            actual_value=str(workspace),
        )
        update_check(run_store_path, run_id, check)
        checks.append(check)

        settings = Settings.load(workspace)
        export_zip = options.data_dir / "chatgpt-export.zip"
        first = import_chatgpt_zip(export_zip, workspace, settings)
        second = import_chatgpt_zip(export_zip, workspace, settings)
        check = EvaluationCheck(
            check_name="import_idempotency",
            category="execution",
            status="passed" if first == second else "failed",
            expected_value=str(first),
            actual_value=str(second),
        )
        update_check(run_store_path, run_id, check)
        checks.append(check)

        indexed = build_indexes(workspace, settings, provider)
        check = EvaluationCheck(
            check_name="index_build",
            category="execution",
            status="passed" if indexed > 0 else "failed",
            expected_value=">0",
            actual_value=str(indexed),
        )
        update_check(run_store_path, run_id, check)
        checks.append(check)

        with connection(workspace) as conn:
            db_checks, database_counts = inspect_database(workspace, conn, manifest)
        for check in db_checks:
            update_check(run_store_path, run_id, check)
        checks.extend(db_checks)

        # --- Memory extraction ---
        from convsearch.memory.store import extract_and_store_memories

        with connection(workspace) as mem_conn:
            mem_summary = extract_and_store_memories(mem_conn)
        mem_extraction_check = EvaluationCheck(
            check_name="memory_extraction",
            category="memory",
            status="passed" if mem_summary.inserted > 0 else "failed",
            expected_value=">0",
            actual_value=str(mem_summary.inserted),
            details={
                "extracted": mem_summary.extracted,
                "inserted": mem_summary.inserted,
                "superseded": mem_summary.superseded,
                "contested": mem_summary.contested,
                "entities": mem_summary.entities,
            },
        )
        update_check(run_store_path, run_id, mem_extraction_check)
        checks.append(mem_extraction_check)

        # --- Memory accuracy checks ---
        memory_check_results, memory_accuracy = run_memory_checks(
            workspace, memory_cases, run_store_path, run_id
        )
        checks.extend(memory_check_results)

        # --- Segment recall checks ---
        segment_check_results, segment_recall = run_segment_checks(
            workspace, segment_cases, run_store_path, run_id
        )
        checks.extend(segment_check_results)

        query_results = run_queries(workspace, settings, provider, cases, run_store_path, run_id)
        metrics = calculate_metrics(
            query_results,
            segment_recall=segment_recall,
            memory_accuracy=memory_accuracy,
        )
        final_status: EvaluationStatus = (
            "passed"
            if all(c.status == "passed" for c in checks)
            and all(q.status == "passed" for q in query_results)
            else "failed"
        )
        finished = now()
        finalize_run(run_store_path, run_id, final_status, finished, metrics)
        report = EvaluationReport(
            run=run.model_copy(update={"status": final_status, "finished_at": finished}),
            fixture_version=manifest_int(manifest, "fixture_version"),
            run_dir=str(run_dir),
            workspace_dir=str(workspace),
            checks=checks,
            queries=query_results,
            metrics=metrics,
            database_counts=database_counts,
        )
        (run_dir / "report.json").write_text(report_to_json(report), encoding="utf-8")
        return report, final_status == "passed"
    except Exception as exc:
        metrics = EvaluationMetrics()
        finished = now()
        if run_store_path.exists():
            finalize_run(run_store_path, run_id, "error", finished, metrics, sanitize_error(exc))
        report = EvaluationReport(
            run=run.model_copy(
                update={
                    "status": "error",
                    "finished_at": finished,
                    "error_message": sanitize_error(exc),
                }
            ),
            fixture_version=manifest_int(manifest, "fixture_version", default=0),
            run_dir=str(run_dir),
            workspace_dir=str(workspace),
            checks=checks,
            queries=[],
            metrics=metrics,
        )
        (run_dir / "report.json").write_text(report_to_json(report), encoding="utf-8")
        return report, False
    finally:
        if not options.keep_run and run_dir.exists():
            remove_run_dir(run_dir)


MemoryCases = list[dict[str, object]]
SegmentCases = list[dict[str, object]]


def load_fixture(
    data_dir: Path,
) -> tuple[dict[str, object], list[EvaluationQueryCase], MemoryCases, SegmentCases]:
    manifest_path = data_dir / "manifest.json"
    searches_path = data_dir / "expected_searches.json"
    export_path = data_dir / "chatgpt-export.zip"
    missing = [
        str(path) for path in [manifest_path, searches_path, export_path] if not path.exists()
    ]
    if missing:
        raise ValueError(f"Missing synthetic fixture files: {', '.join(missing)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [
        EvaluationQueryCase.model_validate(item)
        for item in json.loads(searches_path.read_text(encoding="utf-8"))
    ]
    memories_path = data_dir / "expected_memories.json"
    memory_cases: list[dict[str, object]] = (
        json.loads(memories_path.read_text(encoding="utf-8")) if memories_path.exists() else []
    )
    segments_path = data_dir / "expected_segments.json"
    segment_cases: list[dict[str, object]] = (
        json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []
    )
    return manifest, cases, memory_cases, segment_cases


REQUIRED_FIXTURE_FILES = ["chatgpt-export.zip", "expected_searches.json", "manifest.json"]


def validate_fixture(data_dir: Path, manifest: dict[str, object]) -> list[EvaluationCheck]:
    checks = []
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return [
            EvaluationCheck(
                check_name="fixture_files",
                category="fixture",
                status="failed",
                details={"reason": "manifest files missing"},
            )
        ]
    missing = [name for name in REQUIRED_FIXTURE_FILES if not (data_dir / name).exists()]
    checks.append(
        EvaluationCheck(
            check_name="fixture_files",
            category="fixture",
            status="passed" if not missing else "failed",
            expected_value="all required files",
            actual_value=",".join(missing) if missing else "present",
        )
    )
    hash_failures = []
    for name, meta in files.items():
        if isinstance(meta, dict) and meta.get("sha256") and (data_dir / name).exists():
            actual = sha256_file(data_dir / name)
            if actual != meta["sha256"]:
                hash_failures.append(name)
    checks.append(
        EvaluationCheck(
            check_name="manifest_hashes",
            category="fixture",
            status="passed" if not hash_failures else "failed",
            expected_value="matching hashes",
            actual_value=",".join(hash_failures) if hash_failures else "matching",
        )
    )
    return checks


def initialize_workspace(workspace: Path) -> None:
    for child in ["database", "imports", "indexes", "cache", "logs"]:
        (workspace / child).mkdir(parents=True, exist_ok=True)
    Settings.default().write(workspace)
    initialize_database(workspace)
    if not database_path(workspace).exists():
        raise RuntimeError("workspace database not created")


def run_queries(
    workspace: Path,
    settings: Settings,
    provider: EmbeddingProvider,
    cases: list[EvaluationQueryCase],
    run_store_path: Path,
    run_id: str,
) -> list[EvaluationQueryResult]:
    results: list[EvaluationQueryResult] = []
    for case in cases:
        started = time.perf_counter()
        conversations = search_conversations(
            workspace,
            case.query,
            settings,
            provider,
            limit=10,
            profile=case.profile,
            show_passages=5,
            include_branches=case.include_branches,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        id_map = source_conversation_ids(
            workspace, [result.conversation_id for result in conversations]
        )
        returned_ids = [
            id_map.get(result.conversation_id, str(result.conversation_id))
            for result in conversations
        ]
        rank = first_expected_rank(returned_ids, case.expected_conversation_ids)
        expected_results = [
            result
            for result in conversations
            if id_map.get(result.conversation_id, str(result.conversation_id))
            in case.expected_conversation_ids
        ]
        passages = [hit for result in expected_results for hit in result.best_passages]
        passage_text = "\n".join(hit.text for hit in passages[:5])
        terms_passed = all(term.lower() in passage_text.lower() for term in case.expected_terms)
        scoped_returned_ids = returned_ids[: case.minimum_expected_rank]
        forbidden_result_violations = len(
            set(scoped_returned_ids) & set(case.forbidden_conversation_ids)
        )
        forbidden_term_violations = sum(
            1 for term in case.forbidden_terms if term.lower() in passage_text.lower()
        )
        branch_passed = branch_ok(case, passages)
        passed = (
            rank is not None
            and rank <= case.minimum_expected_rank
            and terms_passed
            and forbidden_result_violations == 0
            and forbidden_term_violations == 0
            and branch_passed
        )
        result = EvaluationQueryResult(
            case_id=case.case_id,
            query=case.query,
            status="passed" if passed else "failed",
            expected_conversation_ids=case.expected_conversation_ids,
            returned_conversation_ids=returned_ids,
            expected_rank=case.minimum_expected_rank,
            actual_rank=rank,
            reciprocal_rank=(1 / rank) if rank else 0.0,
            latency_ms=latency_ms,
            details={
                "terms_passed": terms_passed,
                "branch_passed": branch_passed,
                "forbidden_violations": forbidden_result_violations + forbidden_term_violations,
                "short_excerpt": passage_text[:240],
            },
        )
        update_query(run_store_path, run_id, result)
        update_check(
            run_store_path,
            run_id,
            EvaluationCheck(
                check_name=f"query:{case.case_id}",
                category="retrieval",
                status=result.status,
                expected_value="passed",
                actual_value=result.status,
                details=result.details,
            ),
        )
        results.append(result)
    return results


def run_memory_checks(
    workspace: Path,
    memory_cases: list[dict[str, object]],
    run_store_path: Path,
    run_id: str,
) -> tuple[list[EvaluationCheck], float]:
    """Run expected_memories.json assertion cases and return (checks, accuracy)."""
    if not memory_cases:
        return [], 1.0

    from convsearch.projects.reconstruct import reconstruct_project

    checks: list[EvaluationCheck] = []
    passed_count = 0

    with connection(workspace) as conn:
        for case in memory_cases:
            case_id = str(case.get("case_id", "unknown"))
            check_name = f"memory:{case_id}"

            # Project report case
            if "project_name" in case:
                project_name = str(case["project_name"])
                report = reconstruct_project(conn, project_name)
                _raw_egt = case.get("evidence_count_gt")
                evidence_gt = int(_raw_egt) if isinstance(_raw_egt, (int, float, str)) else 0
                decisions_nonempty = bool(case.get("decisions_nonempty", False))
                superseded_nonempty = bool(case.get("superseded_decisions_nonempty", False))
                if report is None:
                    check = EvaluationCheck(
                        check_name=check_name,
                        category="memory",
                        status="failed",
                        expected_value=f"report for {project_name!r}",
                        actual_value="None",
                    )
                else:
                    ok = (
                        report.evidence_count > evidence_gt
                        and (not decisions_nonempty or len(report.decisions) > 0)
                        and (not superseded_nonempty or len(report.superseded_decisions) > 0)
                    )
                    check = EvaluationCheck(
                        check_name=check_name,
                        category="memory",
                        status="passed" if ok else "failed",
                        expected_value=(
                            f"evidence>{evidence_gt},"
                            f"decisions={'nonempty' if decisions_nonempty else 'any'},"
                            f"superseded={'nonempty' if superseded_nonempty else 'any'}"
                        ),
                        actual_value=(
                            f"evidence={report.evidence_count},"
                            f"decisions={len(report.decisions)},"
                            f"superseded={len(report.superseded_decisions)}"
                        ),
                    )

            # Relation check
            elif "relation" in case:
                relation = str(case["relation"])
                from_contains = str(case.get("from_statement_contains", ""))
                to_contains = str(case.get("to_statement_contains", ""))
                rows = conn.execute(
                    """
                    SELECT mr.relation, mf.statement AS from_stmt, mt.statement AS to_stmt
                    FROM memory_relations mr
                    JOIN memories mf ON mf.memory_id = mr.from_memory_id
                    JOIN memories mt ON mt.memory_id = mr.to_memory_id
                    WHERE mr.relation = ?
                    """,
                    (relation,),
                ).fetchall()
                found = any(
                    from_contains.lower() in str(r["from_stmt"]).lower()
                    and to_contains.lower() in str(r["to_stmt"]).lower()
                    for r in rows
                )
                check = EvaluationCheck(
                    check_name=check_name,
                    category="memory",
                    status="passed" if found else "failed",
                    expected_value=(
                        f"relation={relation!r} from~{from_contains!r} to~{to_contains!r}"
                    ),
                    actual_value=f"found={found} ({len(rows)} total {relation!r} relations)",
                )

            # Rejected alternative check
            elif "rejected_alternative_contains" in case:
                project_name = str(case.get("project", ""))
                needle = str(case["rejected_alternative_contains"])
                rows = conn.execute(
                    """
                    SELECT metadata_json FROM memories
                    WHERE kind = 'decision'
                      AND (? = '' OR LOWER(project) = LOWER(?))
                    """,
                    (project_name, project_name),
                ).fetchall()
                found = any(
                    needle.lower()
                    in json.loads(str(r["metadata_json"] or "{}"))
                    .get("rejected_alternative", "")
                    .lower()
                    for r in rows
                )
                check = EvaluationCheck(
                    check_name=check_name,
                    category="memory",
                    status="passed" if found else "failed",
                    expected_value=f"rejected_alternative contains {needle!r}",
                    actual_value=f"found={found}",
                )

            # Standard memory check (kind/status/statement_contains/project)
            else:
                kind = str(case.get("kind", ""))
                status_val = str(case.get("status", ""))
                statement_contains = str(case.get("statement_contains", ""))
                project = str(case.get("project", ""))
                task_state = str(case.get("task_state", ""))

                rows = conn.execute(
                    """
                    SELECT memory_id, kind, status, statement, task_state, project
                    FROM memories
                    WHERE (? = '' OR kind = ?)
                      AND (? = '' OR status = ?)
                      AND (? = '' OR LOWER(project) = LOWER(?))
                      AND (? = '' OR task_state = ?)
                    """,
                    (kind, kind, status_val, status_val, project, project, task_state, task_state),
                ).fetchall()
                if statement_contains:
                    rows = [
                        r for r in rows if statement_contains.lower() in str(r["statement"]).lower()
                    ]
                found = len(rows) > 0
                check = EvaluationCheck(
                    check_name=check_name,
                    category="memory",
                    status="passed" if found else "failed",
                    expected_value=(
                        f"kind={kind!r} status={status_val!r}"
                        f" contains={statement_contains!r} project={project!r}"
                    ),
                    actual_value=f"found={found} (matched {len(rows)} rows)",
                )

            if check.status == "passed":
                passed_count += 1
            update_check(run_store_path, run_id, check)
            checks.append(check)

    accuracy = passed_count / len(memory_cases) if memory_cases else 1.0
    return checks, accuracy


def run_segment_checks(
    workspace: Path,
    segment_cases: list[dict[str, object]],
    run_store_path: Path,
    run_id: str,
) -> tuple[list[EvaluationCheck], float]:
    """Run expected_segments.json cases and return (checks, segment_recall).

    Uses a direct FTS query against segment_fts rather than search_segments()
    to avoid a SQLite limitation where bm25() cannot be used in certain complex
    JOIN contexts.  Results are ordered by segment_id (stable for the fixture).
    """
    if not segment_cases:
        return [], 1.0

    from convsearch.retrieval.query import build_fts_expressions, parse_query

    checks: list[EvaluationCheck] = []
    passed_count = 0

    with connection(workspace) as conn:
        for sc in segment_cases:
            case_id = str(sc.get("case_id", "unknown"))
            check_name = f"segment:{case_id}"
            query = str(sc.get("query", ""))
            _raw_ids = sc.get("expected_conversation_ids")
            expected_ids = [str(x) for x in _raw_ids] if isinstance(_raw_ids, list) else []
            _raw_rank = sc.get("minimum_expected_rank")
            min_rank = int(_raw_rank) if isinstance(_raw_rank, (int, float, str)) else 3
            limit = min_rank + 5

            parsed = parse_query(query)
            expressions = build_fts_expressions(parsed)
            returned_ids: list[str] = []
            for _level, fts_query in expressions:
                rows = conn.execute(
                    """
                    SELECT c.source_conversation_id
                    FROM segment_fts
                    JOIN segments s ON s.segment_id = segment_fts.rowid
                    JOIN conversations c ON c.conversation_id = s.conversation_id
                    WHERE segment_fts MATCH ?
                    GROUP BY c.conversation_id
                    ORDER BY s.segment_id
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
                for row in rows:
                    src_id = str(row["source_conversation_id"])
                    if src_id not in returned_ids:
                        returned_ids.append(src_id)
                if len(returned_ids) >= limit:
                    break

            rank = first_expected_rank(returned_ids, expected_ids)
            found = rank is not None and rank <= min_rank
            check = EvaluationCheck(
                check_name=check_name,
                category="segment",
                status="passed" if found else "failed",
                expected_value=f"rank<={min_rank} ids={expected_ids}",
                actual_value=f"rank={rank} returned={returned_ids[:min_rank]}",
            )
            if found:
                passed_count += 1
            update_check(run_store_path, run_id, check)
            checks.append(check)

    recall = passed_count / len(segment_cases) if segment_cases else 1.0
    return checks, recall


def first_expected_rank(returned: list[str], expected: list[str]) -> int | None:
    for index, conversation_id in enumerate(returned, start=1):
        if conversation_id in expected:
            return index
    return None


def source_conversation_ids(workspace: Path, conversation_ids: list[int]) -> dict[int, str]:
    if not conversation_ids:
        return {}
    placeholders = ",".join("?" for _ in conversation_ids)
    with connection(workspace) as conn:
        rows = conn.execute(
            f"""
            SELECT conversation_id, source_conversation_id
            FROM conversations
            WHERE conversation_id IN ({placeholders})
            """,
            conversation_ids,
        ).fetchall()
    return {int(row["conversation_id"]): str(row["source_conversation_id"]) for row in rows}


def branch_ok(case: EvaluationQueryCase, passages: Sequence[PassageHit]) -> bool:
    if case.expected_branch == "any":
        return True
    if case.expected_branch == "primary":
        return all(getattr(hit, "is_primary_path", False) for hit in passages[:5])
    if case.expected_branch == "alternate":
        return any(not getattr(hit, "is_primary_path", True) for hit in passages[:5])
    return True


def now() -> str:
    return datetime.now(UTC).isoformat()


def sanitize_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:500]


def default_run_root() -> Path:
    return Path(tempfile.gettempdir())


def manifest_int(manifest: dict[str, object], key: str, default: int | None = None) -> int:
    value = manifest.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    if default is not None:
        return default
    raise TypeError(f"Manifest value is not an integer: {key}")


def remove_run_dir(run_dir: Path) -> None:
    last_error: Exception | None = None
    for _ in range(10):
        try:
            gc.collect()
            shutil.rmtree(run_dir)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        tombstone = run_dir.with_name(f"removed-{run_dir.name}")
        try:
            run_dir.rename(tombstone)
        except PermissionError as exc:
            raise last_error from exc
