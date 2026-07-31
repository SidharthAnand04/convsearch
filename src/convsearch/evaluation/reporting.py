from __future__ import annotations

import json

from rich.console import Console

from convsearch.evaluation.models import EvaluationReport


def report_to_json(report: EvaluationReport) -> str:
    return report.model_dump_json(indent=2)


def print_report(report: EvaluationReport, *, as_json: bool = False) -> None:
    console = Console(width=120)
    if as_json:
        console.print(json.dumps(report.model_dump(mode="json"), indent=2), markup=False)
        return
    status = report.run.status.upper()
    console.print(f"Synthetic evaluation: {status}")
    console.print("")
    console.print("Run")
    console.print(f"- ID: {report.run.run_id}")
    console.print(f"- Fixture version: {report.fixture_version}")
    console.print(f"- Embedding provider: {report.run.embedding_provider}")
    console.print(f"- Ephemeral or kept: {'kept' if not report.run.is_ephemeral else 'ephemeral'}")
    console.print(f"- Directory: {report.run_dir}")
    console.print("")
    console.print("Database")
    for key in [
        "conversations",
        "messages",
        "passages",
        "primary_messages",
        "alternate_messages",
        "fts_records",
        "embedding_records",
    ]:
        console.print(f"- {key}: {report.database_counts.get(key, 0)}")
    console.print("")
    console.print("Retrieval")
    console.print(f"- Recall@1: {report.metrics.recall_at_1:.3f}")
    console.print(f"- Recall@3: {report.metrics.recall_at_3:.3f}")
    console.print(f"- Recall@10: {report.metrics.recall_at_10:.3f}")
    console.print(f"- MRR: {report.metrics.mean_reciprocal_rank:.3f}")
    console.print(f"- Passage Recall@5: {report.metrics.passage_recall_at_5:.3f}")
    console.print(f"- Segment Recall: {report.metrics.segment_recall:.3f}")
    console.print(f"- Memory Accuracy: {report.metrics.memory_accuracy:.3f}")
    console.print(
        f"- Passed cases: {sum(q.status == 'passed' for q in report.queries)}/{len(report.queries)}"
    )
    failures = [check for check in report.checks if check.status not in {"passed", "skipped"}] + [
        query for query in report.queries if query.status != "passed"
    ]
    if failures:
        console.print("")
        console.print("Failures")
        for failure in failures:
            name = getattr(failure, "check_name", getattr(failure, "case_id", "unknown"))
            expected = getattr(failure, "expected_value", None) or getattr(
                failure, "expected_conversation_ids", None
            )
            actual = getattr(failure, "actual_value", None) or getattr(
                failure, "returned_conversation_ids", None
            )
            console.print(f"- {name}: expected={expected} actual={actual}")
