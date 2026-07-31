from __future__ import annotations

from convsearch.evaluation.models import EvaluationMetrics, EvaluationQueryResult


def calculate_metrics(
    results: list[EvaluationQueryResult],
    *,
    segment_recall: float = 0.0,
    memory_accuracy: float = 0.0,
) -> EvaluationMetrics:
    total = len(results)
    if total == 0:
        return EvaluationMetrics(segment_recall=segment_recall, memory_accuracy=memory_accuracy)
    recall_1 = (
        sum(1 for result in results if result.actual_rank and result.actual_rank <= 1) / total
    )
    recall_3 = (
        sum(1 for result in results if result.actual_rank and result.actual_rank <= 3) / total
    )
    recall_10 = (
        sum(1 for result in results if result.actual_rank and result.actual_rank <= 10) / total
    )
    mrr = sum(result.reciprocal_rank for result in results) / total
    passed = sum(1 for result in results if result.status == "passed")
    passage_hits = sum(1 for result in results if bool(result.details.get("terms_passed")))
    branch_cases = [result for result in results if "branch_passed" in result.details]
    branch_ok = sum(1 for result in branch_cases if result.details.get("branch_passed"))
    forbidden = sum(detail_int(result.details.get("forbidden_violations", 0)) for result in results)
    return EvaluationMetrics(
        recall_at_1=recall_1,
        recall_at_3=recall_3,
        recall_at_10=recall_10,
        mean_reciprocal_rank=mrr,
        passage_recall_at_5=passage_hits / total,
        case_pass_rate=passed / total,
        branch_filter_accuracy=(branch_ok / len(branch_cases)) if branch_cases else 1.0,
        forbidden_result_violations=forbidden,
        segment_recall=segment_recall,
        memory_accuracy=memory_accuracy,
    )


def detail_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0
