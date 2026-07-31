from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EvaluationStatus = Literal["pending", "running", "passed", "failed", "skipped", "error"]
ReportFormat = Literal["console", "json"]


class EvaluationQueryCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    query: str
    profile: Literal["balanced", "exact", "semantic"] = "balanced"
    include_branches: bool = False
    expected_conversation_ids: list[str]
    expected_terms: list[str] = Field(default_factory=list)
    forbidden_conversation_ids: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    minimum_expected_rank: int = Field(default=3, ge=1)
    expected_branch: Literal["primary", "alternate", "any"] = "any"


class EvaluationCheck(BaseModel):
    check_name: str
    category: str
    status: EvaluationStatus
    expected_value: str | None = None
    actual_value: str | None = None
    details: dict[str, object] = Field(default_factory=dict)


class EvaluationQueryResult(BaseModel):
    case_id: str
    query: str
    status: EvaluationStatus
    expected_conversation_ids: list[str]
    returned_conversation_ids: list[str] = Field(default_factory=list)
    expected_rank: int | None = None
    actual_rank: int | None = None
    reciprocal_rank: float = 0.0
    latency_ms: float | None = None
    details: dict[str, object] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0
    passage_recall_at_5: float = 0.0
    case_pass_rate: float = 0.0
    branch_filter_accuracy: float = 0.0
    forbidden_result_violations: int = 0
    segment_recall: float = 0.0
    memory_accuracy: float = 0.0


class EvaluationRun(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None = None
    status: EvaluationStatus
    is_ephemeral: bool
    data_directory: str
    data_manifest_hash: str
    embedding_provider: str
    error_message: str | None = None
    run_dir: str


class EvaluationReport(BaseModel):
    run: EvaluationRun
    fixture_version: int
    run_dir: str
    workspace_dir: str
    checks: list[EvaluationCheck]
    queries: list[EvaluationQueryResult]
    metrics: EvaluationMetrics
    database_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationOptions(BaseModel):
    data_dir: Path
    run_root: Path
    keep_run: bool = False
    real_model: bool = False
    report: ReportFormat = "console"
    fail_on_regression: bool = True
