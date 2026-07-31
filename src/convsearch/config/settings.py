from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SearchWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lexical: float = 1.0
    semantic: float = 1.0


class AggregationWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_passage: float = 0.65
    mean_top_three: float = 0.30
    distinct_message_bonus: float = 0.05


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_min_score: dict[str, float] = Field(
        default_factory=lambda: {"balanced": 0.20, "exact": 0.25, "semantic": 0.15}
    )
    title_weight: float = 0.6
    lexical_fallback_min_results: int = 5

    def semantic_floor(self, profile: str) -> float:
        return self.semantic_min_score.get(profile, self.semantic_min_score["balanced"])


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = "claude-haiku-4-5"
    max_expansion_terms: int = Field(default=6, ge=0)
    failure_policy: str = Field(default="skip", pattern="^(error|skip)$")
    # Answer backend selection. "auto" tries a local Ollama model first and falls
    # back to the cloud (Anthropic) model; "ollama" and "anthropic" force one path.
    backend: str = Field(default="auto", pattern="^(auto|ollama|anthropic)$")
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"
    answer_max_passages: int = Field(default=8, ge=1)
    answer_max_tokens: int = Field(default=1024, ge=64)


class RerankingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: str = Field(default="cross_encoder", pattern="^(cross_encoder|llm)$")
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    llm_model: str = "claude-haiku-4-5"
    device: str = "auto"
    batch_size: int = Field(default=16, ge=1)
    candidate_limit: int = Field(default=50, ge=1)
    failure_policy: str = Field(default="error", pattern="^(error|skip)$")
    weight: float = 1.0
    rrf_k: int = Field(default=60, ge=1)


class SegmentationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    strategy: str = Field(default="rules", pattern="^(rules|semantic|hybrid)$")
    minimum_segment_messages: int = Field(default=2, ge=1)
    maximum_segment_messages: int = Field(default=8, ge=1)
    time_gap_minutes: int = Field(default=240, ge=1)
    semantic_shift_threshold: float = 0.55
    merge_similarity_threshold: float = 0.80


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)
    passage_target_words: int = Field(default=180, ge=20)
    passage_overlap_words: int = Field(default=30, ge=0)
    lexical_candidate_limit: int = Field(default=50, ge=1)
    semantic_candidate_limit: int = Field(default=50, ge=1)
    final_result_limit: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    balanced_weights: SearchWeights = SearchWeights()
    exact_weights: SearchWeights = SearchWeights(lexical=1.5, semantic=0.5)
    semantic_weights: SearchWeights = SearchWeights(lexical=0.5, semantic=1.5)
    aggregation_weights: AggregationWeights = AggregationWeights()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()
    reranking: RerankingSettings = RerankingSettings()
    segmentation: SegmentationSettings = SegmentationSettings()

    @classmethod
    def default(cls) -> Settings:
        return cls()

    @classmethod
    def load(cls, workspace: Path) -> Settings:
        config_path = workspace / "config.yaml"
        if not config_path.exists():
            return cls.default()
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if "semantic_min_score" in data and "retrieval" not in data:
            data["retrieval"] = {"semantic_min_score": data.pop("semantic_min_score")}
        return cls.model_validate(data)

    def write(self, workspace: Path) -> None:
        data: dict[str, Any] = self.model_dump(mode="json")
        with (workspace / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)

    def profile_weights(self, profile: str) -> SearchWeights:
        if profile == "balanced":
            return self.balanced_weights
        if profile == "exact":
            return self.exact_weights
        if profile == "semantic":
            return self.semantic_weights
        raise ValueError(f"Unknown search profile: {profile}")


def database_path(workspace: Path) -> Path:
    return workspace / "database" / "convsearch.sqlite3"


def faiss_index_path(workspace: Path) -> Path:
    return workspace / "indexes" / "passages.faiss"


def vector_map_path(workspace: Path) -> Path:
    return workspace / "indexes" / "passage_vectors.json"
