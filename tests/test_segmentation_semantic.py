from __future__ import annotations

from pathlib import Path

import pytest

from convsearch.config.settings import SegmentationSettings, Settings
from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.segmentation.build import rebuild_segments
from convsearch.segmentation.hybrid import HybridSegmentationProvider
from convsearch.segmentation.models import SegmentableMessage
from convsearch.segmentation.rules import RuleBasedSegmentationProvider
from convsearch.segmentation.semantic import SemanticShiftSegmentationProvider

TOPIC_A = "kubernetes cluster deploy pods"
TOPIC_B = "sourdough bread baking starter"


def make_messages(specs: list[tuple[str, str]]) -> list[SegmentableMessage]:
    return [
        SegmentableMessage(
            message_id=index + 1,
            conversation_id=1,
            source_order=index,
            role=role,
            text=text,
            created_at="2026-01-01T00:00:00+00:00",
            is_primary_path=True,
        )
        for index, (role, text) in enumerate(specs)
    ]


def segmentation_settings() -> SegmentationSettings:
    return SegmentationSettings(strategy="semantic")


def test_semantic_provider_splits_at_topic_shift() -> None:
    messages = make_messages(
        [
            ("user", TOPIC_A),
            ("assistant", TOPIC_A),
            ("user", TOPIC_A),
            ("assistant", TOPIC_A),
            ("user", TOPIC_B),
            ("assistant", TOPIC_B),
        ]
    )
    provider = SemanticShiftSegmentationProvider(
        segmentation_settings(), DeterministicEmbeddingProvider()
    )
    segments = provider.segment(messages)
    assert len(segments) == 2
    assert segments[0].message_ids == (1, 2, 3, 4)
    assert segments[1].message_ids == (5, 6)
    assert "semantic_shift" in segments[1].reasons
    assert any(reason.startswith("similarity=") for reason in segments[1].reasons)
    assert 0.0 < segments[1].boundary_confidence <= 1.0
    assert provider.version == "semantic-shift-v1"


def test_semantic_provider_never_splits_question_answer_pair() -> None:
    messages = make_messages(
        [
            ("user", TOPIC_A),
            ("assistant", TOPIC_A),
            ("user", TOPIC_A),
            ("assistant", TOPIC_B),
        ]
    )
    provider = SemanticShiftSegmentationProvider(
        segmentation_settings(), DeterministicEmbeddingProvider()
    )
    segments = provider.segment(messages)
    for segment in segments:
        assert 3 in segment.message_ids or 4 not in segment.message_ids
    matching = [segment for segment in segments if 3 in segment.message_ids]
    assert len(matching) == 1
    assert 4 in matching[0].message_ids


def test_hybrid_merges_highly_similar_adjacent_segments() -> None:
    messages = make_messages(
        [
            ("user", TOPIC_A),
            ("assistant", TOPIC_A),
            ("user", f"another question about {TOPIC_A}"),
            ("assistant", TOPIC_A),
        ]
    )
    settings = SegmentationSettings(strategy="hybrid")
    embedding_provider = DeterministicEmbeddingProvider()
    rules_segments = RuleBasedSegmentationProvider(settings).segment(messages)
    assert len(rules_segments) == 2
    hybrid = HybridSegmentationProvider(settings, embedding_provider)
    segments = hybrid.segment(messages)
    assert len(segments) == 1
    assert segments[0].message_ids == (1, 2, 3, 4)
    assert hybrid.version == "hybrid-v1"


def test_hybrid_keeps_dissimilar_segments_split() -> None:
    messages = make_messages(
        [
            ("user", TOPIC_A),
            ("assistant", TOPIC_A),
            ("user", TOPIC_B),
            ("assistant", TOPIC_B),
        ]
    )
    settings = SegmentationSettings(strategy="hybrid")
    hybrid = HybridSegmentationProvider(settings, DeterministicEmbeddingProvider())
    segments = hybrid.segment(messages)
    assert len(segments) == 2
    assert segments[0].message_ids == (1, 2)
    assert segments[1].message_ids == (3, 4)


def test_rebuild_segments_semantic_without_provider_raises(
    workspace: Path, settings: Settings
) -> None:
    settings.segmentation.strategy = "semantic"
    with pytest.raises(RuntimeError, match="embedding provider"):
        rebuild_segments(workspace, settings)


def test_rebuild_segments_semantic_with_provider(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    settings.segmentation.strategy = "semantic"
    count = rebuild_segments(workspace, settings, provider=DeterministicEmbeddingProvider())
    assert count >= 1
