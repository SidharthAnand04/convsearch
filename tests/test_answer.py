"""Tests for the RAG answer feature.

Run with::

    uv run pytest tests/test_answer.py -q

These tests are deterministic and require no network, LLM, or model download:
retrieval is monkeypatched and generation is injected as a plain function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from convsearch.answer import answer as answer_mod
from convsearch.answer.answer import (
    AnswerResult,
    AnswerSource,
    answer_question,
    build_prompt,
)
from convsearch.config.settings import Settings
from convsearch.domain.models import ConversationResult, PassageHit
from convsearch.llm.generate import GenerationResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_conv_result(
    conv_id: int = 3,
    *,
    title: str = "Setting up FAISS",
    text: str = "You installed faiss-cpu and built an IndexFlatIP.",
    created_at: str | None = "2024-01-02T10:30:00",
    role: str = "assistant",
) -> ConversationResult:
    passage = PassageHit(
        passage_id=100,
        conversation_id=conv_id,
        message_id=10,
        title=title,
        role=role,
        text=text,
        created_at=created_at,
        is_primary_path=True,
        fused_score=0.8,
    )
    return ConversationResult(
        conversation_id=conv_id,
        title=title,
        created_at=created_at,
        updated_at=created_at,
        score=0.8,
        best_passages=[passage],
        distinct_message_count=1,
    )


# ---------------------------------------------------------------------------
# build_prompt — pure, no I/O
# ---------------------------------------------------------------------------


def test_build_prompt_includes_question_citations_titles_and_instruction() -> None:
    sources = [
        AnswerSource(
            index=1,
            conversation_id=3,
            title="Setting up FAISS",
            date="2024-01-02",
            role="assistant",
            quote="You installed faiss-cpu and built an IndexFlatIP.",
        ),
        AnswerSource(
            index=2,
            conversation_id=7,
            title="Choosing an embedding model",
            date="2024-02-11",
            role="user",
            quote="We went with BAAI/bge-small-en-v1.5.",
        ),
    ]
    question = "How did I set up FAISS?"

    system, user = build_prompt(question, sources)

    # The user's question is carried into the prompt.
    assert question in user
    # Citation markers for each source are present.
    assert "[1]" in user
    assert "[2]" in user
    # Titles of the retrieved conversations appear in the context.
    assert "Setting up FAISS" in user
    assert "Choosing an embedding model" in user
    # The system prompt instructs the model to cite its sources.
    assert "cite" in system.lower()


# ---------------------------------------------------------------------------
# answer_question — monkeypatched retrieval + injected generate
# ---------------------------------------------------------------------------


def test_answer_question_empty_results_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_mod, "search_conversations", lambda *a, **kw: [])

    def _fail_generate(*a: Any, **kw: Any) -> GenerationResult:  # pragma: no cover
        raise AssertionError("generate should not be called when there are no sources")

    result = answer_question(
        Path("/tmp/fake-workspace"),
        "anything at all?",
        Settings(),
        object(),  # provider is never used because retrieval is stubbed
        generate=_fail_generate,
    )

    assert isinstance(result, AnswerResult)
    assert result.backend == "none"
    assert result.model == "none"
    assert result.sources == []
    assert result.answer  # a sensible non-empty message


def test_answer_question_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    conv = _make_conv_result(
        conv_id=3,
        title="Setting up FAISS",
        text="You installed faiss-cpu and built an IndexFlatIP.",
        created_at="2024-01-02T10:30:00",
    )
    monkeypatch.setattr(answer_mod, "search_conversations", lambda *a, **kw: [conv])

    captured: dict[str, Any] = {}

    def _fake_generate(system: str, user: str, **kw: Any) -> GenerationResult:
        captured["system"] = system
        captured["user"] = user
        return GenerationResult(text="stub answer [1]", backend="ollama", model="llama3.2:1b")

    result = answer_question(
        Path("/tmp/fake-workspace"),
        "how did I set up FAISS?",
        Settings(),
        object(),
        limit=5,
        passages_per_conversation=3,
        generate=_fake_generate,
    )

    assert result.answer == "stub answer [1]"
    assert result.backend == "ollama"
    assert result.model == "llama3.2:1b"
    assert len(result.sources) == 1

    source = result.sources[0]
    assert source.title == "Setting up FAISS"
    assert source.date == "2024-01-02"  # created_at truncated to the date
    assert source.conversation_id == 3
    assert source.role == "assistant"

    # The prompt actually saw the question and the citation marker.
    assert "how did I set up FAISS?" in captured["user"]
    assert "[1]" in captured["user"]
