from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from convsearch.config.settings import Settings
from convsearch.domain.models import ConversationResult
from convsearch.embeddings.sentence_transformers import EmbeddingProvider
from convsearch.llm.generate import GenerationResult, generate_text
from convsearch.retrieval.service import search_conversations

# Passages are trimmed to this many characters when embedded in the prompt so a
# single long message cannot dominate the context window.
_PROMPT_QUOTE_LIMIT = 600
# The full quote stored on an ``AnswerSource`` is allowed to be longer than the
# prompt copy so callers can display more context, but is still bounded.
_SOURCE_QUOTE_LIMIT = 1200

_NO_RESULTS_MESSAGE = "I couldn't find anything in your conversation history about that."

GenerateFn = Callable[..., GenerationResult]


@dataclass(frozen=True)
class AnswerSource:
    index: int
    conversation_id: int
    title: str
    date: str | None
    role: str
    quote: str


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    backend: str
    model: str
    sources: list[AnswerSource]


def _trim(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _collect_sources(results: list[ConversationResult], max_passages: int) -> list[AnswerSource]:
    sources: list[AnswerSource] = []
    for result in results:
        for passage in result.best_passages:
            if len(sources) >= max_passages:
                return sources
            date = passage.created_at[:10] if passage.created_at else None
            sources.append(
                AnswerSource(
                    index=len(sources) + 1,
                    conversation_id=passage.conversation_id,
                    title=passage.title,
                    date=date,
                    role=passage.role,
                    quote=_trim(passage.text, _SOURCE_QUOTE_LIMIT),
                )
            )
    return sources


def build_prompt(question: str, sources: list[AnswerSource]) -> tuple[str, str]:
    system = (
        "You answer questions about the USER'S OWN past ChatGPT conversations. "
        "Use ONLY the numbered context passages provided below to answer; do not "
        "rely on outside knowledge. Cite the passages you use inline as [1], [2], "
        "and so on. If the passages do not contain the answer, say so plainly "
        "instead of guessing. Be concise and specific."
    )
    blocks: list[str] = []
    for source in sources:
        header_bits = [source.title, source.date or "unknown date", source.role]
        header = " — ".join(header_bits)
        quote = _trim(source.quote, _PROMPT_QUOTE_LIMIT)
        blocks.append(f"[{source.index}] ({header})\n{quote}")
    context = "\n\n".join(blocks)
    user = f"Question: {question}\n\nContext passages:\n{context}"
    return system, user


def answer_question(
    workspace: Path,
    question: str,
    settings: Settings,
    provider: EmbeddingProvider,
    *,
    limit: int = 5,
    passages_per_conversation: int = 3,
    generate: GenerateFn | None = None,
) -> AnswerResult:
    generate_fn: GenerateFn = generate_text if generate is None else generate

    results = search_conversations(
        workspace,
        question,
        settings,
        provider,
        limit=limit,
        profile="balanced",
        show_passages=passages_per_conversation,
    )

    sources = _collect_sources(results, settings.llm.answer_max_passages)

    if not sources:
        return AnswerResult(
            question=question,
            answer=_NO_RESULTS_MESSAGE,
            backend="none",
            model="none",
            sources=[],
        )

    system, user = build_prompt(question, sources)
    result = generate_fn(
        system,
        user,
        settings=settings,
        max_tokens=settings.llm.answer_max_tokens,
    )
    return AnswerResult(
        question=question,
        answer=result.text.strip(),
        backend=result.backend,
        model=result.model,
        sources=sources,
    )
