from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class MessagesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ExpandedQuery:
    raw: str
    rewritten_query: str
    expansion_terms: tuple[str, ...]

    @property
    def search_text(self) -> str:
        parts = [self.rewritten_query or self.raw, *self.expansion_terms]
        return " ".join(dict.fromkeys(part for part in parts if part))


_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_query": {
            "type": "string",
            "description": "The query rewritten as concise search keywords.",
        },
        "expansion_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Synonyms or related terms likely to appear in matching text.",
        },
    },
    "required": ["rewritten_query", "expansion_terms"],
    "additionalProperties": False,
}

_SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {"type": "number"},
            "description": "One relevance score from 0 to 10 per passage, in input order.",
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def make_messages_client() -> MessagesClient:
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError(
            "The anthropic package is not installed. Run `uv sync --extra llm --group dev`."
        ) from exc
    client: Any = anthropic.Anthropic()
    messages: MessagesClient = client.messages
    return messages


def _structured_response(client: MessagesClient, **kwargs: Any) -> dict[str, Any]:
    response = client.create(**kwargs)
    text = next(block.text for block in response.content if block.type == "text")
    result: dict[str, Any] = json.loads(text)
    return result


def expand_query(
    query: str,
    model: str,
    *,
    max_terms: int,
    client: MessagesClient | None = None,
) -> ExpandedQuery:
    messages = client if client is not None else make_messages_client()
    data = _structured_response(
        messages,
        model=model,
        max_tokens=512,
        system=(
            "You turn a user's natural-language question about their chat history into "
            "search keywords. Keep identifiers, file names, and error strings verbatim. "
            f"Return at most {max_terms} expansion terms; fewer is fine."
        ),
        messages=[{"role": "user", "content": query}],
        output_config={"format": {"type": "json_schema", "schema": _EXPANSION_SCHEMA}},
    )
    terms = tuple(str(term) for term in data.get("expansion_terms", [])[:max_terms])
    return ExpandedQuery(
        raw=query,
        rewritten_query=str(data.get("rewritten_query", "")).strip(),
        expansion_terms=terms,
    )


class LLMReranker:
    def __init__(self, model: str, client: MessagesClient | None = None) -> None:
        self.model_id = model
        self._client = client if client is not None else make_messages_client()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        numbered = "\n\n".join(f"[{index}] {passage}" for index, passage in enumerate(passages))
        data = _structured_response(
            self._client,
            model=self.model_id,
            max_tokens=2048,
            system=(
                "You are a search reranker. Score how relevant each numbered passage is "
                "to the query on a 0-10 scale (10 = directly answers it, 0 = unrelated). "
                "Return exactly one score per passage, in the same order."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nPassages ({len(passages)}):\n\n{numbered}",
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": _SCORES_SCHEMA}},
        )
        scores = [float(score) for score in data["scores"]]
        if len(scores) < len(passages):
            scores.extend([0.0] * (len(passages) - len(scores)))
        return scores[: len(passages)]
