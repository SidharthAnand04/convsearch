from __future__ import annotations

from convsearch.domain.models import ConversationResult, PassageHit

# Human-readable phrase for each retrieval channel.
_CHANNEL_PHRASES: dict[str, str] = {
    "lexical": "exact keyword match",
    "semantic": "semantic match",
    "title": "title match",
}


def _channel_phrase(channel: str) -> str:
    """Map a retrieval channel name to a human-readable phrase."""
    return _CHANNEL_PHRASES.get(channel, channel)


def passage_explain(hit: PassageHit) -> dict[str, object]:
    """Flat, JSON-serializable scoring breakdown for one passage."""
    return {
        "lexical_score": hit.lexical_score,
        "semantic_score": hit.semantic_score,
        "title_score": hit.title_score,
        "reranker_score": hit.reranker_score,
        "fused_score": hit.fused_score,
        "final_score": hit.final_score,
        "ranks": {
            "lexical": hit.lexical_rank,
            "semantic": hit.semantic_rank,
            "title": hit.title_rank,
            "reranker": hit.reranker_rank,
        },
        "channels": list(hit.channels),
        "branch": "selected" if hit.is_primary_path else "alternate",
    }


def _has_title_signal(result: ConversationResult) -> bool:
    """True if any passage matched on the title channel or carries a title score."""
    for hit in result.best_passages:
        if "title" in hit.channels or hit.title_score is not None:
            return True
    return False


def build_reason(result: ConversationResult) -> str:
    """One human-readable sentence: why this conversation ranked, from its data.

    Composed from the top passage's channels + available scores + branch. Only
    derives from the given data; never fabricates.
    """
    if not result.best_passages:
        return "Ranked by overall relevance."

    top = result.best_passages[0]

    # Collect channel phrases in a stable order, from the top passage's channels.
    phrases: list[str] = []
    seen: set[str] = set()
    for channel in top.channels:
        # Mention title match only when there is a real title signal.
        if channel == "title" and not _has_title_signal(result):
            continue
        phrase = _channel_phrase(channel)
        if phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    reason = "Ranked by " + _join_phrases(phrases) if phrases else "Ranked by overall relevance"

    branch_clause = (
        "found on the selected conversation path"
        if top.is_primary_path
        else "found on an alternate conversation branch"
    )
    reason += f"; {branch_clause}."

    if len(result.best_passages) > 1:
        reason += f" Appears in {len(result.best_passages)} passages."

    return reason


def _join_phrases(phrases: list[str]) -> str:
    """Join phrases into a natural-language list ('a', 'a and b', 'a, b, and c')."""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
