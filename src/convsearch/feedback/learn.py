"""Self-improvement job: turn logged interactions into learned preference notes.

Reads the interaction log, builds a compact digest of the user's search behaviour,
and asks the local LLM (Ollama-first via ``generate_text``) to summarise it into a
few short, durable preference notes. When the LLM is unavailable it falls back to a
deterministic summary derived from the most popular queries. Notes are persisted to
the ``learned_preferences`` table so later ranking/suggestions can bias toward them.

Stdlib + sqlite3 + datetime(UTC) only, plus the stated cross-module contracts.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from convsearch.config.settings import Settings
from convsearch.feedback.store import (
    interaction_stats,
    popular_queries,
    recent_queries,
)
from convsearch.llm.generate import GenerationResult, generate_text

GenerateFn = Callable[..., GenerationResult]

_SYSTEM_PROMPT = (
    "You summarize a user's local search behavior into a few short, durable "
    "preference notes that could bias future ranking/suggestions. Output one note "
    "per line, max {max_notes}, each a concise imperative like 'Prioritize "
    "conversations about FAISS/SQLite storage decisions.' Use ONLY the provided "
    "data; no preamble."
)

# Descending weights by rank; ranks beyond this list clamp to the final value.
_RANK_WEIGHTS = (1.0, 0.9, 0.8, 0.7, 0.6)


@dataclass(frozen=True)
class LearnSummary:
    events_read: int
    notes_written: int
    backend: str  # 'ollama'|'anthropic'|'none'
    model: str
    notes: list[str]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _weight_for_rank(rank: int) -> float:
    """Weight for a 0-based note rank, clamped to the smallest configured weight."""
    if rank < len(_RANK_WEIGHTS):
        return _RANK_WEIGHTS[rank]
    return _RANK_WEIGHTS[-1]


def _build_digest(
    stats: dict[str, int],
    popular: list[tuple[str, int]],
    recent: list[str],
) -> str:
    """Compact, LLM-friendly description of the user's search behaviour."""
    lines: list[str] = []
    lines.append(
        "Interaction counts: "
        f"total={stats.get('total', 0)}, search={stats.get('search', 0)}, "
        f"open={stats.get('open', 0)}, inspect={stats.get('inspect', 0)}, "
        f"ask={stats.get('ask', 0)}, distinct_queries={stats.get('distinct_queries', 0)}."
    )
    if popular:
        lines.append("Top queries (by frequency):")
        for query, count in popular:
            lines.append(f'  - "{query}" (seen {count}x)')
    if recent:
        lines.append("Recent focus (most recent first):")
        for query in recent:
            lines.append(f'  - "{query}"')
    return "\n".join(lines)


def _parse_notes(text: str, max_notes: int) -> list[str]:
    """Up to ``max_notes`` non-empty, stripped lines from an LLM reply."""
    notes: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        notes.append(line)
        if len(notes) >= max_notes:
            break
    return notes


def _heuristic_notes(popular: list[tuple[str, int]], max_notes: int) -> list[str]:
    """Deterministic fallback notes from the most popular queries."""
    notes: list[str] = []
    for query, count in popular[:max_notes]:
        notes.append(f'Prioritize results for recurring query: "{query}" (seen {count}x).')
    return notes


def _persist_notes(
    conn: sqlite3.Connection,
    notes: list[str],
    source: str,
) -> None:
    created_at = _now_iso()
    for rank, note in enumerate(notes):
        conn.execute(
            "INSERT INTO learned_preferences(created_at, note, weight, source) VALUES (?, ?, ?, ?)",
            (created_at, note, _weight_for_rank(rank), source),
        )
    conn.commit()


def run_self_improvement(
    conn: sqlite3.Connection,
    settings: Settings,
    *,
    use_llm: bool = True,
    generate: GenerateFn | None = None,
    max_notes: int = 5,
) -> LearnSummary:
    """Summarise logged interactions into learned preference notes.

    Reads interaction stats and query history, then either asks the local LLM to
    distil durable preference notes or (when the LLM is unavailable, or ``use_llm``
    is False) falls back to a deterministic summary of popular queries. Persists the
    resulting notes to ``learned_preferences`` and returns a :class:`LearnSummary`.
    """
    generate_fn: GenerateFn = generate if generate is not None else generate_text

    stats = interaction_stats(conn)
    if stats.get("total", 0) == 0:
        # No signal to learn from: never touch the LLM, never write notes.
        return LearnSummary(0, 0, "none", "none", [])

    popular = popular_queries(conn, 20)
    recent = recent_queries(conn, 20)
    digest = _build_digest(stats, popular, recent)

    notes: list[str]
    backend: str
    model: str
    source: str

    if use_llm:
        system = _SYSTEM_PROMPT.format(max_notes=max_notes)
        try:
            result = generate_fn(system, digest, settings=settings)
            notes = _parse_notes(result.text, max_notes)
            backend = result.backend or "heuristic"
            model = result.model or "none"
            source = "llm"
        except Exception:  # LLMUnavailableError or any failure -> deterministic fallback
            notes = _heuristic_notes(popular, max_notes)
            backend = "none"
            model = "none"
            source = "heuristic"
    else:
        notes = _heuristic_notes(popular, max_notes)
        backend = "none"
        model = "none"
        source = "heuristic"

    if not notes:
        # LLM returned nothing usable; still leave a deterministic footprint.
        notes = _heuristic_notes(popular, max_notes)
        backend = "none"
        model = "none"
        source = "heuristic"

    _persist_notes(conn, notes, source)

    return LearnSummary(
        events_read=stats["total"],
        notes_written=len(notes),
        backend=backend,
        model=model,
        notes=notes,
    )


def list_learned_preferences(
    conn: sqlite3.Connection, limit: int = 20
) -> list[tuple[int, str, float, str]]:
    """Return ``(pref_id, note, weight, created_at)`` rows, newest first."""
    rows = conn.execute(
        "SELECT pref_id, note, weight, created_at FROM learned_preferences "
        "ORDER BY pref_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [(int(row[0]), str(row[1]), float(row[2]), str(row[3])) for row in rows]
