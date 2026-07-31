from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass

from convsearch.config.settings import Settings
from convsearch.llm.generate import LLMUnavailableError, generate_text
from convsearch.memory.extract import _detect_project, _subject_key
from convsearch.memory.models import MEMORY_KINDS, ExtractedMemory
from convsearch.memory.quality import is_usable_statement
from convsearch.retrieval.query import IDENTIFIER_RE
from convsearch.utils import memory_effective_timestamp_sql

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\:-]+")

# LLM-derived memories are given a fixed, deliberately-lower-than-rules confidence: the
# rules path's confidence reflects trigger-phrase/identifier heuristics that do not apply
# here, and a model proposal has no equivalent signal to distinguish 0.7 from 0.9.
_LLM_CONFIDENCE = 0.6

_KIND_LIST = ", ".join(MEMORY_KINDS)

_SYSTEM_PROMPT = f"""You extract structured memories from a conversation between a user and an \
assistant. A memory is a single, self-contained statement worth remembering later: a decision, \
an open or completed task, a stated preference, a description of project/system state, a risk, \
a constraint, or an open question.

Valid kinds: {_KIND_LIST}

Respond with ONLY a JSON array (no prose, no markdown fences). Each element must be an object \
with exactly these keys:
  "kind": one of the valid kinds above
  "statement": a short, self-contained restatement of the memory
  "quote": a VERBATIM substring copied character-for-character from the conversation text below \
that supports this memory. Do not paraphrase, summarize, or alter the quote in any way -- copy \
it exactly, including punctuation.
  "project": the project or topic name this relates to, or null if none is evident

If there are no memories worth extracting, respond with an empty JSON array: []
Do not invent statements that are not grounded in the text. Do not include labels, headings, or \
table rows as quotes."""


@dataclass(frozen=True)
class LlmProposalResult:
    """Result of an LLM-assisted extraction pass over one or more conversations.

    ``accepted`` are proposals that survived JSON parsing, verbatim-quote verification, the
    existing precision filter (``is_usable_statement``), and dedup against memories already
    present in ``memories`` -- the same reject-only discipline the rules path uses. Every other
    field is a count of *why* a proposal did not make it, so a caller can report a discard rate
    instead of a single opaque number.
    """

    accepted: tuple[ExtractedMemory, ...]
    proposed: int
    discarded_malformed: int
    discarded_not_verbatim: int
    discarded_precision_filter: int
    discarded_duplicate: int
    backend: str
    model: str
    conversations_processed: int
    llm_calls: int
    elapsed_seconds: float


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _fetch_messages(
    conn: sqlite3.Connection, conversation_ids: tuple[int, ...] | None
) -> list[sqlite3.Row]:
    ts = memory_effective_timestamp_sql("m", "c")
    where_extra = ""
    params: tuple[object, ...] = ()
    if conversation_ids:
        placeholders = ", ".join("?" for _ in conversation_ids)
        where_extra = f" AND m.conversation_id IN ({placeholders})"
        params = tuple(conversation_ids)
    rows = conn.execute(
        f"""
        SELECT m.message_id, m.conversation_id, m.text,
               {ts} AS created_at,
               c.title AS conversation_title
        FROM messages m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        WHERE m.is_primary_path = 1 AND m.text != ''{where_extra}
        ORDER BY m.conversation_id, m.source_order
        """,
        params,
    ).fetchall()
    return list(rows)


def _existing_statements(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Normalized (kind -> statement) pairs already stored, for dedup."""
    existing: dict[str, set[str]] = {}
    try:
        rows = conn.execute("SELECT kind, statement FROM memories").fetchall()
    except sqlite3.OperationalError:
        return existing
    for row in rows:
        existing.setdefault(row["kind"], set()).add(_normalize(row["statement"]))
    return existing


def _build_prompt(rows: list[sqlite3.Row]) -> str:
    parts = ["Conversation messages, in order:\n"]
    for row in rows:
        parts.append(f"--- message ---\n{row['text']}\n")
    return "\n".join(parts)


def _parse_response(raw: str) -> list[dict[str, object]] | None:
    """Strictly parse the model's JSON array. Any deviation is a discard, not a guess."""
    text = raw.strip()
    # Some local models wrap output in markdown fences despite instructions -- strip those,
    # but do not attempt any other repair (e.g. trailing-comma fixing) since that would be
    # papering over exactly the kind of unreliable output this function must reject.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    return data


@dataclass(frozen=True)
class _ResolvedQuote:
    conversation_id: int
    message_id: int
    created_at: str | None
    conversation_title: str | None
    start_offset: int
    end_offset: int
    quote: str


def _resolve_quote(quote: str, rows: list[sqlite3.Row]) -> _ResolvedQuote | None:
    """Find which message's real text contains ``quote`` as a verbatim (whitespace-normalized)

    substring. Returns the resolved location for the first match, or None if no message's
    actual text contains it. The model's own claim about which message it came from is never
    trusted -- this always re-derives the answer from the real ``messages.text`` rows passed in.
    """
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return None
    candidate = quote.strip()
    collapsed_candidate = _WHITESPACE_RE.sub(" ", candidate)
    for row in rows:
        text = row["text"]
        if normalized_quote not in _normalize(text):
            continue
        # Locate a real character span in the original text. Prefer an exact literal match;
        # whitespace normalization can shift offsets, so fall back to a whitespace-collapsed
        # search (still a verbatim match modulo whitespace, never a fuzzy/near match).
        idx = text.find(candidate)
        if idx != -1:
            start, end = idx, idx + len(candidate)
            quote_out = candidate
        else:
            # Whitespace-collapse and case-fold both sides for the fallback search (still a
            # verbatim match modulo whitespace/case, never a fuzzy/near match), then slice the
            # *collapsed* text (not the candidate) so the returned quote reflects real casing.
            collapsed_text = _WHITESPACE_RE.sub(" ", text)
            idx = collapsed_text.lower().find(collapsed_candidate.lower())
            if idx == -1:
                continue
            start, end = idx, idx + len(collapsed_candidate)
            quote_out = collapsed_text[start:end]
        return _ResolvedQuote(
            conversation_id=row["conversation_id"],
            message_id=row["message_id"],
            created_at=row["created_at"],
            conversation_title=row["conversation_title"],
            start_offset=start,
            end_offset=end,
            quote=quote_out,
        )
    return None


def propose_memories(
    conn: sqlite3.Connection,
    *,
    settings: Settings,
    conversation_ids: list[int] | tuple[int, ...] | None = None,
    backend: str = "auto",
) -> LlmProposalResult:
    """Propose additional memories using an LLM, as an opt-in augmentation of the rules path.

    Groups primary-path messages by conversation and asks the model for structured candidate
    memories once per conversation. Every survivor must pass, in order:
      1. strict JSON parsing (malformed output is discarded, never guessed at)
      2. verbatim-quote verification against the real ``messages.text`` this function fetched
         itself (never the model's own claim about where the quote came from)
      3. the existing rules-path precision filter, ``quality.is_usable_statement``
      4. dedup against memories already present in ``conn`` (or already proposed in this call)

    Returns an ``LlmProposalResult`` with the accepted proposals plus a count for each discard
    reason. Raises nothing on LLM unavailability -- if the backend cannot be reached, this
    returns a result with zero accepted proposals and the counts reflecting no processing, so
    the rules path is entirely unaffected by calling this.
    """
    start_time = time.monotonic()
    conv_ids_tuple = tuple(conversation_ids) if conversation_ids else None
    rows = _fetch_messages(conn, conv_ids_tuple)

    by_conversation: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_conversation.setdefault(row["conversation_id"], []).append(row)

    existing = _existing_statements(conn)
    seen_this_call: set[tuple[str, str]] = set()

    accepted: list[ExtractedMemory] = []
    proposed_count = 0
    discarded_malformed = 0
    discarded_not_verbatim = 0
    discarded_precision_filter = 0
    discarded_duplicate = 0
    llm_calls = 0
    backend_used = backend
    model_used = ""

    for conversation_id, conv_rows in by_conversation.items():
        prompt = _build_prompt(conv_rows)
        try:
            result = generate_text(
                _SYSTEM_PROMPT,
                prompt,
                settings=settings,
                max_tokens=1024,
            )
        except LLMUnavailableError:
            logger.debug("LLM unavailable for conversation %s; skipping", conversation_id)
            continue
        llm_calls += 1
        backend_used = result.backend
        model_used = result.model

        items = _parse_response(result.text)
        if items is None:
            discarded_malformed += 1
            continue

        for item in items:
            proposed_count += 1
            if not isinstance(item, dict):
                discarded_malformed += 1
                continue
            kind = item.get("kind")
            statement = item.get("statement")
            quote = item.get("quote")
            project_hint = item.get("project")
            if (
                not isinstance(kind, str)
                or kind not in MEMORY_KINDS
                or not isinstance(statement, str)
                or not statement.strip()
                or not isinstance(quote, str)
                or not quote.strip()
            ):
                discarded_malformed += 1
                continue
            if project_hint is not None and not isinstance(project_hint, str):
                discarded_malformed += 1
                continue

            resolved = _resolve_quote(quote, conv_rows)
            if resolved is None:
                discarded_not_verbatim += 1
                continue

            usable, _reason = is_usable_statement(statement.strip(), kind)
            if not usable:
                discarded_precision_filter += 1
                continue

            norm_statement = _normalize(statement)
            dedup_key = (kind, norm_statement)
            if norm_statement in existing.get(kind, set()) or dedup_key in seen_this_call:
                discarded_duplicate += 1
                continue
            seen_this_call.add(dedup_key)

            subject = _subject_key(statement)
            project = _detect_project(statement, project_hint or resolved.conversation_title)

            tokens = IDENTIFIER_TOKEN_RE.findall(statement)
            seen_entities: dict[str, str] = {}
            for tok in tokens:
                if IDENTIFIER_RE.match(tok):
                    key = tok.lower()
                    if key not in seen_entities:
                        seen_entities[key] = tok
            entities = tuple(seen_entities.values())

            accepted.append(
                ExtractedMemory(
                    kind=kind,
                    subject_key=subject,
                    statement=statement.strip(),
                    confidence=_LLM_CONFIDENCE,
                    project=project,
                    task_state=None,
                    conversation_id=resolved.conversation_id,
                    message_id=resolved.message_id,
                    created_at=resolved.created_at,
                    quote=resolved.quote,
                    start_offset=resolved.start_offset,
                    end_offset=resolved.end_offset,
                    entities=entities,
                    metadata={},
                )
            )

    elapsed = time.monotonic() - start_time
    return LlmProposalResult(
        accepted=tuple(accepted),
        proposed=proposed_count,
        discarded_malformed=discarded_malformed,
        discarded_not_verbatim=discarded_not_verbatim,
        discarded_precision_filter=discarded_precision_filter,
        discarded_duplicate=discarded_duplicate,
        backend=backend_used,
        model=model_used,
        conversations_processed=len(by_conversation),
        llm_calls=llm_calls,
        elapsed_seconds=elapsed,
    )
