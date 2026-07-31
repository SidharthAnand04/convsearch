from __future__ import annotations

import logging
import re

from convsearch.memory.models import ExtractedMemory
from convsearch.memory.quality import is_usable_statement
from convsearch.retrieval.query import IDENTIFIER_RE, STOPWORDS

logger = logging.getLogger(__name__)

IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\:-]+")
TERM_RE = re.compile(r"[A-Za-z0-9]+")

# sentence splitter - splits on . ! ? followed by whitespace/end, or newlines
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")

# Pattern rules: (kind, priority, list of trigger phrases)
_RULES: list[tuple[str, int, list[str]]] = [
    (
        "decision",
        1,
        [
            "we decided",
            "decided to",
            "decision:",
            "let's go with",
            "we'll use",
            "we will use",
            "going with",
            "we chose",
            "switching to",
            "switched to",
            "switched from",
            "settled on",
            "i'll go with",
            "i would go with",
            "we're going with",
            "we are going with",
            "let's use",
            "i would use",
            "i'll use",
        ],
    ),
    (
        "task",
        2,
        [
            "todo",
            "need to",
            "needs to",
            "next step",
            "action item",
            "we should",
            "remaining work",
            "marked as done",
            "completed the",
            "finished the",
            "is now done",
            "implemented the",
            "fixed the",
        ],
    ),
    (
        "preference",
        3,
        [
            "i prefer",
            "prefer to",
            "always use",
            "never use",
            "my preference",
        ],
    ),
    (
        "risk",
        4,
        [
            "risk",
            "concern is",
            "concerned that",
            "worried",
            "might break",
            "could fail",
        ],
    ),
    (
        "project_state",
        5,
        [
            "we are building",
            "the architecture",
            "the stack is",
            "current state",
            "the project uses",
        ],
    ),
    (
        "constraint",
        6,
        [
            "must not",
            "cannot",
            "constraint",
            "limited to",
            "required to",
            "has to",
            "budget is",
            "no more than",
            "at most",
        ],
    ),
    (
        "open_question",
        7,
        [
            "unclear",
            "not sure whether",
            "open question",
            "still deciding",
            "tbd",
            "need to decide",
        ],
    ),
]

_TASK_COMPLETED_TRIGGERS: set[str] = {
    "marked as done",
    "completed the",
    "finished the",
    "is now done",
    "implemented the",
    "fixed the",
}
_TASK_OPEN_TRIGGERS: set[str] = {
    "todo",
    "need to",
    "needs to",
    "next step",
    "action item",
    "we should",
    "remaining work",
}

_PROJECT_RE = re.compile(
    r"(?:for|on|in) the ([A-Za-z0-9 _-]{2,40}) project|project ([A-Za-z0-9_-]{2,40})",
    re.IGNORECASE,
)
_ALT_RE = re.compile(
    r"(?:instead of|over|rather than)\s+([A-Za-z0-9_./\\:-]+)",
    re.IGNORECASE,
)

# A bare label line ("Decision:", "Open question:", ...) is common in structured
# ChatGPT answers where the label and its statement are split across a line break by
# sentence segmentation (e.g. "Decision:\nUse SQLite for storage."). The label sentence
# itself is rejected by quality.py as a trailing-colon fragment, so without this the
# statement on the *next* sentence is never even considered a candidate -- it carries no
# trigger phrase of its own. This maps a recognized label word to the kind that should be
# attributed to the immediately following sentence, one sentence of lookahead only.
_LABEL_KIND_NAMES: dict[str, str] = {
    "decision": "decision",
    "task": "task",
    "preference": "preference",
    "risk": "risk",
    "constraint": "constraint",
    "open question": "open_question",
    "project state": "project_state",
}
_TRAILING_LABEL_PUNCT_RE = re.compile(r"[:\s]+$")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into (sentence, start, end) tuples using sentence boundaries."""
    results: list[tuple[str, int, int]] = []
    pos = 0
    for m in _SENT_BOUNDARY.finditer(text):
        boundary_start = m.start()
        sentence = text[pos:boundary_start].rstrip()
        if sentence:
            results.append((sentence, pos, pos + len(sentence)))
        pos = m.end()
    # remainder
    remainder = text[pos:].rstrip()
    if remainder:
        results.append((remainder, pos, pos + len(remainder)))
    return results


def _subject_key(sentence: str) -> str:
    """Derive a subject key from a sentence."""
    tokens: list[str] = IDENTIFIER_TOKEN_RE.findall(sentence)
    for token in tokens:
        if IDENTIFIER_RE.match(token):
            return token.lower()
    # fallback: first 3 non-stopword alphanumeric terms
    term_tokens = TERM_RE.findall(sentence)
    filtered = [t for t in term_tokens if t.lower() not in STOPWORDS][:3]
    if filtered:
        return "-".join(t.lower() for t in filtered)
    return "general"


def _detect_project(sentence: str, default_project: str | None) -> str | None:
    """Extract project name from sentence or return default."""
    m = _PROJECT_RE.search(sentence)
    if m:
        return (m.group(1) or m.group(2) or "").strip() or default_project
    return default_project


def extract_from_message(
    text: str,
    *,
    conversation_id: int,
    message_id: int,
    created_at: str | None = None,
    default_project: str | None = None,
    reject_counts: dict[str, int] | None = None,
) -> list[ExtractedMemory]:
    """Extract structured memories from a message text.

    ``reject_counts``, if given, is incremented in place with one entry per rejection
    reason (as returned by ``is_usable_statement``) so a caller can report how much the
    precision filter dropped without re-running extraction. Optional and additive: omit it
    to get the old silent-debug-only behavior.
    """
    results: list[ExtractedMemory] = []
    sentences = split_sentences(text)

    pending_label_kind: str | None = None

    for sentence, start, end in sentences:
        sentence_lower = sentence.lower()

        matched_kind: str | None = None
        matched_trigger: str | None = None
        best_priority = 999

        for kind, priority, triggers in _RULES:
            if priority >= best_priority:
                continue
            for trigger in triggers:
                if trigger in sentence_lower:
                    matched_kind = kind
                    matched_trigger = trigger
                    best_priority = priority
                    break

        from_label = False
        if matched_kind is None and pending_label_kind is not None:
            matched_kind = pending_label_kind
            matched_trigger = f"{pending_label_kind} label"
            from_label = True

        # Recompute for the *next* sentence based on this one, regardless of whether this
        # sentence matched anything above -- a label carries forward exactly one sentence.
        label_key = _TRAILING_LABEL_PUNCT_RE.sub("", sentence.strip()).lower()
        pending_label_kind = _LABEL_KIND_NAMES.get(label_key)

        if matched_kind is None or matched_trigger is None:
            continue

        usable, reject_reason = is_usable_statement(sentence.strip(), matched_kind)
        if not usable:
            # Debug-level so a human can inspect individual rejections (e.g. via -v
            # logging); reject_counts (if the caller wants it) carries the aggregate so a
            # summary can report "extracted N, filtered M" without re-running extraction.
            logger.debug(
                "rejected %s candidate (%s): %r", matched_kind, reject_reason, sentence.strip()
            )
            if reject_counts is not None and reject_reason is not None:
                reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
            continue

        # Determine task_state
        task_state: str | None = None
        if matched_kind == "task":
            if matched_trigger in _TASK_COMPLETED_TRIGGERS:
                task_state = "completed"
            elif matched_trigger in _TASK_OPEN_TRIGGERS:
                task_state = "open"

        subject = _subject_key(sentence)
        project = _detect_project(sentence, default_project)

        # Determine confidence
        sentence_stripped = sentence.strip()
        starts_with_trigger = sentence_stripped.lower().startswith(matched_trigger)
        has_identifier = any(
            IDENTIFIER_RE.match(tok) for tok in IDENTIFIER_TOKEN_RE.findall(sentence)
        )
        confidence = 0.9 if (from_label or starts_with_trigger or has_identifier) else 0.7

        # Entities: all IDENTIFIER_RE-matching tokens (deduped, original casing)
        tokens = IDENTIFIER_TOKEN_RE.findall(sentence)
        seen_entities: dict[str, str] = {}
        for tok in tokens:
            if IDENTIFIER_RE.match(tok):
                key = tok.lower()
                if key not in seen_entities:
                    seen_entities[key] = tok
        entities = tuple(seen_entities.values())

        # Metadata: rejected_alternative for decisions
        metadata: dict[str, str] = {}
        if matched_kind == "decision":
            alt_m = _ALT_RE.search(sentence)
            if alt_m:
                metadata["rejected_alternative"] = alt_m.group(1)

        results.append(
            ExtractedMemory(
                kind=matched_kind,
                subject_key=subject,
                statement=sentence_stripped,
                confidence=confidence,
                project=project,
                task_state=task_state,
                conversation_id=conversation_id,
                message_id=message_id,
                created_at=created_at,
                quote=sentence,
                start_offset=start,
                end_offset=end,
                entities=entities,
                metadata=metadata,
            )
        )

    return results
