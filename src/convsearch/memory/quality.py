from __future__ import annotations

import re

# Trailing punctuation/quote characters to strip before checking sentence-final shape.
# U+FFFD (replacement character) shows up when source text had an encoding mangling (seen in
# real workspace data wrapping quoted asides), and smart quotes are common in pasted prose.
_SMART_QUOTE_CODEPOINTS = (0x201C, 0x201D, 0x2018, 0x2019, 0xFFFD)
_TRAILING_CHARS = " \t\n\r\"'" + "".join(chr(c) for c in _SMART_QUOTE_CODEPOINTS)

_MIN_WORD_COUNT = 4

# A statement is table-row debris if it still carries the raw column separators from a
# markdown/plaintext table: tab characters, 2+ pipe characters, or a leading bare row number
# followed by whitespace (e.g. "3   Massive / Polygon-style   Good   ..."). Real prose never
# needs a literal tab or multiple pipes, so this cannot plausibly reject a real memory.
_LEADING_ROW_NUMBER_RE = re.compile(r"^\s*\d+\s{2,}\S")

# "do not need to" / "don't need to" and its variants assert that an action is NOT required --
# the opposite of an open task. Deliberately narrow (requires "need to") so it does not touch
# "do not deploy without running migrations", which asserts a real constraint on an action that
# IS still required.
_TASK_NEGATION_RE = re.compile(
    r"\b(?:do|does)\s+not\s+need\s+to\b"
    r"|\b(?:don|doesn)'?t\s+need\s+to\b"
    r"|\bno\s+longer\s+need\s+to\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def is_usable_statement(text: str, kind: str) -> tuple[bool, str | None]:
    """Reject-only precision filter for extracted memory candidates.

    Returns (True, None) if the statement should be kept as-is, or (False, reason) if it is
    confidently not a usable memory. This never rewrites or reclassifies a statement -- it only
    decides whether to drop it. When in doubt, keep it: a missed rejection is cheap, a wrongly
    rejected real memory is not.
    """
    stripped = text.strip()

    # Rule 1: trailing-colon fragments. A sentence that ends in ':' is introducing a list or
    # clause (e.g. "For your strategy, you need to know:") -- it is a lead-in, not itself a
    # standalone memory. A real memory is a complete statement and would not end this way.
    if stripped.rstrip(_TRAILING_CHARS).endswith(":"):
        return False, "trailing-colon fragment (introduces a list/clause, not a statement)"

    # Rule 2: table-row debris. Tabs, multiple pipes, or a leading bare row number are
    # artifacts of a markdown/plaintext table row surviving sentence splitting, not prose a
    # person would state as a memory.
    if "\t" in stripped or stripped.count("|") >= 2 or _LEADING_ROW_NUMBER_RE.match(stripped):
        return False, "table-row debris (looks like a mangled table row, not prose)"

    # Rule 3: minimum word count. Below this length a statement is reliably a label or
    # fragment ("constraints", "risk-free rate", "Account constraint") rather than a complete
    # thought. Kept deliberately low (4 words) so it never catches a real, if terse, statement.
    if _word_count(stripped) < _MIN_WORD_COUNT:
        return False, f"below minimum word count ({_MIN_WORD_COUNT} words)"

    if kind == "task":
        # Rule 4 (task-only): negation of need. "That means you do not need to compute delta
        # yourself" states a task is NOT required -- the opposite of an open task. Narrow
        # phrasing ("need to") keeps it from matching real constraints like "do not deploy
        # without X", which still require an action.
        if _TASK_NEGATION_RE.search(stripped):
            return False, "negates the need for action (opposite of an open task)"

        # Rule 5 (task-only): pure questions. A sentence that is only a question is an
        # open_question, not a task -- rejecting here (rather than reclassifying, which would
        # require changing `kind` in the caller/storage layer this module does not own) keeps
        # the fix inside the rejection-only contract.
        if stripped.rstrip(_TRAILING_CHARS).endswith("?"):
            return False, "purely a question, not an actionable task statement"

    return True, None
