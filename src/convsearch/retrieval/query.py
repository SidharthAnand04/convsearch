from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

IDENTIFIER_RE = re.compile(
    r"^(?:[A-Za-z0-9_./\\:-]*[._/\\:-][A-Za-z0-9_./\\:-]+|[A-Z0-9_]{2,}|[A-Za-z]+[A-Z][A-Za-z0-9]*)$"
)
TERM_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\\:-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "did",
    "do",
    "does",
    "for",
    "i",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "where",
}


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    phrases: tuple[str, ...]
    required_terms: tuple[str, ...]
    optional_terms: tuple[str, ...]
    excluded_terms: tuple[str, ...]
    identifiers: tuple[str, ...]


def parse_query(query: str) -> ParsedQuery:
    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()
    phrases: list[str] = []
    required_terms: list[str] = []
    optional_terms: list[str] = []
    excluded_terms: list[str] = []
    identifiers: list[str] = []
    for token in tokens:
        if not token:
            continue
        is_excluded = token.startswith("-") and len(token) > 1
        clean = token[1:] if is_excluded else token
        terms = TERM_RE.findall(clean)
        if not terms:
            continue
        value = " ".join(terms) if " " in clean else terms[0]
        if is_excluded:
            excluded_terms.append(value)
            continue
        if " " in clean:
            phrases.append(value)
        elif IDENTIFIER_RE.match(value):
            identifiers.append(value)
            required_terms.append(value)
        elif value.lower() in STOPWORDS:
            continue
        else:
            optional_terms.append(value)
    return ParsedQuery(
        raw=query,
        phrases=tuple(dict.fromkeys(phrases)),
        required_terms=tuple(dict.fromkeys(required_terms)),
        optional_terms=tuple(dict.fromkeys(optional_terms)),
        excluded_terms=tuple(dict.fromkeys(excluded_terms)),
        identifiers=tuple(dict.fromkeys(identifiers)),
    )


def fts_quote(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def build_fts_expressions(parsed: ParsedQuery) -> list[tuple[str, str]]:
    positive = [*parsed.phrases, *parsed.required_terms]
    optional = list(parsed.optional_terms)
    excluded = [f"NOT {fts_quote(term)}" for term in parsed.excluded_terms]
    suffix = " ".join(excluded)
    expressions: list[tuple[str, str]] = []
    if positive:
        expressions.append(("strict", " AND ".join(fts_quote(term) for term in positive)))
    if positive or optional:
        all_terms = [*positive, *optional]
        expressions.append(("and", " AND ".join(fts_quote(term) for term in all_terms)))
    if optional:
        expressions.append(("or", " OR ".join(fts_quote(term) for term in [*positive, *optional])))
    if not expressions:
        expressions.append(("or", fts_quote(parsed.raw)))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for level, expression in expressions:
        final = f"{expression} {suffix}".strip()
        if final not in seen:
            deduped.append((level, final))
            seen.add(final)
    return deduped
