"""Render a ProjectReport as a self-contained, evidence-cited Markdown document.

This module is pure: it takes a ProjectReport and returns a string. It performs no file
IO, no network calls, and no LLM calls. The CLI (or any other caller) is responsible for
writing the returned text to disk.

Every claim that comes from a decision, task, risk, or rejected alternative is annotated
with a citation marker (e.g. ``[E3]``) that resolves to an entry in the evidence appendix.
Items with no recorded evidence are marked ``_(no evidence recorded)_`` rather than being
presented as an unsupported fact. No quote, date, or reason is ever invented -- only what
is present on the report is rendered.
"""

from __future__ import annotations

from convsearch.projects.reconstruct import (
    EvidenceRef,
    ProjectItem,
    ProjectReport,
    SupersededBy,
)

_NO_EVIDENCE = "_(no evidence recorded)_"

# Characters that can alter Markdown structure if they appear unescaped in user text.
_MD_INLINE_ESCAPE = "\\`*_{}[]()#|"
_MD_LEADING_ESCAPE = ("-", ">", "!", "+")


def _escape_text(text: str) -> str:
    """Collapse whitespace and escape Markdown control characters in inline text."""
    collapsed = " ".join(text.split())
    escaped = collapsed.replace("\\", "\\\\")
    for ch in _MD_INLINE_ESCAPE[1:]:
        escaped = escaped.replace(ch, "\\" + ch)
    if escaped.startswith(_MD_LEADING_ESCAPE):
        escaped = "\\" + escaped
    return escaped


def _blockquote(text: str) -> str:
    """Render verbatim quoted text as a single-line Markdown blockquote."""
    return f"> {_escape_text(text)}"


class _CitationBook:
    """Accumulates evidence citations in encounter order and numbers them stably."""

    def __init__(self, max_per_item: int) -> None:
        self._entries: list[tuple[str, str | None, str | None]] = []
        self._max_per_item = max_per_item

    def _add(self, quote: str, conversation_title: str | None, timestamp: str | None) -> int:
        self._entries.append((quote, conversation_title, timestamp))
        return len(self._entries)

    def cite(self, evidence: tuple[EvidenceRef, ...], created_at: str | None) -> str:
        """Register evidence for a ProjectItem and return its citation markers."""
        if not evidence:
            return _NO_EVIDENCE
        capped = evidence[: self._max_per_item] if self._max_per_item > 0 else evidence
        numbers = [self._add(ev.quote, ev.conversation_title, created_at) for ev in capped]
        return "".join(f"[E{n}]" for n in numbers)

    def cite_dicts(self, evidence: object, created_at: str | None) -> str:
        """Register evidence for a JSON-native item dict (known_bugs/next_milestones)."""
        if not isinstance(evidence, list) or not evidence:
            return _NO_EVIDENCE
        capped = evidence[: self._max_per_item] if self._max_per_item > 0 else evidence
        numbers: list[int] = []
        for ev in capped:
            if not isinstance(ev, dict):
                continue
            quote = ev.get("quote")
            title = ev.get("conversation_title")
            if not isinstance(quote, str):
                continue
            conversation_title = title if isinstance(title, str) else None
            numbers.append(self._add(quote, conversation_title, created_at))
        if not numbers:
            return _NO_EVIDENCE
        return "".join(f"[E{n}]" for n in numbers)

    def render_appendix(self) -> list[str]:
        if not self._entries:
            return []
        lines = ["## Evidence appendix", ""]
        for i, (quote, title, timestamp) in enumerate(self._entries, start=1):
            conv = _escape_text(title) if title else "(untitled conversation)"
            when = f", {_escape_text(timestamp)}" if timestamp else ""
            lines.append(f"- **[E{i}]** {conv}{when}")
            lines.append(f"  {_blockquote(quote)}")
        return lines


def _item_dict_fields(
    item: dict[str, object],
) -> tuple[str, str | None, str | None, object]:
    """Pull (statement, status, created_at, evidence) out of a JSON-native item dict."""
    statement = item.get("statement")
    status = item.get("status")
    created_at = item.get("created_at")
    evidence = item.get("evidence")
    return (
        statement if isinstance(statement, str) else "",
        status if isinstance(status, str) else None,
        created_at if isinstance(created_at, str) else None,
        evidence,
    )


def _render_items(
    citations: _CitationBook,
    heading: str,
    items: tuple[ProjectItem, ...],
    *,
    bullet: str = "-",
) -> list[str]:
    if not items:
        return []
    lines = [f"## {heading}", ""]
    for item in items:
        marker = citations.cite(item.evidence, item.created_at)
        lines.append(f"{bullet} [{item.status}] {_escape_text(item.statement)} {marker}")
    lines.append("")
    return lines


def _render_dict_items(
    citations: _CitationBook,
    heading: str,
    items: tuple[dict[str, object], ...],
) -> list[str]:
    if not items:
        return []
    lines = [f"## {heading}", ""]
    for item in items:
        statement, status, created_at, evidence = _item_dict_fields(item)
        marker = citations.cite_dicts(evidence, created_at)
        status_tag = f"[{status}] " if status else ""
        lines.append(f"- {status_tag}{_escape_text(statement)} {marker}")
    lines.append("")
    return lines


def _render_supersession(sup: SupersededBy | None) -> str:
    """Render what replaced a superseded decision, and why -- never fabricating a reason.

    Three cases, matching the honesty rule the rest of this module follows:
    - replacement and reason both known -> state both.
    - replacement known, reason absent -> state the replacement, mark the reason unrecorded.
    - neither known -> the original honest fallback (the link itself was never recorded).
    """
    if sup is None:
        return "Replacement/reason: _(not recorded in this report)_"
    replacement = _escape_text(sup.statement)
    if sup.reason:
        return f"Replaced by: {replacement} -- reason: {_escape_text(sup.reason)}"
    return f"Replaced by: {replacement} -- reason not recorded in this report."


def _latest_timeline_date(report: ProjectReport) -> str:
    dated = [e.created_at for e in report.timeline if e.created_at]
    if not dated:
        return "unknown date"
    return max(dated)


def render_project_markdown(
    report: ProjectReport,
    *,
    include_evidence: bool = True,
    max_evidence_per_item: int = 2,
) -> str:
    """Render a ProjectReport as a single self-contained Markdown document.

    Sections are emitted in a fixed order and omitted entirely when they have no
    content. Every decision, task, risk, and rejected alternative carries a citation
    marker (e.g. ``[E1]``) resolved in the trailing evidence appendix, or an explicit
    ``_(no evidence recorded)_`` marker when the report has no supporting evidence for
    it. Output is deterministic: the same report always renders to the same string.

    ``max_evidence_per_item`` caps how many evidence citations are attached to a single
    item's line (extra evidence for that item is simply not cited, never dropped from
    the underlying report).
    """
    citations = _CitationBook(max_evidence_per_item)
    sections: list[list[str]] = []

    # 1. Title + provenance.
    as_of = _latest_timeline_date(report)
    sections.append(
        [
            f"# {_escape_text(report.name)}",
            "",
            f"*Generated on: {as_of}*",
            "",
            "_This document was reconstructed automatically from your own conversation "
            "history; nothing here was written by a human curator._",
            "",
        ]
    )

    # 2. Summary.
    if report.summary:
        sections.append(["## Summary", "", _escape_text(report.summary), ""])

    # 3. Architecture.
    sections.append(_render_items(citations, "Architecture", report.architecture))

    # 4. Active decisions.
    sections.append(_render_items(citations, "Active decisions", report.decisions))

    # 5. Superseded decisions.
    if report.superseded_decisions:
        lines = ["## Superseded decisions", ""]
        for item in report.superseded_decisions:
            marker = citations.cite(item.evidence, item.created_at)
            lines.append(f"- [{item.status}] {_escape_text(item.statement)} {marker}")
            lines.append(f"  {_render_supersession(item.superseded_by)}")
        lines.append("")
        sections.append(lines)

    # 6. Rejected alternatives (no structured evidence link is available for these).
    if report.rejected_alternatives:
        lines = ["## Rejected alternatives", ""]
        for alt in report.rejected_alternatives:
            lines.append(f"- {_escape_text(alt)} {_NO_EVIDENCE}")
        lines.append("")
        sections.append(lines)

    # 7. Open tasks.
    sections.append(_render_items(citations, "Open tasks", report.open_tasks, bullet="- [ ]"))

    # 8. Completed tasks.
    sections.append(
        _render_items(citations, "Completed tasks", report.completed_tasks, bullet="- [x]")
    )

    # 9. Risks.
    sections.append(_render_items(citations, "Risks", report.risks))

    # 10. Known bugs (additive field on ProjectReport).
    sections.append(_render_dict_items(citations, "Known bugs", report.known_bugs))

    # 11. Next milestones (additive field on ProjectReport).
    sections.append(_render_dict_items(citations, "Next milestones", report.next_milestones))

    # 12. Timeline.
    if report.timeline:
        lines = ["## Timeline", ""]
        for entry in report.timeline:
            date_str = entry.created_at or "unknown date"
            lines.append(
                f"- {date_str} [{entry.kind}/{entry.status}] {_escape_text(entry.statement)}"
            )
        lines.append("")
        sections.append(lines)

    # 13. Related conversations.
    if report.conversations:
        lines = ["## Related conversations", ""]
        for conv_id, title in report.conversations:
            name = _escape_text(title) if title else "(untitled)"
            lines.append(f"- [{conv_id}] {name}")
        lines.append("")
        sections.append(lines)

    # 14. Evidence appendix.
    if include_evidence:
        appendix = citations.render_appendix()
        if appendix:
            sections.append(appendix)

    non_empty = [section for section in sections if section]
    body = "\n".join(line for section in non_empty for line in section)
    return body.rstrip("\n") + "\n"
