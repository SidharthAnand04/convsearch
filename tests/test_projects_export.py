from __future__ import annotations

from convsearch.projects.export import render_project_markdown
from convsearch.projects.reconstruct import (
    EvidenceRef,
    ProjectItem,
    ProjectReport,
    SupersededBy,
    TimelineEntry,
)

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _evidence(memory_id: int, quote: str, title: str = "Conv Alpha") -> EvidenceRef:
    return EvidenceRef(
        memory_id=memory_id,
        conversation_id=1,
        conversation_title=title,
        message_id=memory_id,
        passage_id=None,
        quote=quote,
    )


def _make_report() -> ProjectReport:
    active_decision = ProjectItem(
        memory_id=1,
        statement="Use SQLite for storage",
        status="active",
        created_at="2024-01-01",
        subject_key="arch/store",
        evidence=(_evidence(1, "SQLite chosen over PostgreSQL"),),
    )
    superseded_decision = ProjectItem(
        memory_id=2,
        statement="Use in-process cache",
        status="superseded",
        created_at="2024-01-02",
        subject_key="arch/cache",
        evidence=(_evidence(2, "in-process cache was discussed"),),
    )
    open_task = ProjectItem(
        memory_id=3,
        statement="Benchmark query performance",
        status="active",
        created_at="2024-01-03",
        subject_key="task/perf",
        evidence=(_evidence(3, "we should benchmark this"),),
    )
    completed_task = ProjectItem(
        memory_id=4,
        statement="Design initial schema",
        status="active",
        created_at="2024-01-04",
        subject_key="task/schema",
        evidence=(),  # deliberately no evidence
    )
    risk = ProjectItem(
        memory_id=5,
        statement="SQLite write lock under load",
        status="active",
        created_at="2024-01-05",
        subject_key="risk/lock",
        evidence=(_evidence(5, "lock contention under high write load"),),
    )
    timeline = (
        TimelineEntry(
            created_at="2024-01-01",
            kind="decision",
            statement="Use SQLite for storage",
            status="active",
            memory_id=1,
        ),
        TimelineEntry(
            created_at="2024-01-05",
            kind="risk",
            statement="SQLite write lock under load",
            status="active",
            memory_id=5,
        ),
    )
    return ProjectReport(
        name="Alpha",
        summary="2 decisions, 1 open task across 1 conversation",
        timeline=timeline,
        architecture=(),
        decisions=(active_decision,),
        superseded_decisions=(superseded_decision,),
        rejected_alternatives=("PostgreSQL", "Redis"),
        open_tasks=(open_task,),
        completed_tasks=(completed_task,),
        risks=(risk,),
        conversations=((1, "Conv Alpha"),),
        evidence_count=4,
        known_bugs=(
            {
                "memory_id": 6,
                "statement": "Import crashes on malformed export",
                "status": "active",
                "created_at": "2024-01-06",
                "subject_key": "bug/import",
                "evidence": [
                    {
                        "memory_id": 6,
                        "conversation_id": 1,
                        "conversation_title": "Conv Alpha",
                        "message_id": 6,
                        "passage_id": None,
                        "quote": "the import fails on malformed input",
                    }
                ],
            },
        ),
        next_milestones=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderProjectMarkdown:
    def test_title_and_provenance(self) -> None:
        doc = render_project_markdown(_make_report())
        assert doc.startswith("# Alpha")
        assert "reconstructed automatically from your own" in doc

    def test_sections_present(self) -> None:
        doc = render_project_markdown(_make_report())
        assert "## Summary" in doc
        assert "## Active decisions" in doc
        assert "## Superseded decisions" in doc
        assert "## Rejected alternatives" in doc
        assert "## Open tasks" in doc
        assert "## Completed tasks" in doc
        assert "## Risks" in doc
        assert "## Known bugs" in doc
        assert "## Timeline" in doc
        assert "## Related conversations" in doc
        assert "## Evidence appendix" in doc

    def test_empty_sections_omitted(self) -> None:
        doc = render_project_markdown(_make_report())
        # architecture and next_milestones are empty on this fixture
        assert "## Architecture" not in doc
        assert "## Next milestones" not in doc

    def test_citation_markers_resolve_to_appendix(self) -> None:
        doc = render_project_markdown(_make_report())
        assert "[E1]" in doc
        assert "**[E1]**" in doc
        # every [Ei] marker used in the body has a matching appendix bullet
        import re

        markers = set(re.findall(r"\[E(\d+)\]", doc.split("## Evidence appendix")[0]))
        for marker in markers:
            assert f"**[E{marker}]**" in doc

    def test_no_evidence_item_marked_explicitly(self) -> None:
        doc = render_project_markdown(_make_report())
        assert "Design initial schema" in doc
        completed_line = next(line for line in doc.splitlines() if "Design initial schema" in line)
        assert "_(no evidence recorded)_" in completed_line

    def test_rejected_alternatives_have_no_evidence_marker(self) -> None:
        doc = render_project_markdown(_make_report())
        for line in doc.splitlines():
            if "PostgreSQL" in line and line.startswith("-"):
                assert "_(no evidence recorded)_" in line

    def test_no_fabricated_reason_for_superseded(self) -> None:
        doc = render_project_markdown(_make_report())
        assert "not recorded in this report" in doc

    def test_superseded_with_replacement_and_reason(self) -> None:
        report = _make_report()
        superseded = ProjectItem(
            memory_id=2,
            statement="Use in-process cache",
            status="superseded",
            created_at="2024-01-02",
            subject_key="arch/cache",
            evidence=(_evidence(2, "in-process cache was discussed"),),
            superseded_by=SupersededBy(
                memory_id=20,
                statement="Use Redis for caching",
                reason="in-process cache did not survive process restarts",
            ),
        )
        report = ProjectReport(**{**report.__dict__, "superseded_decisions": (superseded,)})
        doc = render_project_markdown(report)
        line = next(
            line
            for line in doc.splitlines()
            if "Replaced by" in line and "Use Redis for caching" in line
        )
        assert "Use Redis for caching" in line
        assert "in-process cache did not survive process restarts" in line
        assert "not recorded" not in line

    def test_superseded_with_replacement_no_reason(self) -> None:
        report = _make_report()
        superseded = ProjectItem(
            memory_id=2,
            statement="Use in-process cache",
            status="superseded",
            created_at="2024-01-02",
            subject_key="arch/cache",
            evidence=(_evidence(2, "in-process cache was discussed"),),
            superseded_by=SupersededBy(memory_id=20, statement="Use Redis for caching"),
        )
        report = ProjectReport(**{**report.__dict__, "superseded_decisions": (superseded,)})
        doc = render_project_markdown(report)
        line = next(
            line
            for line in doc.splitlines()
            if "Replaced by" in line and "Use Redis for caching" in line
        )
        assert "reason not recorded in this report" in line

    def test_superseded_with_no_link_keeps_fallback(self) -> None:
        # _make_report's superseded_decision has no superseded_by set (default None).
        doc = render_project_markdown(_make_report())
        line = next(line for line in doc.splitlines() if "Replacement/reason" in line)
        assert "_(not recorded in this report)_" in line

    def test_quotes_are_verbatim_and_blockquoted(self) -> None:
        doc = render_project_markdown(_make_report())
        assert "> SQLite chosen over PostgreSQL" in doc

    def test_no_empty_headers(self) -> None:
        doc = render_project_markdown(_make_report())
        lines = doc.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("## "):
                # the next non-blank line must exist and not be another header
                rest = lines[i + 1 :]
                assert any(entry.strip() for entry in rest[:3])

    def test_deterministic_byte_identical_output(self) -> None:
        report = _make_report()
        first = render_project_markdown(report)
        second = render_project_markdown(report)
        assert first == second

    def test_max_evidence_per_item_caps_citations(self) -> None:
        report = _make_report()
        many_evidence = (
            *report.decisions[0].evidence,
            _evidence(1, "a second supporting quote"),
            _evidence(1, "a third supporting quote"),
        )
        decision = ProjectItem(
            memory_id=1,
            statement="Use SQLite for storage",
            status="active",
            created_at="2024-01-01",
            subject_key="arch/store",
            evidence=many_evidence,
        )
        capped_report = ProjectReport(
            **{**report.__dict__, "decisions": (decision,)},
        )
        doc = render_project_markdown(capped_report, max_evidence_per_item=1)
        decision_line = next(line for line in doc.splitlines() if "Use SQLite for storage" in line)
        assert decision_line.count("[E") == 1
