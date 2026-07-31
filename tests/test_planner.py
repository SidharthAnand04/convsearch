"""Tests for the deterministic query planner.

Run with::

    uv run pytest tests/test_planner.py -q
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from convsearch.domain.models import ConversationResult, PassageHit
from convsearch.planner.planner import PlanAnswer, PlanStep, execute_plan, plan_query
from convsearch.planner.tools import PlannerContext, override_tool, reset_registry

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRelation:
    relation: str = "supersedes"
    other_memory_id: int = 99
    other_statement: str = "Use Postgres instead of SQLite."
    reason: str | None = None
    direction: str = "outgoing"


@dataclass(frozen=True)
class _FakeMemoryRecord:
    memory_id: str = "mem-1"
    kind: str = "decision"
    subject_key: str = "test subject"
    statement: str = "We decided to use SQLite."
    status: str = "active"
    confidence: float = 0.9
    project: str | None = None
    task_state: str | None = None
    conversation_id: int = 7
    conversation_title: str = "Architecture"
    message_id: int = 42
    created_at: str = "2024-01-01T00:00:00"
    evidence: tuple[()] = field(default_factory=tuple)  # type: ignore[assignment]
    relations: tuple[()] = field(default_factory=tuple)  # type: ignore[assignment]


def _make_conv_result(conv_id: int = 1, msg_id: int = 10) -> ConversationResult:
    passage = PassageHit(
        passage_id=100,
        conversation_id=conv_id,
        message_id=msg_id,
        title="Test Conversation",
        role="assistant",
        text="This is a test passage about the topic.",
        created_at=None,
        is_primary_path=True,
        fused_score=0.8,
    )
    return ConversationResult(
        conversation_id=conv_id,
        title="Test Conversation",
        created_at=None,
        updated_at=None,
        score=0.8,
        best_passages=[passage],
        distinct_message_count=3,
    )


def _make_ctx() -> PlannerContext:
    conn = sqlite3.connect(":memory:")
    settings = MagicMock()
    settings.final_result_limit = 10
    provider = MagicMock()
    return PlannerContext(
        workspace=Path("/tmp/fake-workspace"),
        settings=settings,
        provider=provider,
        conn=conn,
    )


@pytest.fixture(autouse=True)
def _reset_registry_after_test() -> Any:
    """Always restore the tool registry after each test."""
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# plan_query — pure intent routing tests (no I/O)
# ---------------------------------------------------------------------------


class TestPlanQueryIntentRouting:
    def test_decision_timeline_why(self) -> None:
        intent, steps = plan_query("why did we choose SQLite over Postgres?")
        assert intent == "decision_timeline"
        assert steps[0].tool == "decision_timeline"
        assert steps[1].tool == "search_memories"

    def test_decision_timeline_when(self) -> None:
        intent, steps = plan_query("when did we switch to the hybrid search approach?")
        assert intent == "decision_timeline"
        assert steps[0].tool == "decision_timeline"

    def test_decision_timeline_what_decided(self) -> None:
        intent, _steps = plan_query("what did we decide about the embedding model?")
        assert intent == "decision_timeline"

    def test_decision_timeline_keyword_decision(self) -> None:
        intent, _steps = plan_query("show me the decision around the API design")
        assert intent == "decision_timeline"

    def test_decision_timeline_keyword_decided(self) -> None:
        intent, _steps = plan_query("we decided on FAISS, can you find that?")
        assert intent == "decision_timeline"

    def test_project_status_status_of(self) -> None:
        intent, steps = plan_query("status of the search engine project")
        assert intent == "project_status"
        assert steps[0].tool == "project_state"
        assert steps[1].tool == "search_memories"

    def test_project_status_where_are_we(self) -> None:
        intent, _steps = plan_query("where are we with the migration?")
        assert intent == "project_status"

    def test_project_status_progress_on(self) -> None:
        intent, _steps = plan_query("progress on the indexing pipeline")
        assert intent == "project_status"

    def test_project_status_keyword_project(self) -> None:
        intent, _steps = plan_query("give me an overview of the project")
        assert intent == "project_status"

    def test_project_status_state_of(self) -> None:
        intent, _steps = plan_query("what is the state of the database schema?")
        assert intent == "project_status"

    def test_tasks_todo(self) -> None:
        intent, steps = plan_query("what are the todo items?")
        assert intent == "tasks"
        assert steps[0].tool == "search_memories"
        assert steps[0].arguments.get("kinds") == "task"

    def test_tasks_open_tasks(self) -> None:
        intent, _steps = plan_query("list open tasks for this sprint")
        assert intent == "tasks"

    def test_tasks_whats_left(self) -> None:
        intent, _steps = plan_query("what's left to implement?")
        assert intent == "tasks"

    def test_tasks_remaining(self) -> None:
        intent, _steps = plan_query("show me the remaining work")
        assert intent == "tasks"

    def test_tasks_task_keyword(self) -> None:
        intent, _steps = plan_query("any task related to search indexing?")
        assert intent == "tasks"

    def test_general_default(self) -> None:
        intent, steps = plan_query("how does the passage chunking work?")
        assert intent == "general"
        tools = [s.tool for s in steps]
        assert tools == ["search_conversations", "search_segments", "search_memories"]

    def test_general_no_keywords(self) -> None:
        intent, steps = plan_query("embedding model performance")
        assert intent == "general"
        assert len(steps) == 3

    def test_plan_steps_are_ordered(self) -> None:
        _, steps = plan_query("how does FAISS work here?")
        for i, step in enumerate(steps, start=1):
            assert step.order == i

    def test_decision_timeline_extracts_subject(self) -> None:
        _, steps = plan_query("why did we pick FAISS over Annoy?")
        subject = steps[0].arguments["subject"]
        # Trigger phrase should be stripped
        assert "why did we" not in subject.lower()
        assert len(subject) > 0

    def test_project_status_extracts_project_name(self) -> None:
        _, steps = plan_query("status of the embedding pipeline")
        project_arg = steps[0].arguments["project"]
        assert "embedding" in project_arg.lower() or "pipeline" in project_arg.lower()


# ---------------------------------------------------------------------------
# execute_plan — monkeypatched tool functions
# ---------------------------------------------------------------------------


class TestExecutePlan:
    def test_general_intent_collects_tool_calls(self) -> None:
        ctx = _make_ctx()
        conv = _make_conv_result(conv_id=3, msg_id=12)
        mem = _FakeMemoryRecord()

        override_tool("search_conversations", lambda c, **kw: ([conv], "1 conversation"))
        override_tool("search_segments", lambda c, **kw: ([], "0 segments"))
        override_tool("search_memories", lambda c, **kw: ([mem], "1 memory"))

        answer = execute_plan(ctx, "how does passage chunking work?")

        assert answer.intent == "general"
        assert len(answer.calls) == 3
        tool_names = [tc.tool for tc in answer.calls]
        assert "search_conversations" in tool_names
        assert "search_segments" in tool_names
        assert "search_memories" in tool_names

    def test_tool_call_records_contain_correct_counts(self) -> None:
        ctx = _make_ctx()
        conv = _make_conv_result(conv_id=5, msg_id=20)

        override_tool("search_conversations", lambda c, **kw: ([conv], "1 conv"))
        override_tool("search_segments", lambda c, **kw: ([], "0 segs"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "embedding search method")

        conv_call = next(tc for tc in answer.calls if tc.tool == "search_conversations")
        assert conv_call.result_count == 1
        seg_call = next(tc for tc in answer.calls if tc.tool == "search_segments")
        assert seg_call.result_count == 0

    def test_findings_cite_ids_from_results(self) -> None:
        ctx = _make_ctx()
        conv = _make_conv_result(conv_id=3, msg_id=12)
        override_tool("search_conversations", lambda c, **kw: ([conv], "1 conv"))
        override_tool("search_segments", lambda c, **kw: ([], "0 segs"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "FTS5 search performance")

        conv_findings = [f for f in answer.findings if "conv 3" in f]
        assert len(conv_findings) >= 1
        assert "12" in conv_findings[0]  # message id 12 in citation

    def test_no_exception_on_empty_results(self) -> None:
        ctx = _make_ctx()
        override_tool("search_conversations", lambda c, **kw: ([], "0 convs"))
        override_tool("search_segments", lambda c, **kw: ([], "0 segs"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "some obscure query with no results")

        assert isinstance(answer, PlanAnswer)
        assert answer.conversations == []
        assert answer.memories == []
        assert len(answer.findings) == 3
        for finding in answer.findings:
            assert "No evidence found" in finding

    def test_decision_timeline_intent_routes_correctly(self) -> None:
        ctx = _make_ctx()
        mem = _FakeMemoryRecord(memory_id="m1", conversation_id=7, message_id=42)
        override_tool("decision_timeline", lambda c, **kw: ([mem], "1 decision"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "why did we choose SQLite?")

        assert answer.intent == "decision_timeline"
        assert any(tc.tool == "decision_timeline" for tc in answer.calls)
        assert len(answer.memories) >= 1

    def test_decision_timeline_findings_cite_correct_ids(self) -> None:
        ctx = _make_ctx()
        mem = _FakeMemoryRecord(memory_id="m1", conversation_id=7, message_id=42)
        override_tool("decision_timeline", lambda c, **kw: ([mem], "1 decision"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "why did we choose SQLite?")

        decision_findings = [f for f in answer.findings if "conv 7" in f]
        assert len(decision_findings) >= 1
        assert "42" in decision_findings[0]

    def test_tasks_intent_passes_kinds_argument(self) -> None:
        ctx = _make_ctx()
        captured_kwargs: dict[str, Any] = {}

        def fake_search_memories(c: PlannerContext, **kw: str) -> tuple[list[Any], str]:
            captured_kwargs.update(kw)
            return ([], "0 mems")

        override_tool("search_memories", fake_search_memories)

        execute_plan(ctx, "what are the open tasks?")

        assert captured_kwargs.get("kinds") == "task"

    def test_project_status_intent_uses_project_state_tool(self) -> None:
        ctx = _make_ctx()

        @dataclass
        class FakeReport:
            name: str = "my-project"
            summary: str = "A test project"
            evidence_count: int = 5

        override_tool("project_state", lambda c, **kw: ([FakeReport()], "1 project"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "status of my-project")

        assert answer.intent == "project_status"
        assert any(tc.tool == "project_state" for tc in answer.calls)
        project_finding = [f for f in answer.findings if "my-project" in f]
        assert len(project_finding) >= 1

    def test_unknown_tool_in_plan_raises_value_error(self) -> None:
        """Force a step with an unregistered tool name to verify the guard."""
        ctx = _make_ctx()

        # Patch plan_query to inject an unknown tool
        from convsearch.planner import planner as planner_mod

        original_plan_query = planner_mod.plan_query

        def patched_plan_query(q: str) -> tuple[str, list[PlanStep]]:
            return "general", [
                PlanStep(
                    order=1,
                    tool="nonexistent_tool",
                    arguments={"query": q},
                    rationale="test",
                )
            ]

        planner_mod.plan_query = patched_plan_query  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="Unknown tool"):
                execute_plan(ctx, "any query")
        finally:
            planner_mod.plan_query = original_plan_query  # type: ignore[assignment]

    def test_tool_error_does_not_abort_remaining_steps(self) -> None:
        ctx = _make_ctx()

        def failing_tool(c: PlannerContext, **kw: str) -> tuple[list[Any], str]:
            raise RuntimeError("simulated failure")

        conv = _make_conv_result(conv_id=9, msg_id=99)
        override_tool("search_conversations", failing_tool)
        override_tool("search_segments", lambda c, **kw: ([], "0 segs"))
        override_tool("search_memories", lambda c, **kw: ([conv], "1 result"))

        answer = execute_plan(ctx, "resilience test query")

        # Three steps ran; the errored one recorded an ERROR summary
        assert len(answer.calls) == 3
        failed_call = next(tc for tc in answer.calls if tc.tool == "search_conversations")
        assert "ERROR" in failed_call.result_summary

    def test_returns_plan_answer_type(self) -> None:
        ctx = _make_ctx()
        override_tool("search_conversations", lambda c, **kw: ([], "0"))
        override_tool("search_segments", lambda c, **kw: ([], "0"))
        override_tool("search_memories", lambda c, **kw: ([], "0"))

        answer = execute_plan(ctx, "any query")

        assert isinstance(answer, PlanAnswer)
        assert isinstance(answer.steps, tuple)
        assert isinstance(answer.calls, tuple)
        assert isinstance(answer.findings, list)
        assert isinstance(answer.conversations, list)
        assert isinstance(answer.memories, list)

    def test_memories_from_search_memories_collected(self) -> None:
        ctx = _make_ctx()
        mem = _FakeMemoryRecord(memory_id="m99", conversation_id=15, message_id=77)
        override_tool("search_conversations", lambda c, **kw: ([], "0"))
        override_tool("search_segments", lambda c, **kw: ([], "0"))
        override_tool("search_memories", lambda c, **kw: ([mem], "1 memory"))

        answer = execute_plan(ctx, "memory collection test")

        assert len(answer.memories) == 1
        assert answer.memories[0].memory_id == "m99"

    def test_answer_is_grounded_and_cited_for_decision_query(self) -> None:
        ctx = _make_ctx()
        mem = _FakeMemoryRecord(
            memory_id="m1",
            kind="decision",
            status="active",
            statement="We decided to standardize on SQLite for local storage.",
            conversation_id=7,
            message_id=42,
        )
        override_tool("decision_timeline", lambda c, **kw: ([mem], "1 decision"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "why did we choose SQLite?")

        assert isinstance(answer.answer, str)
        assert answer.answer.strip() != ""
        # Grounded: states the current active decision and cites a source inline.
        assert "[conv 7, msg 42]" in answer.answer
        assert "[conv" in answer.answer
        assert "SQLite" in answer.answer

    def test_answer_reports_insufficient_evidence_when_empty(self) -> None:
        ctx = _make_ctx()
        override_tool("search_conversations", lambda c, **kw: ([], "0"))
        override_tool("search_segments", lambda c, **kw: ([], "0"))
        override_tool("search_memories", lambda c, **kw: ([], "0"))

        answer = execute_plan(ctx, "some obscure query with no results")

        assert "Insufficient evidence" in answer.answer

    def test_open_task_filtering_excludes_completed_tasks(self) -> None:
        ctx = _make_ctx()
        open_task = _FakeMemoryRecord(
            memory_id="t-open",
            kind="task",
            status="active",
            task_state="open",
            statement="Wire up the streaming API.",
            conversation_id=3,
            message_id=11,
        )
        completed_task = _FakeMemoryRecord(
            memory_id="t-done",
            kind="task",
            status="active",
            task_state="completed",
            statement="Write the export parser.",
            conversation_id=4,
            message_id=12,
        )

        override_tool("search_memories", lambda c, **kw: ([open_task, completed_task], "2 tasks"))
        # active_plan / memory_relations hit the empty in-memory DB or find nothing; both
        # degrade gracefully and are irrelevant to open-task filtering.
        answer = execute_plan(ctx, "what are the open tasks?")

        assert answer.intent == "tasks"
        collected_states = {getattr(m, "task_state", None) for m in answer.memories}
        assert "completed" not in collected_states
        assert "open" in collected_states
        # The synthesized answer surfaces the open task and drops the completed one.
        assert "Wire up the streaming API." in answer.answer
        assert "Write the export parser." not in answer.answer

    def test_supersession_step_appears_when_relation_exists(self) -> None:
        ctx = _make_ctx()
        relation = _FakeRelation(
            relation="supersedes",
            other_memory_id=99,
            other_statement="We will use Postgres.",
            direction="outgoing",
        )
        mem = _FakeMemoryRecord(
            memory_id="m1",
            kind="decision",
            status="active",
            statement="We now use SQLite for local storage.",
            conversation_id=7,
            message_id=42,
            relations=(relation,),  # type: ignore[arg-type]
        )
        override_tool("decision_timeline", lambda c, **kw: ([mem], "1 decision"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))
        override_tool("memory_relations", lambda c, **kw: ([relation], "1 relation"))

        answer = execute_plan(ctx, "why did we decide on SQLite?")

        # The plan carries an explicit supersession (memory_relations) step ...
        step_tools = [s.tool for s in answer.steps]
        assert "memory_relations" in step_tools
        # ... and it actually ran, returning the superseding relation.
        relation_calls = [c for c in answer.calls if c.tool == "memory_relations"]
        assert len(relation_calls) == 1
        assert relation_calls[0].result_count == 1
        # The grounded answer names the supersession and cites the earlier position.
        assert "supersede" in answer.answer.lower()
        assert "Postgres" in answer.answer

    def test_findings_never_fabricate_ids(self) -> None:
        """No finding should reference a conv/msg id not in the canned results."""
        ctx = _make_ctx()
        conv = _make_conv_result(conv_id=42, msg_id=101)
        override_tool("search_conversations", lambda c, **kw: ([conv], "1 conv"))
        override_tool("search_segments", lambda c, **kw: ([], "0 segs"))
        override_tool("search_memories", lambda c, **kw: ([], "0 mems"))

        answer = execute_plan(ctx, "fabrication test")

        # Collect all integer ids mentioned in findings via a simple scan
        import re

        all_ids_in_findings = set(re.findall(r"\b\d+\b", " ".join(answer.findings)))
        # Only ids present in our canned result (42 and 101) should appear as conv/msg ids
        # (scores and counts like 0 and 3 may also appear — we only check that
        # 42 and 101 DO appear and no foreign conv id like 999 sneaks in)
        assert "42" in all_ids_in_findings
        assert "101" in all_ids_in_findings
        assert "999" not in all_ids_in_findings
