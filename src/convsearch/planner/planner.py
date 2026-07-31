"""Deterministic, local, rule-based query planner for convsearch.

Constraints (enforced here and in tools.py):

1. Only registered tools may execute — ``execute_plan`` calls ``get_tool()`` for
   every step and raises ``ValueError`` on an unknown name.
2. No memory writes — all tools are read-only by construction (see tools.py).
3. No source invention — findings are derived exclusively from tool results;
   every finding cites conversation_id / message_id values that appear in
   the returned data.  Callers must not fabricate citations.
4. ``include_branches=False`` by default — branch messages are excluded unless
   a tool keyword-argument explicitly overrides this.
5. Network / LLM calls are disabled — ``rerank=False`` and ``llm_query=False``
   are passed to search_conversations so no remote calls occur.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from convsearch.domain.models import ConversationResult
from convsearch.planner.tools import PlannerContext, ToolCall, get_tool


@dataclass(frozen=True)
class PlanStep:
    """A single planned tool invocation."""

    order: int
    tool: str
    arguments: dict[str, str]
    rationale: str


@dataclass(frozen=True)
class PlanAnswer:
    """The complete result of executing a query plan."""

    query: str
    intent: str
    steps: tuple[PlanStep, ...]
    calls: tuple[ToolCall, ...]
    conversations: list[ConversationResult]
    memories: list[Any]  # list[MemoryRecord] — typed loosely for lazy import
    findings: list[str]  # human-readable lines; each cites ids from results
    # Grounded, cited natural-language synthesis of the evidence above.  Additive and
    # default-safe so existing callers (e.g. the CLI ``plan`` command) keep working.
    answer: str = ""


# ---------------------------------------------------------------------------
# Intent classification helpers
# ---------------------------------------------------------------------------

_DECISION_TIMELINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhy did we\b", re.IGNORECASE),
    re.compile(r"\bwhen did we\b", re.IGNORECASE),
    re.compile(r"\bwhat did we decide\b", re.IGNORECASE),
    re.compile(r"\bdecision\b", re.IGNORECASE),
    re.compile(r"\bdecided\b", re.IGNORECASE),
]

_PROJECT_STATUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bstatus of\b", re.IGNORECASE),
    re.compile(r"\bstate of\b", re.IGNORECASE),
    re.compile(r"\bwhere are we\b", re.IGNORECASE),
    re.compile(r"\bprogress on\b", re.IGNORECASE),
    re.compile(r"\bproject\b", re.IGNORECASE),
]

_TASK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btodo\b", re.IGNORECASE),
    re.compile(r"\bopen tasks?\b", re.IGNORECASE),
    re.compile(r"\bwhat'?s left\b", re.IGNORECASE),
    re.compile(r"\bremaining\b", re.IGNORECASE),
    re.compile(r"\btask\b", re.IGNORECASE),
]

# Trigger words stripped when deriving the subject for decision_timeline
_DECISION_TRIGGERS = re.compile(
    r"\b(why did we|when did we|what did we decide|decision|decided)\b",
    re.IGNORECASE,
)

# Patterns to capture the project name after a trigger phrase
_PROJECT_NAME_CAPTURE = re.compile(
    r"(?:status of|state of|progress on)\s+([a-zA-Z0-9_\- ]+?)(?:\s*$|\s+\b(?:project|is|are)\b)",
    re.IGNORECASE,
)


def _matches_any(query: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.search(query) for p in patterns)


def _extract_subject(query: str) -> str:
    """Strip decision-trigger words from the query to get the bare subject."""
    subject = _DECISION_TRIGGERS.sub("", query).strip(" ?.,")
    # Collapse multiple spaces
    subject = re.sub(r"\s{2,}", " ", subject)
    return subject or query


def _extract_project_name(query: str) -> str:
    """Best-effort: capture project name after a trigger phrase, else return the full query."""
    match = _PROJECT_NAME_CAPTURE.search(query)
    if match:
        return match.group(1).strip()
    # Try extracting everything after "project"
    m2 = re.search(r"\bproject\s+([a-zA-Z0-9_\- ]+)", query, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return query


# ---------------------------------------------------------------------------
# plan_query — pure, deterministic, no I/O
# ---------------------------------------------------------------------------


def plan_query(query: str) -> tuple[str, list[PlanStep]]:
    """Classify *query* and return ``(intent, steps)`` deterministically.

    Intent routing order (first match wins):

    1. **decision_timeline** — query contains "why did we", "when did we",
       "what did we decide", "decision", or "decided".
       Steps: decision_timeline → search_memories.

    2. **project_status** — query contains "status of", "state of",
       "where are we", "progress on", or "project".
       Steps: project_state → search_memories.

    3. **tasks** — query contains "todo", "open tasks", "what's left",
       "remaining", or "task".
       Steps: search_memories(kinds=task).

    4. **general** (default) — Steps: search_conversations → search_segments
       → search_memories.
    """
    # 1. decision_timeline
    if _matches_any(query, _DECISION_TIMELINE_PATTERNS):
        subject = _extract_subject(query)
        steps = [
            PlanStep(
                order=1,
                tool="decision_timeline",
                arguments={"subject": subject},
                rationale="Retrieve chronological decision records for the extracted subject.",
            ),
            PlanStep(
                order=2,
                tool="search_memories",
                arguments={"query": query},
                rationale="Broaden with semantic memory search in case the timeline is sparse.",
            ),
            PlanStep(
                order=3,
                tool="memory_relations",
                arguments={},
                rationale=(
                    "Trace supersession: follow supersedes/conflicts_with links from the "
                    "current decision (memory_id resolved at execution time)."
                ),
            ),
        ]
        return "decision_timeline", steps

    # 2. project_status
    if _matches_any(query, _PROJECT_STATUS_PATTERNS):
        project_name = _extract_project_name(query)
        steps = [
            PlanStep(
                order=1,
                tool="project_state",
                arguments={"project": project_name},
                rationale="Reconstruct the full project report for the identified project.",
            ),
            PlanStep(
                order=2,
                tool="search_memories",
                arguments={"query": query, "project": project_name},
                rationale="Supplement project report with related memory records.",
            ),
            PlanStep(
                order=3,
                tool="memory_relations",
                arguments={},
                rationale=(
                    "Trace supersession: follow supersedes/conflicts_with links from the "
                    "current project state (memory_id resolved at execution time)."
                ),
            ),
        ]
        return "project_status", steps

    # 3. tasks
    if _matches_any(query, _TASK_PATTERNS):
        steps = [
            PlanStep(
                order=1,
                tool="search_memories",
                arguments={"query": query, "kinds": "task"},
                rationale=(
                    "Find unresolved tasks: retrieve task-kind memory records "
                    "(only OPEN tasks are kept during synthesis)."
                ),
            ),
            PlanStep(
                order=2,
                tool="active_plan",
                arguments={},
                rationale="Pull the latest active plan: current active decision/project memories.",
            ),
            PlanStep(
                order=3,
                tool="memory_relations",
                arguments={},
                rationale=(
                    "Trace supersession: follow supersedes/conflicts_with links from the "
                    "current plan (memory_id resolved at execution time)."
                ),
            ),
        ]
        return "tasks", steps

    # 4. general (default)
    steps = [
        PlanStep(
            order=1,
            tool="search_conversations",
            arguments={"query": query},
            rationale="Hybrid lexical+semantic conversation search for broad coverage.",
        ),
        PlanStep(
            order=2,
            tool="search_segments",
            arguments={"query": query},
            rationale="Segment-level search for topic-focused sub-conversation results.",
        ),
        PlanStep(
            order=3,
            tool="search_memories",
            arguments={"query": query},
            rationale="Memory search to surface extracted facts and decisions.",
        ),
    ]
    return "general", steps


# ---------------------------------------------------------------------------
# execute_plan — runs the plan, collects results, builds findings
# ---------------------------------------------------------------------------


def _format_conv_finding(conv: ConversationResult, query: str) -> str:
    """One human-readable line per conversation result with citation."""
    passage_texts = [p.text[:120] for p in conv.best_passages[:2]]
    snippet = " | ".join(passage_texts) if passage_texts else "(no passage)"
    msg_ids = sorted({p.message_id for p in conv.best_passages})
    citation = f"[conv {conv.conversation_id}, msgs {msg_ids}]"
    return f'Conversation "{conv.title}" (score={conv.score:.3f}): {snippet} {citation}'


def _format_memory_finding(mem: Any) -> str:
    """One human-readable line per memory record with citation."""
    conv_id = getattr(mem, "conversation_id", "?")
    msg_id = getattr(mem, "message_id", "?")
    subject = getattr(mem, "subject_key", "?")
    statement = getattr(mem, "statement", "")[:120]
    kind = getattr(mem, "kind", "?")
    citation = f"[conv {conv_id}, msg {msg_id}]"
    return f"Memory ({kind}) about '{subject}': {statement} {citation}"


def _format_project_finding(report: Any) -> str:
    """One human-readable line for a project report."""
    name = getattr(report, "name", "?")
    summary = getattr(report, "summary", "")[:120]
    ev = getattr(report, "evidence_count", 0)
    return f'Project "{name}": {summary} [evidence_count={ev}]'


@dataclass
class _MutablePlanAnswer:
    query: str
    intent: str
    steps: tuple[PlanStep, ...]
    calls: list[ToolCall] = field(default_factory=list)
    conversations: list[ConversationResult] = field(default_factory=list)
    memories: list[Any] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)  # MemoryRelation from memory_relations


# ---------------------------------------------------------------------------
# Grounded answer synthesis (deterministic, template-based — no LLM)
# ---------------------------------------------------------------------------

_SUPERSESSION_RELATIONS = ("supersedes", "conflicts_with")


def _short(text: Any, limit: int = 160) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _mem_citation(mem: Any) -> str:
    conv_id = getattr(mem, "conversation_id", "?")
    msg_id = getattr(mem, "message_id", "?")
    return f"[conv {conv_id}, msg {msg_id}]"


def _conv_citation(conv: Any) -> str:
    conv_id = getattr(conv, "conversation_id", "?")
    passages = getattr(conv, "best_passages", []) or []
    raw_ids = (getattr(p, "message_id", None) for p in passages)
    msg_ids = sorted({mid for mid in raw_ids if mid is not None})
    if msg_ids:
        return f"[conv {conv_id}, msg {msg_ids[0]}]"
    return f"[conv {conv_id}]"


def _primary_memory(memories: list[Any]) -> Any | None:
    """The latest ACTIVE decision / project_state memory, or ``None``."""
    active = [
        m
        for m in memories
        if getattr(m, "status", None) == "active"
        and getattr(m, "kind", None) in ("decision", "project_state")
    ]
    if not active:
        return None

    def _sort_key(m: Any) -> tuple[bool, str]:
        created = getattr(m, "created_at", None)
        return (created is not None, created or "")

    return max(active, key=_sort_key)


def _relation_ref(r: Any) -> str:
    statement = _short(getattr(r, "other_statement", ""), 120)
    return f'"{statement}" (memory {getattr(r, "other_memory_id", "?")})'


def _supersession_relations(mem: Any) -> list[Any]:
    if mem is None:
        return []
    return [
        r
        for r in getattr(mem, "relations", ()) or ()
        if getattr(r, "relation", None) in _SUPERSESSION_RELATIONS
    ]


def _synthesize_answer(
    *,
    query: str,
    intent: str,
    conversations: list[ConversationResult],
    memories: list[Any],
    relations: list[Any],
    open_tasks: list[Any],
    primary: Any | None,
) -> str:
    """Build a grounded, inline-cited prose paragraph from tool evidence only.

    Deterministic and template-based (no LLM).  Every claim cites ids that appear in
    the collected results; nothing is invented.  Falls back to a plain "insufficient
    evidence" statement when no relevant evidence was found.
    """
    sentences: list[str] = []

    if primary is not None:
        kind_label = "project state" if primary.kind == "project_state" else primary.kind
        stmt = _short(primary.statement)
        sentences.append(f"The current active {kind_label} is: {stmt} {_mem_citation(primary)}.")
        superseded = [
            r
            for r in relations
            if getattr(r, "relation", None) == "supersedes"
            and getattr(r, "direction", "outgoing") == "outgoing"
        ]
        conflicts = [r for r in relations if getattr(r, "relation", None) == "conflicts_with"]
        if superseded:
            listed = "; ".join(_relation_ref(r) for r in superseded)
            sentences.append(f"It supersedes earlier position(s): {listed}.")
        if conflicts:
            listed = "; ".join(_relation_ref(r) for r in conflicts)
            sentences.append(f"It conflicts with: {listed}.")
    elif memories:
        m = memories[0]
        stmt = _short(getattr(m, "statement", "") or getattr(m, "title", ""))
        sentences.append(f"Based on the most relevant memory: {stmt} {_mem_citation(m)}.")
    elif conversations:
        c = conversations[0]
        sentences.append(f'The most relevant conversation is "{c.title}" {_conv_citation(c)}.')

    if intent in ("tasks", "project_status"):
        if open_tasks:
            listed = "; ".join(
                f"{_short(getattr(t, 'statement', ''), 100)} {_mem_citation(t)}" for t in open_tasks
            )
            sentences.append(f"Unresolved open task(s): {listed}.")
        else:
            sentences.append("No open tasks remain unresolved.")

    if not sentences:
        return (
            f'Insufficient evidence to answer "{query}": no matching decisions, project '
            "state, tasks, or conversations were found."
        )
    return " ".join(sentences)


def _run_supersession_step(ctx: PlannerContext, step: PlanStep, acc: _MutablePlanAnswer) -> None:
    """Execute the memory_relations step, tracing supersedes/conflicts_with links.

    Resolves ``memory_id`` from the current primary memory at execution time and only
    hits the tool when that memory actually carries supersession relations, so the step
    stays cheap and read-only when there is nothing to trace.
    """
    primary = _primary_memory(acc.memories)
    rels = _supersession_relations(primary)
    if primary is None or not rels:
        acc.calls.append(
            ToolCall(
                tool="memory_relations",
                arguments=step.arguments,
                result_summary="No supersession relations to trace.",
                result_count=0,
            )
        )
        return

    resolved = {**step.arguments, "memory_id": str(getattr(primary, "memory_id", ""))}
    tool_fn = get_tool("memory_relations")
    if tool_fn is None:
        raise ValueError(
            "Unknown tool 'memory_relations' — only registered read-only tools may run."
        )
    try:
        results, summary = tool_fn(ctx, **resolved)
    except Exception as exc:
        acc.calls.append(
            ToolCall(
                tool="memory_relations",
                arguments=resolved,
                result_summary=f"ERROR: {exc}",
                result_count=0,
            )
        )
        return

    count = len(results) if isinstance(results, list) else 0
    acc.calls.append(
        ToolCall(
            tool="memory_relations",
            arguments=resolved,
            result_summary=summary,
            result_count=count,
        )
    )
    for rel in results:
        acc.relations.append(rel)
        acc.findings.append(
            f"Supersession: memory {getattr(primary, 'memory_id', '?')} "
            f"{getattr(rel, 'relation', '?')} memory {getattr(rel, 'other_memory_id', '?')} "
            f'— "{_short(getattr(rel, "other_statement", ""), 120)}"'
        )


def execute_plan(ctx: PlannerContext, query: str) -> PlanAnswer:
    """Execute the deterministic plan for *query* and return a ``PlanAnswer``.

    Behaviour:
    - Calls ``plan_query`` to determine intent and steps.
    - For each step, looks up the tool in the registry; raises ``ValueError``
      if the name is unknown (constraint: only registered tools may run).
    - Tolerates empty results: findings say "No evidence found for …".
    - Findings cite only ids present in returned data (no fabrication).
    """
    intent, steps = plan_query(query)
    acc = _MutablePlanAnswer(query=query, intent=intent, steps=tuple(steps))

    for step in steps:
        # Supersession tracing resolves its memory_id from evidence gathered by earlier
        # steps, so it is handled separately from the static-argument tools below.
        if step.tool == "memory_relations":
            _run_supersession_step(ctx, step, acc)
            continue

        tool_fn = get_tool(step.tool)
        if tool_fn is None:
            raise ValueError(
                f"Unknown tool '{step.tool}' — only registered read-only tools may run."
            )

        try:
            results, summary = tool_fn(ctx, **step.arguments)
        except Exception as exc:
            # Record the failure but keep going so other steps can still run.
            acc.calls.append(
                ToolCall(
                    tool=step.tool,
                    arguments=step.arguments,
                    result_summary=f"ERROR: {exc}",
                    result_count=0,
                )
            )
            acc.findings.append(f"No evidence found for '{query}' via {step.tool} (error: {exc})")
            continue

        result_count = len(results) if isinstance(results, list) else 0
        acc.calls.append(
            ToolCall(
                tool=step.tool,
                arguments=step.arguments,
                result_summary=summary,
                result_count=result_count,
            )
        )

        if not results:
            acc.findings.append(f"No evidence found for '{query}' via {step.tool}.")
            continue

        # Collect typed results and build findings from them
        if step.tool in ("search_conversations", "search_passages"):
            for conv in results:
                if isinstance(conv, ConversationResult):
                    acc.conversations.append(conv)
                    acc.findings.append(_format_conv_finding(conv, query))

        elif step.tool == "search_segments":
            for seg in results:
                # SegmentResult — cite its conversation_id
                conv_id = getattr(seg, "conversation_id", "?")
                title = getattr(seg, "conversation_title", "?")
                seg_title = getattr(seg, "title", "")
                score = getattr(seg, "score", 0.0)
                passages = getattr(seg, "best_passages", [])
                msg_ids = sorted({p.message_id for p in passages})
                citation = f"[conv {conv_id}, msgs {msg_ids}]"
                acc.findings.append(
                    f'Segment "{seg_title or title}" in conv "{title}"'
                    f" (score={score:.3f}) {citation}"
                )

        elif step.tool in ("decision_timeline", "search_memories", "active_plan"):
            # For the tasks intent's task step, keep only OPEN tasks (exclude completed /
            # stateless), so we surface unresolved work rather than every task ever seen.
            filter_open_tasks = intent == "tasks" and step.arguments.get("kinds") == "task"
            for mem in results:
                if filter_open_tasks and getattr(mem, "task_state", None) != "open":
                    continue
                acc.memories.append(mem)
                acc.findings.append(_format_memory_finding(mem))

        elif step.tool == "project_state":
            for report in results:
                acc.findings.append(_format_project_finding(report))

    # Synthesize the grounded, cited answer from everything the plan collected.
    primary = _primary_memory(acc.memories)
    open_tasks = [
        m
        for m in acc.memories
        if getattr(m, "kind", None) == "task" and getattr(m, "task_state", None) == "open"
    ]
    relations = acc.relations or _supersession_relations(primary)
    answer_text = _synthesize_answer(
        query=query,
        intent=intent,
        conversations=acc.conversations,
        memories=acc.memories,
        relations=relations,
        open_tasks=open_tasks,
        primary=primary,
    )

    return PlanAnswer(
        query=acc.query,
        intent=acc.intent,
        steps=acc.steps,
        calls=tuple(acc.calls),
        conversations=acc.conversations,
        memories=acc.memories,
        findings=acc.findings,
        answer=answer_text,
    )
