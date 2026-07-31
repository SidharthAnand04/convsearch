"""Read-only tool registry for the convsearch query planner.

All tools in this module are read-only by construction: they only call retrieval,
memory-search, and project-reconstruction APIs that perform SELECT queries.
No tool may INSERT, UPDATE, or DELETE data.  This is enforced by design: every
tool delegates to a read API and never receives a writable cursor or session.

The registry maps tool names to callables with the signature::

    (ctx: PlannerContext, **kwargs: str) -> tuple[list[Any], str]

where the return value is (results, human-readable summary).

Lazy imports are used for convsearch.memory and convsearch.projects so this
module loads cleanly even before those packages are present.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from convsearch.config.settings import Settings
from convsearch.embeddings.sentence_transformers import EmbeddingProvider


@dataclass
class PlannerContext:
    """Runtime context threaded through every tool call."""

    workspace: Path
    settings: Settings
    provider: EmbeddingProvider
    conn: sqlite3.Connection


@dataclass(frozen=True)
class ToolCall:
    """Record of a single executed tool call."""

    tool: str
    arguments: dict[str, str]
    result_summary: str
    result_count: int


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


def search_conversations_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Full hybrid search returning ranked ConversationResult objects.

    Read-only: delegates to search_conversations which only performs SELECT.
    """
    from convsearch.retrieval.service import search_conversations  # lazy

    query = kwargs.get("query", "")
    limit = int(kwargs.get("limit", str(ctx.settings.final_result_limit)))
    profile = kwargs.get("profile", "balanced")
    show_passages = int(kwargs.get("show_passages", "3"))

    results = search_conversations(
        ctx.workspace,
        query,
        ctx.settings,
        ctx.provider,
        limit=limit,
        profile=profile,
        show_passages=show_passages,
        include_branches=False,
        rerank=False,
        llm_query=False,
    )
    summary = f"{len(results)} conversation(s) matched"
    return results, summary


def search_passages_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Passage-level search: returns ConversationResult objects but surfaces passage text.

    Read-only: thin wrapper over search_conversations with show_passages=10.
    """
    from convsearch.retrieval.service import search_conversations  # lazy

    query = kwargs.get("query", "")
    limit = int(kwargs.get("limit", str(ctx.settings.final_result_limit)))
    profile = kwargs.get("profile", "balanced")

    results = search_conversations(
        ctx.workspace,
        query,
        ctx.settings,
        ctx.provider,
        limit=limit,
        profile=profile,
        show_passages=10,
        include_branches=False,
        rerank=False,
        llm_query=False,
    )
    summary = f"{len(results)} conversation(s) with passage detail"
    return results, summary


def search_segments_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Segment-level FTS search returning SegmentResult objects.

    Read-only: delegates to search_segments which only performs SELECT.
    """
    from convsearch.retrieval.service import search_segments  # lazy

    query = kwargs.get("query", "")
    limit = int(kwargs.get("limit", str(ctx.settings.final_result_limit)))

    results = search_segments(
        ctx.workspace,
        query,
        ctx.settings,
        limit=limit,
        include_branches=False,
    )
    summary = f"{len(results)} segment(s) matched"
    return results, summary


def search_memories_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Search extracted memory records by semantic similarity.

    Read-only: delegates to search_memories which only performs SELECT.
    """
    from convsearch.memory.search import search_memories

    query = kwargs.get("query", "")
    kinds_raw = kwargs.get("kinds", "")
    kinds = [k.strip() for k in kinds_raw.split(",") if k.strip()] if kinds_raw else None
    statuses_raw = kwargs.get("statuses", "")
    statuses = [s.strip() for s in statuses_raw.split(",") if s.strip()] if statuses_raw else None
    project = kwargs.get("project") or None
    limit = int(kwargs.get("limit", "20"))

    results = search_memories(
        ctx.conn,
        query,
        kinds=kinds,
        statuses=statuses,
        project=project,
        limit=limit,
    )
    summary = f"{len(results)} memory record(s) matched"
    return results, summary


def active_plan_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Return the latest ACTIVE decision / project_state memories (the current plan).

    Read-only: delegates to list_memories which only performs SELECT.  Results are
    ordered newest-first so the caller can treat the first record as the current plan.
    """
    from convsearch.memory.search import list_memories  # lazy

    project = kwargs.get("project") or None
    limit = int(kwargs.get("limit", "20"))
    records: list[Any] = []
    for kind in ("project_state", "decision"):
        records.extend(
            list_memories(ctx.conn, kind=kind, status="active", project=project, limit=limit)
        )
    # Newest first: records with a created_at rank above those without, then by date desc.
    records.sort(key=lambda m: (m.created_at is not None, m.created_at or ""), reverse=True)
    summary = f"{len(records)} active plan memory record(s)"
    return records, summary


def decision_timeline_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Chronological decision timeline for a subject.

    Read-only: delegates to decision_timeline which only performs SELECT.
    """
    from convsearch.memory.search import decision_timeline  # lazy

    subject = kwargs.get("subject", "")
    results = decision_timeline(ctx.conn, subject)
    summary = f"{len(results)} decision record(s) for subject '{subject}'"
    return results, summary


def project_state_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Reconstruct full project report for a named project.

    Read-only: delegates to reconstruct_project which only performs SELECT.
    """
    from convsearch.projects.reconstruct import reconstruct_project  # lazy

    project_name = kwargs.get("project", "")
    report = reconstruct_project(ctx.conn, project_name)
    if report is None:
        return [], f"No project found with name '{project_name}'"
    return [report], f"Project '{report.name}' reconstructed"


def memory_relations_tool(
    ctx: PlannerContext,
    **kwargs: str,
) -> tuple[list[Any], str]:
    """Fetch a single memory record and return its relations.

    Read-only: delegates to get_memory which only performs SELECT.
    """
    from convsearch.memory.search import get_memory  # lazy

    memory_id_str = kwargs.get("memory_id", "")
    record = get_memory(ctx.conn, int(memory_id_str)) if memory_id_str else None
    if record is None:
        return [], f"Memory '{memory_id_str}' not found"
    relations = list(record.relations) if record.relations else []
    summary = f"{len(relations)} relation(s) for memory '{memory_id_str}'"
    return relations, summary


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps tool name -> callable (ctx, **kwargs) -> (results, summary)
# All entries are read-only by construction (see module docstring).

ToolFn = Any  # Callable[[PlannerContext, ...], tuple[list[Any], str]]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "search_conversations": search_conversations_tool,
    "search_passages": search_passages_tool,
    "search_segments": search_segments_tool,
    "search_memories": search_memories_tool,
    "active_plan": active_plan_tool,
    "decision_timeline": decision_timeline_tool,
    "project_state": project_state_tool,
    "memory_relations": memory_relations_tool,
}


@dataclass
class _RegistryState:
    """Mutable singleton wrapping TOOL_REGISTRY for monkeypatching in tests."""

    _registry: dict[str, ToolFn] = field(default_factory=lambda: dict(TOOL_REGISTRY))

    def get(self, name: str) -> ToolFn | None:
        return self._registry.get(name)

    def override(self, name: str, fn: ToolFn) -> None:
        self._registry[name] = fn

    def reset(self) -> None:
        self._registry = dict(TOOL_REGISTRY)


_state = _RegistryState()


def get_tool(name: str) -> ToolFn | None:
    """Look up a tool by name from the active registry."""
    return _state.get(name)


def override_tool(name: str, fn: ToolFn) -> None:
    """Replace a tool in the active registry (for testing only)."""
    _state.override(name, fn)


def reset_registry() -> None:
    """Restore the registry to its original state (for testing only)."""
    _state.reset()
