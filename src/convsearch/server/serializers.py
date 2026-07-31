"""JSON serialization for the local HTTP API.

These are pure functions that turn engine domain objects into plain, JSON-safe dicts. They
live apart from `app.py` so the request handler stays about routing and locking while the
exact response shapes (which the frontend and tests depend on) sit in one place.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from convsearch.answer.answer import AnswerResult
from convsearch.capture.inventory import CaptureInventory, CaptureItem
from convsearch.diagnostics.doctor import Check
from convsearch.diagnostics.llm_readiness import LlmReadiness
from convsearch.domain.models import PassageHit, SegmentResult
from convsearch.memory.models import MemoryEvidence, MemoryRecord
from convsearch.memory.review import ConflictRef, ReviewItem, ReviewQueue, SupersessionRef
from convsearch.planner.planner import PlanAnswer
from convsearch.projects.reconstruct import ProjectItem, ProjectReport, ProjectSummary
from convsearch.retrieval.explain import build_reason, passage_explain
from convsearch.tasks.query import TaskEvidence, TaskItem, TaskList
from convsearch.timeline.build import Timeline, TimelineEvidence, TimelineNode


def passage_payload(hit: PassageHit, *, explain: bool = False) -> dict[str, Any]:
    """Serialize one passage hit, optionally attaching its scoring breakdown."""
    payload: dict[str, Any] = {
        "passage_id": hit.passage_id,
        "message_id": hit.message_id,
        "role": hit.role,
        "text": hit.text,
        "created_at": hit.created_at,
        "is_primary_path": hit.is_primary_path,
        "branch": "selected path" if hit.is_primary_path else "alternate branch",
        "segment_id": hit.segment_id,
        "segment_title": hit.segment_title,
        "channels": list(hit.channels),
        "score": hit.final_score if hit.final_score is not None else hit.fused_score,
    }
    if explain:
        payload["explain"] = passage_explain(hit)
    return payload


def _passage_sort_key(hit: PassageHit) -> float:
    return hit.final_score if hit.final_score is not None else hit.fused_score


def segment_payload(result: SegmentResult, *, explain: bool = False) -> dict[str, Any]:
    return {
        "segment_id": result.segment_id,
        "conversation_id": result.conversation_id,
        "conversation_title": result.conversation_title,
        "title": result.title,
        "score": result.score,
        "passages": [passage_payload(hit, explain=explain) for hit in result.best_passages],
    }


def flat_passage_payload(
    passages: list[PassageHit], *, explain: bool = False
) -> list[dict[str, Any]]:
    """Flatten and rank passages across conversation results, best score first."""
    ranked = sorted(passages, key=_passage_sort_key, reverse=True)
    return [passage_payload(hit, explain=explain) for hit in ranked]


def memory_list_item(record: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "statement": record.statement,
        "kind": record.kind,
        "status": record.status,
        "project": record.project,
        "subject_key": record.subject_key,
        "confidence": record.confidence,
        "conversation_id": getattr(record, "conversation_id", None),
        "conversation_title": record.conversation_title,
        "created_at": record.created_at,
    }


def memory_detail_payload(
    record: MemoryRecord, status_history: list[sqlite3.Row]
) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "statement": record.statement,
        "kind": record.kind,
        "status": record.status,
        "project": record.project,
        "subject_key": record.subject_key,
        "confidence": record.confidence,
        "task_state": record.task_state,
        "conversation_id": getattr(record, "conversation_id", None),
        "conversation_title": record.conversation_title,
        "created_at": record.created_at,
        "evidence": [
            {
                "evidence_id": ev.evidence_id,
                "message_id": ev.message_id,
                "passage_id": ev.passage_id,
                "quote": ev.quote,
                "start_offset": ev.start_offset,
                "end_offset": ev.end_offset,
            }
            for ev in record.evidence
        ],
        "relations": [
            {
                "direction": rel.direction,
                "relation": rel.relation,
                "other_memory_id": rel.other_memory_id,
                "other_statement": rel.other_statement,
                "reason": rel.reason,
            }
            for rel in record.relations
        ],
        "status_history": [
            {
                "old_status": row["old_status"],
                "new_status": row["new_status"],
                "reason": row["reason"],
                "changed_at": row["changed_at"],
            }
            for row in status_history
        ],
    }


def _review_evidence_payload(ev: MemoryEvidence) -> dict[str, Any]:
    return {
        "evidence_id": ev.evidence_id,
        "message_id": ev.message_id,
        "passage_id": ev.passage_id,
        "quote": ev.quote,
        "start_offset": ev.start_offset,
        "end_offset": ev.end_offset,
    }


def _review_conflict_payload(ref: ConflictRef) -> dict[str, Any]:
    return {
        "memory_id": ref.memory_id,
        "statement": ref.statement,
        "status": ref.status,
        "reason": ref.reason,
    }


def _review_supersession_payload(ref: SupersessionRef) -> dict[str, Any]:
    return {"memory_id": ref.memory_id, "statement": ref.statement}


def review_state_payload(
    *,
    memory_id: int,
    kind: str,
    statement: str,
    status: str,
    project: str | None,
    confidence: float,
    created_at: str | None,
    pinned: bool,
    reviewed_at: str | None,
    conversation_id: int,
    conversation_title: str | None,
    evidence: tuple[MemoryEvidence, ...],
    conflicts: tuple[ConflictRef, ...],
    superseded_by: tuple[SupersessionRef, ...],
    review_reason: str | None,
) -> dict[str, Any]:
    """The review-item shape shared by the queue listing and the mutation endpoints.

    The mutation endpoints (confirm/invalidate/pin) re-read a memory that may no longer
    qualify for the pending queue -- e.g. it was just invalidated or pinned -- so
    `review_reason` is optional here even though `ReviewItem.review_reason` (queue
    members only) never is.
    """
    return {
        "memory_id": memory_id,
        "kind": kind,
        "statement": statement,
        "status": status,
        "project": project,
        "confidence": confidence,
        "created_at": created_at,
        "pinned": pinned,
        "reviewed_at": reviewed_at,
        "conversation_id": conversation_id,
        "conversation_title": conversation_title,
        "evidence": [_review_evidence_payload(ev) for ev in evidence],
        "conflicts": [_review_conflict_payload(c) for c in conflicts],
        "superseded_by": [_review_supersession_payload(s) for s in superseded_by],
        "review_reason": review_reason,
    }


def review_item_payload(item: ReviewItem) -> dict[str, Any]:
    return review_state_payload(
        memory_id=item.memory_id,
        kind=item.kind,
        statement=item.statement,
        status=item.status,
        project=item.project,
        confidence=item.confidence,
        created_at=item.created_at,
        pinned=item.pinned,
        reviewed_at=item.reviewed_at,
        conversation_id=item.conversation_id,
        conversation_title=item.conversation_title,
        evidence=item.evidence,
        conflicts=item.conflicts,
        superseded_by=item.superseded_by,
        review_reason=item.review_reason,
    )


def review_queue_payload(queue: ReviewQueue) -> dict[str, Any]:
    return {
        "total_pending": queue.total_pending,
        "total_pinned": queue.total_pinned,
        "total_contested": queue.total_contested,
        "total_invalidated": queue.total_invalidated,
        "count": len(queue.items),
        "items": [review_item_payload(i) for i in queue.items],
    }


def project_summary_payload(summary: ProjectSummary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "memory_count": summary.memory_count,
        "conversation_count": summary.conversation_count,
        "decision_count": summary.decision_count,
        "open_task_count": summary.open_task_count,
        "last_activity": summary.last_activity,
        "date_source": summary.date_source,
    }


def _project_item_payload(item: ProjectItem) -> dict[str, Any]:
    return {
        "memory_id": item.memory_id,
        "statement": item.statement,
        "status": item.status,
        "created_at": item.created_at,
        "date_source": item.date_source,
        "subject_key": item.subject_key,
        "evidence": [
            {
                "memory_id": ev.memory_id,
                "conversation_id": ev.conversation_id,
                "conversation_title": ev.conversation_title,
                "message_id": ev.message_id,
                "passage_id": ev.passage_id,
                "quote": ev.quote,
            }
            for ev in item.evidence
        ],
    }


def project_report_payload(report: ProjectReport) -> dict[str, Any]:
    return {
        "name": report.name,
        "summary": report.summary,
        "evidence_count": report.evidence_count,
        "architecture": [_project_item_payload(i) for i in report.architecture],
        "decisions": [_project_item_payload(i) for i in report.decisions],
        "superseded_decisions": [_project_item_payload(i) for i in report.superseded_decisions],
        "rejected_alternatives": list(report.rejected_alternatives),
        "open_tasks": [_project_item_payload(i) for i in report.open_tasks],
        "completed_tasks": [_project_item_payload(i) for i in report.completed_tasks],
        "risks": [_project_item_payload(i) for i in report.risks],
        "timeline": [
            {
                "created_at": entry.created_at,
                "kind": entry.kind,
                "status": entry.status,
                "statement": entry.statement,
                "memory_id": entry.memory_id,
                "date_source": entry.date_source,
            }
            for entry in report.timeline
        ],
        "conversations": [
            {"conversation_id": conv_id, "title": title} for conv_id, title in report.conversations
        ],
        # Populated by a parallel change; default to empty so the shape is stable now.
        "known_bugs": list(getattr(report, "known_bugs", [])),
        "next_milestones": list(getattr(report, "next_milestones", [])),
    }


def conversation_payload(
    row: sqlite3.Row, messages: list[sqlite3.Row], url: str | None
) -> dict[str, Any]:
    return {
        "conversation_id": row["conversation_id"],
        "title": row["title"],
        "url": url,
        "source_conversation_id": row["source_conversation_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [
            {
                "message_id": msg["message_id"],
                "role": msg["role"],
                "text": msg["text"],
                "created_at": msg["created_at"],
                "is_primary_path": bool(msg["is_primary_path"]),
                "source_order": msg["source_order"],
            }
            for msg in messages
        ],
    }


def answer_payload(result: AnswerResult) -> dict[str, Any]:
    return {
        "question": result.question,
        "answer": result.answer,
        "backend": result.backend,
        "model": result.model,
        "sources": [
            {
                "index": src.index,
                "conversation_id": src.conversation_id,
                "title": src.title,
                "date": src.date,
                "role": src.role,
                "quote": src.quote,
            }
            for src in result.sources
        ],
    }


def plan_payload(query: str, answer: PlanAnswer) -> dict[str, Any]:
    """Serialize a planner result: intent, cited answer, plan steps and executed calls."""
    return {
        "query": query,
        "intent": answer.intent,
        "answer": answer.answer,
        "steps": [
            {"order": step.order, "tool": step.tool, "rationale": step.rationale}
            for step in answer.steps
        ],
        "calls": [
            {
                "tool": call.tool,
                "result_count": call.result_count,
                "result_summary": call.result_summary,
            }
            for call in answer.calls
        ],
        "findings": list(answer.findings),
    }


def suggestions_payload(recent: list[str], popular: list[tuple[str, int]]) -> dict[str, Any]:
    """Query suggestions for the UI: recent distinct queries and popular (query, count)s."""
    return {
        "recent": list(recent),
        "popular": [[query, count] for query, count in popular],
    }


def _task_evidence_payload(ev: TaskEvidence) -> dict[str, Any]:
    return {
        "quote": ev.quote,
        "conversation_id": ev.conversation_id,
        "conversation_title": ev.conversation_title,
        "message_id": ev.message_id,
        "passage_id": ev.passage_id,
        "role": ev.role,
        "timestamp": ev.timestamp,
    }


def task_item_payload(item: TaskItem) -> dict[str, Any]:
    return {
        "memory_id": item.memory_id,
        "statement": item.statement,
        "project": item.project,
        "task_state": item.task_state,
        "status": item.status,
        "confidence": item.confidence,
        "created_at": item.created_at,
        "date_source": item.date_source,
        "conversation_id": item.conversation_id,
        "conversation_title": item.conversation_title,
        "has_evidence": item.has_evidence,
        "evidence": [_task_evidence_payload(ev) for ev in item.evidence],
        "task_state_changed_at": item.task_state_changed_at,
        "task_state_source": item.task_state_source,
    }


def task_list_payload(result: TaskList) -> dict[str, Any]:
    return {
        "total_open": result.total_open,
        "total_completed": result.total_completed,
        "projects": list(result.projects),
        "count": len(result.items),
        "items": [task_item_payload(item) for item in result.items],
    }


def _timeline_evidence_payload(ev: TimelineEvidence) -> dict[str, Any]:
    return {
        "quote": ev.quote,
        "conversation_id": ev.conversation_id,
        "conversation_title": ev.conversation_title,
        "message_id": ev.message_id,
        "timestamp": ev.timestamp,
    }


def _timeline_node_payload(node: TimelineNode) -> dict[str, Any]:
    return {
        "memory_id": node.memory_id,
        "kind": node.kind,
        "statement": node.statement,
        "status": node.status,
        "project": node.project,
        "created_at": node.created_at,
        "date_source": node.date_source,
        "confidence": node.confidence,
        "conversation_id": node.conversation_id,
        "conversation_title": node.conversation_title,
        "supersedes": list(node.supersedes),
        "superseded_by": list(node.superseded_by),
        "conflicts_with": list(node.conflicts_with),
        "reasons": list(node.reasons),
        "evidence": [_timeline_evidence_payload(ev) for ev in node.evidence],
    }


def timeline_payload(result: Timeline) -> dict[str, Any]:
    return {
        "topic": result.topic,
        "matched_count": result.matched_count,
        "first_seen": result.first_seen,
        "first_seen_source": result.first_seen_source,
        "last_seen": result.last_seen,
        "last_seen_source": result.last_seen_source,
        "nodes": [_timeline_node_payload(n) for n in result.nodes],
        "active": [_timeline_node_payload(n) for n in result.active],
        "superseded": [_timeline_node_payload(n) for n in result.superseded],
        "contested": [_timeline_node_payload(n) for n in result.contested],
        "rejected": [_timeline_node_payload(n) for n in result.rejected],
    }


def _capture_item_payload(item: CaptureItem) -> dict[str, Any]:
    return {
        "conversation_id": item.conversation_id,
        "source_conversation_id": item.source_conversation_id,
        "title": item.title,
        "captured_at": item.captured_at,
        "date_source": item.date_source,
        "updated_at": item.updated_at,
        "message_count": item.message_count,
        "source": item.source,
        "indexed": item.indexed,
        "segmented": item.segmented,
        "memories_extracted": item.memories_extracted,
        "passage_count": item.passage_count,
        "memory_count": item.memory_count,
        "source_url": item.source_url,
        "warnings": list(item.warnings),
    }


def capture_inventory_payload(inventory: CaptureInventory) -> dict[str, Any]:
    return {
        "total": inventory.total,
        "live_captured": inventory.live_captured,
        "imported": inventory.imported,
        "not_indexed": inventory.not_indexed,
        "not_segmented": inventory.not_segmented,
        "stale_index": inventory.stale_index,
        "count": len(inventory.items),
        "items": [_capture_item_payload(i) for i in inventory.items],
    }


def diagnostics_payload(checks: list[Check], readiness: LlmReadiness) -> dict[str, Any]:
    return {
        "ready": readiness.ready,
        "backend": readiness.backend,
        "summary": readiness.summary,
        "remediation": list(readiness.remediation),
        "llm_checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in readiness.checks],
        "doctor_checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }


# What actually leaves the machine when a cloud backend serves an /ask or /plan request,
# per `answer.build_prompt` (question + numbered context blocks) and the trim limit that
# bounds each excerpt (`answer.answer._PROMPT_QUOTE_LIMIT`, 600 chars).
_CLOUD_PAYLOAD_NOTE = (
    "If a cloud backend serves the request, only the question text and the retrieved "
    "passage excerpts used as context are sent to the Anthropic API -- each excerpt is "
    "trimmed to at most 600 characters and labeled with its conversation title, date, and "
    "role. No other conversation history, memories, or workspace metadata leave the "
    "machine for that request."
)


def privacy_payload(
    *,
    workspace_path: str,
    database_path: str,
    index_path: str,
    server_bind: str,
    loopback_only: bool,
    backend_mode: str,
    ollama_host: str,
    cloud_configured: bool,
    readiness: LlmReadiness,
    counts: dict[str, int],
) -> dict[str, Any]:
    """Report the real, verified state behind the "everything stays local" claim.

    `effective_backend` and `cloud_would_be_used` come straight from `probe_llm_readiness`,
    which is the same resolution `/ask` and `/plan` use to pick a backend -- this must never
    disagree with what those routes actually do. `local_only` is the complement of
    `cloud_would_be_used`: true whenever a cloud call would NOT happen right now, whether
    because Ollama is serving (auto/ollama mode) or because nothing is configured to serve
    at all (a broken `anthropic` mode has no cloud call either -- it just fails).
    """
    effective_backend = readiness.backend
    cloud_would_be_used = effective_backend == "anthropic"
    local_only = not cloud_would_be_used
    return {
        "local_only": local_only,
        "workspace_path": workspace_path,
        "database_path": database_path,
        "index_path": index_path,
        "server_bind": server_bind,
        "loopback_only": loopback_only,
        "llm": {
            "backend_mode": backend_mode,
            "effective_backend": effective_backend,
            "ollama_host": ollama_host,
            "cloud_configured": cloud_configured,
            "cloud_would_be_used": cloud_would_be_used,
        },
        "cloud_payload_note": _CLOUD_PAYLOAD_NOTE,
        "counts": dict(counts),
    }


def build_conversation_result_payload(
    results: Any, source_ids: dict[int, str], *, explain: bool = False
) -> list[dict[str, Any]]:
    """Conversation-level results with optional explain fields.

    Kept here so the `explain=1` variant and the default share one passage serializer.
    """
    payload: list[dict[str, Any]] = []
    for result in results:
        source_id = source_ids.get(result.conversation_id)
        item: dict[str, Any] = {
            "conversation_id": result.conversation_id,
            "source_conversation_id": source_id,
            "url": f"https://chatgpt.com/c/{source_id}" if source_id else None,
            "title": result.title,
            "created_at": result.created_at,
            "updated_at": result.updated_at,
            "score": result.score,
            "distinct_message_count": result.distinct_message_count,
            "features": result.features,
            "passages": [passage_payload(hit, explain=explain) for hit in result.best_passages],
        }
        if explain:
            item["reason"] = build_reason(result)
        payload.append(item)
    return payload
