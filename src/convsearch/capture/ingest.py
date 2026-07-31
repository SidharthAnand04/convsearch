"""Ingest conversations scraped live from chatgpt.com by the browser extension.

The browser only ever sees the *visible* conversation: one linear chain of turns with no
alternate branches. That is the only structural difference from the export-ZIP importer, so
this module normalises the captured payload into the same `ImportedConversation` /
`ImportedMessage` domain models and reuses
`convsearch.importers.chatgpt.persist_conversation` for all the SQL. Nothing here
duplicates the importer's persistence logic.

Capture never builds embeddings — it only records that the vector index went stale so
`POST /reindex` can rebuild it out of band.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from convsearch.capture.state import ensure_capture_import, set_index_stale
from convsearch.config.settings import Settings
from convsearch.domain.models import ImportedConversation, ImportedMessage
from convsearch.importers.chatgpt import persist_conversation
from convsearch.storage.database import connection
from convsearch.utils import stable_hash

MAX_CAPTURE_BYTES = 8 * 1024 * 1024

DEFAULT_TITLE = "Untitled conversation"


class CaptureValidationError(ValueError):
    """The request body was not a usable capture payload."""


class CapturedMessage(BaseModel):
    """One visible turn. Unknown keys are ignored so the extension can evolve freely."""

    model_config = ConfigDict(extra="ignore")

    source_message_id: str | None = None
    role: str = "unknown"
    text: str = ""
    order: int | None = None
    created_at: str | None = None


class CapturedConversation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_conversation_id: str = Field(min_length=1)
    title: str = DEFAULT_TITLE
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[CapturedMessage] = Field(default_factory=list)


class CapturePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conversations: list[CapturedConversation]


@dataclass(frozen=True)
class CaptureResult:
    conversations_written: int = 0
    messages_written: int = 0
    skipped_unchanged: int = 0

    def payload(self, *, stale_index: bool) -> dict[str, Any]:
        return {
            "conversations_written": self.conversations_written,
            "messages_written": self.messages_written,
            "skipped_unchanged": self.skipped_unchanged,
            "stale_index": stale_index,
        }


def parse_capture_payload(raw: bytes) -> CapturePayload:
    """Decode and validate a `POST /capture` body, raising `CaptureValidationError`."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureValidationError(f"body is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise CaptureValidationError("body must be a JSON object")
    if "conversations" not in decoded:
        raise CaptureValidationError("body is missing `conversations`")
    try:
        return CapturePayload.model_validate(decoded)
    except ValidationError as exc:
        detail = f"{exc.error_count()} error(s)"
        raise CaptureValidationError(f"invalid capture payload: {detail}") from exc


def to_imported_conversation(captured: CapturedConversation) -> ImportedConversation | None:
    """Flatten a captured conversation into the importer's domain model.

    Every message is on the primary path and message N's parent is message N-1, because the
    DOM only exposes the selected branch. Turns with no text are dropped, and a conversation
    with no usable turns is dropped entirely.
    """
    ordered = sorted(
        enumerate(captured.messages),
        key=lambda pair: (pair[1].order if pair[1].order is not None else pair[0], pair[0]),
    )
    messages: list[ImportedMessage] = []
    previous_node_id: str | None = None
    for order, (_, message) in enumerate(ordered):
        text = message.text.strip()
        if not text:
            continue
        node_id = message.source_message_id or f"{captured.source_conversation_id}-{order}"
        messages.append(
            ImportedMessage(
                source_node_id=node_id,
                parent_source_node_id=previous_node_id,
                source_message_id=node_id,
                role=message.role or "unknown",
                created_at=message.created_at,
                source_order=order,
                is_primary_path=True,
                text=text,
            )
        )
        previous_node_id = node_id
    if not messages:
        return None
    return ImportedConversation(
        source_conversation_id=captured.source_conversation_id,
        title=captured.title.strip() or DEFAULT_TITLE,
        created_at=captured.created_at,
        updated_at=captured.updated_at,
        messages=messages,
    )


def conversation_content_hash(conversation: ImportedConversation) -> str:
    """Mirror the hash `persist_conversation` stores, so re-capture can short-circuit."""
    return stable_hash(
        conversation.source_conversation_id,
        conversation.title,
        *[message.text for message in conversation.messages],
    )


def capture_conversations(
    workspace: Path, settings: Settings, payload: CapturePayload
) -> CaptureResult:
    """Upsert captured conversations. Idempotent: unchanged conversations write nothing."""
    conversations_written = 0
    messages_written = 0
    skipped_unchanged = 0
    with connection(workspace) as conn, conn:
        import_id = ensure_capture_import(conn)
        for captured in payload.conversations:
            conversation = to_imported_conversation(captured)
            if conversation is None:
                continue
            content_hash = conversation_content_hash(conversation)
            existing = conn.execute(
                "SELECT conversation_id, content_hash FROM conversations "
                "WHERE source_conversation_id = ?",
                (conversation.source_conversation_id,),
            ).fetchone()
            if existing is not None and str(existing["content_hash"]) == content_hash:
                skipped_unchanged += 1
                continue
            if existing is not None:
                # The conversation grew or was edited: drop the old turns so stale messages
                # and their passages cannot linger. Passages cascade from messages.
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (int(existing["conversation_id"]),),
                )
            persist_conversation(conn, import_id, conversation, settings)
            # `persist_conversation` only sets import_id on insert, so re-home a conversation
            # that was first seen in an export ZIP and is now maintained by live capture.
            conn.execute(
                "UPDATE conversations SET import_id = ? WHERE source_conversation_id = ?",
                (import_id, conversation.source_conversation_id),
            )
            conversations_written += 1
            messages_written += len(conversation.messages)
        if conversations_written:
            set_index_stale(conn, True)
    return CaptureResult(
        conversations_written=conversations_written,
        messages_written=messages_written,
        skipped_unchanged=skipped_unchanged,
    )
