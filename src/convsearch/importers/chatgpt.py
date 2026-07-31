from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from convsearch.config.settings import Settings
from convsearch.domain.models import ImportedConversation, ImportedMessage
from convsearch.importers.base import ImportParseResult
from convsearch.passages.builder import build_passages_for_message
from convsearch.storage.database import connection
from convsearch.utils import sha256_file, stable_hash


class ImportErrorWithContext(RuntimeError):
    pass


def ensure_safe_zip(zip_path: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise ImportErrorWithContext(f"Input is not a valid ZIP file: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ImportErrorWithContext(f"Unsafe ZIP member path rejected: {info.filename}")


def import_chatgpt_zip(export_zip: Path, workspace: Path, settings: Settings) -> int:
    export_zip = export_zip.resolve()
    ensure_safe_zip(export_zip)
    source_hash = sha256_file(export_zip)
    imports_dir = workspace / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    preserved_zip = imports_dir / f"{source_hash}.zip"
    if not preserved_zip.exists():
        shutil.copy2(export_zip, preserved_zip)

    parsed = parse_chatgpt_zip(export_zip)
    with connection(workspace) as conn:
        existing = conn.execute(
            "SELECT import_id FROM imports WHERE source_hash = ?", (source_hash,)
        ).fetchone()
        if existing is not None:
            return int(existing["import_id"])
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO imports(source_path, source_hash, status, warning_count, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(preserved_zip),
                    source_hash,
                    "complete",
                    len(parsed.warnings),
                    json.dumps({"parser": "chatgpt-conversations-json"}),
                ),
            )
            if cursor.lastrowid is None:
                raise ImportErrorWithContext("Import insert did not return an import_id")
            import_id = int(cursor.lastrowid)
            for context, warning in parsed.warnings:
                conn.execute(
                    "INSERT INTO import_warnings(import_id, context, message) VALUES (?, ?, ?)",
                    (import_id, context, warning),
                )
            for conversation in parsed.conversations:
                persist_conversation(conn, import_id, conversation, settings)
    return import_id


def parse_chatgpt_zip(zip_path: Path) -> ImportParseResult:
    result = ImportParseResult()
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("conversations.json")]
        if not names:
            raise ImportErrorWithContext("ZIP does not contain conversations.json")
        with archive.open(names[0]) as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ImportErrorWithContext("conversations.json is not valid JSON") from exc
    if not isinstance(data, list):
        raise ImportErrorWithContext("conversations.json must contain a list")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            result.warnings.append(
                (f"conversation[{index}]", "Conversation record is not an object")
            )
            continue
        try:
            conversation = parse_conversation(record, index, result.warnings)
        except Exception as exc:
            result.warnings.append((f"conversation[{index}]", f"Skipped malformed record: {exc}"))
            continue
        if conversation is None:
            continue
        result.conversations.append(conversation)
    return result


def parse_conversation(
    record: dict[str, Any], index: int, warnings: list[tuple[str, str]]
) -> ImportedConversation | None:
    source_id = str(record.get("id") or record.get("conversation_id") or f"conversation-{index}")
    title = str(record.get("title") or "Untitled conversation")
    mapping = record.get("mapping")
    if not isinstance(mapping, dict):
        warnings.append((source_id, "Missing or malformed mapping"))
        return None
    primary_ids = primary_path_ids(record, mapping, warnings, source_id)
    messages: list[ImportedMessage] = []
    for order, node_id in enumerate(traverse_mapping(mapping)):
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        message = parse_message_node(node_id, node, order, node_id in primary_ids, warnings)
        if message is not None:
            messages.append(message)
    if not messages:
        warnings.append((source_id, "Conversation contains no searchable messages"))
        return None
    return ImportedConversation(
        source_conversation_id=source_id,
        title=title,
        created_at=format_timestamp(record.get("create_time")),
        updated_at=format_timestamp(record.get("update_time")),
        messages=messages,
    )


def traverse_mapping(mapping: dict[str, Any]) -> list[str]:
    children: dict[str | None, list[str]] = defaultdict(list)
    for node_id, node in mapping.items():
        parent = node.get("parent") if isinstance(node, dict) else None
        children[parent].append(str(node_id))
    for values in children.values():
        values.sort()
    roots = children.get(None) or [
        node_id
        for node_id, node in mapping.items()
        if isinstance(node, dict) and not node.get("parent")
    ]
    ordered: list[str] = []

    def walk(node_id: str) -> None:
        ordered.append(node_id)
        for child_id in children.get(node_id, []):
            walk(child_id)

    for root in sorted(set(roots)):
        walk(str(root))
    remaining = [str(node_id) for node_id in mapping if str(node_id) not in set(ordered)]
    return ordered + remaining


def primary_path_ids(
    record: dict[str, Any],
    mapping: dict[str, Any],
    warnings: list[tuple[str, str]],
    source_id: str,
) -> set[str]:
    current = record.get("current_node")
    if not current:
        leaves = [
            node_id
            for node_id, node in mapping.items()
            if isinstance(node, dict) and not node.get("children")
        ]
        current = sorted(leaves)[-1] if leaves else None
    path: set[str] = set()
    seen: set[str] = set()
    while current:
        current_id = str(current)
        if current_id in seen:
            warnings.append(
                (source_id, f"Cycle detected while following current_node: {current_id}")
            )
            break
        seen.add(current_id)
        node = mapping.get(current_id)
        if not isinstance(node, dict):
            warnings.append((source_id, f"current_node path references missing node: {current_id}"))
            break
        path.add(current_id)
        current = node.get("parent")
    return path


def parse_message_node(
    node_id: str,
    node: dict[str, Any],
    order: int,
    is_primary: bool,
    warnings: list[tuple[str, str]],
) -> ImportedMessage | None:
    raw_message = node.get("message")
    if not isinstance(raw_message, dict):
        return None
    author_raw = raw_message.get("author")
    author: dict[str, Any] = author_raw if isinstance(author_raw, dict) else {}
    role = str(author.get("role") or "unknown")
    text = extract_text(raw_message.get("content"))
    if not text.strip():
        warnings.append((node_id, "Skipped message with no searchable text"))
        return None
    return ImportedMessage(
        source_node_id=node_id,
        parent_source_node_id=str(node["parent"]) if node.get("parent") else None,
        source_message_id=str(raw_message.get("id") or node_id),
        role=role,
        created_at=format_timestamp(raw_message.get("create_time")),
        source_order=order,
        is_primary_path=is_primary,
        text=text.strip(),
    )


def extract_text(content: Any) -> str:
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if isinstance(parts, list):
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    texts.append(value)
        return "\n\n".join(texts)
    text = content.get("text")
    return text if isinstance(text, str) else ""


def format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    if isinstance(value, str):
        return value
    return None


def persist_conversation(
    conn: sqlite3.Connection,
    import_id: int,
    conversation: ImportedConversation,
    settings: Settings,
) -> None:
    content_hash = stable_hash(
        conversation.source_conversation_id,
        conversation.title,
        *[message.text for message in conversation.messages],
    )
    cursor = conn.execute(
        """
        INSERT INTO conversations(
            source_conversation_id, import_id, title, created_at, updated_at, content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_conversation_id) DO UPDATE SET
            title=excluded.title,
            updated_at=excluded.updated_at,
            content_hash=excluded.content_hash
        RETURNING conversation_id
        """,
        (
            conversation.source_conversation_id,
            import_id,
            conversation.title,
            conversation.created_at,
            conversation.updated_at,
            content_hash,
        ),
    )
    conversation_id = int(cursor.fetchone()["conversation_id"])
    node_id_to_database_message_id: dict[str, int] = {}
    for message in conversation.messages:
        message_hash = stable_hash(
            conversation.source_conversation_id,
            message.source_node_id,
            message.source_message_id,
            message.text,
        )
        cursor = conn.execute(
            """
            INSERT INTO messages(
                source_message_id, conversation_id, source_node_id, parent_source_node_id,
                role, created_at, source_order, is_primary_path, text, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_message_id) DO UPDATE SET
                source_node_id=excluded.source_node_id,
                parent_source_node_id=excluded.parent_source_node_id,
                role=excluded.role,
                created_at=excluded.created_at,
                source_order=excluded.source_order,
                is_primary_path=excluded.is_primary_path,
                text=excluded.text,
                content_hash=excluded.content_hash
            RETURNING message_id
            """,
            (
                message.source_message_id,
                conversation_id,
                message.source_node_id,
                message.parent_source_node_id,
                message.role,
                message.created_at,
                message.source_order,
                int(message.is_primary_path),
                message.text,
                message_hash,
            ),
        )
        message_id = int(cursor.fetchone()["message_id"])
        node_id_to_database_message_id[message.source_node_id] = message_id
    for message in conversation.messages:
        message_id = node_id_to_database_message_id[message.source_node_id]
        resolved_parent_message_id = (
            node_id_to_database_message_id.get(message.parent_source_node_id)
            if message.parent_source_node_id
            else None
        )
        conn.execute(
            """
            UPDATE messages
            SET resolved_parent_message_id = ?
            WHERE message_id = ?
            """,
            (resolved_parent_message_id, message_id),
        )
        conn.execute("DELETE FROM passages WHERE message_id = ?", (message_id,))
        passages = build_passages_for_message(
            conversation_id=conversation_id,
            message_id=message_id,
            text=message.text,
            target_words=settings.passage_target_words,
            overlap_words=settings.passage_overlap_words,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO passages(conversation_id, message_id, passage_order, text, "
            "start_offset, end_offset, word_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    passage.conversation_id,
                    passage.message_id,
                    passage.passage_order,
                    passage.text,
                    passage.start_offset,
                    passage.end_offset,
                    passage.word_count,
                    passage.content_hash,
                )
                for passage in passages
            ],
        )


def conversation_json_files(names: Iterable[str]) -> list[str]:
    return [name for name in names if name.endswith("conversations.json")]
