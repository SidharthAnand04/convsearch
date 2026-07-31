from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from convsearch.config.settings import Settings
from convsearch.importers.chatgpt import ImportErrorWithContext, import_chatgpt_zip
from convsearch.storage.database import connection


def test_valid_export_import(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    with connection(workspace) as conn:
        assert conn.execute("SELECT count(*) AS count FROM conversations").fetchone()["count"] == 1
        assert conn.execute("SELECT count(*) AS count FROM messages").fetchone()["count"] == 3
        assert conn.execute("SELECT count(*) AS count FROM passages").fetchone()["count"] >= 3


def test_import_stores_timestamps_from_epoch_seconds(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    """conversations.created_at / messages.created_at come from the export's create_time,

    converted from epoch seconds to an ISO-8601 UTC string, not left NULL.
    """
    import_chatgpt_zip(export_zip, workspace, settings)
    with connection(workspace) as conn:
        conversation_created_at = conn.execute(
            "SELECT created_at FROM conversations LIMIT 1"
        ).fetchone()["created_at"]
        message_created_ats = [
            row["created_at"] for row in conn.execute("SELECT created_at FROM messages")
        ]
    expected = datetime.fromtimestamp(1_700_000_000, tz=UTC).isoformat()
    assert conversation_created_at == expected
    assert message_created_ats
    assert all(value == expected for value in message_created_ats)


def test_duplicate_import_is_idempotent(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    first = import_chatgpt_zip(export_zip, workspace, settings)
    second = import_chatgpt_zip(export_zip, workspace, settings)
    assert first == second
    with connection(workspace) as conn:
        assert conn.execute("SELECT count(*) AS count FROM imports").fetchone()["count"] == 1
        assert conn.execute("SELECT count(*) AS count FROM conversations").fetchone()["count"] == 1


def test_missing_optional_fields(workspace: Path, settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "missing.zip"
    data = [{"mapping": {"m1": {"message": {"content": {"parts": ["hello"]}}, "parent": None}}}]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(data))
    import_chatgpt_zip(path, workspace, settings)
    with connection(workspace) as conn:
        assert (
            conn.execute("SELECT title FROM conversations").fetchone()["title"]
            == "Untitled conversation"
        )


def test_malformed_conversation_record_records_warning(
    workspace: Path, settings: Settings, tmp_path: Path
) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps([123, {"id": "x", "mapping": "bad"}]))
    import_chatgpt_zip(path, workspace, settings)
    with connection(workspace) as conn:
        assert (
            conn.execute("SELECT count(*) AS count FROM import_warnings").fetchone()["count"] >= 2
        )


def test_unsafe_zip_path_rejected(workspace: Path, settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../conversations.json", "[]")
    with pytest.raises(ImportErrorWithContext):
        import_chatgpt_zip(path, workspace, settings)


def test_message_tree_primary_path(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    with connection(workspace) as conn:
        rows = conn.execute(
            "SELECT source_message_id, is_primary_path FROM messages ORDER BY source_message_id"
        ).fetchall()
    primary = {row["source_message_id"] for row in rows if row["is_primary_path"]}
    assert {"u1", "a1"}.issubset(primary)
    assert "a2" not in primary


def test_node_ids_and_message_ids_are_stored_separately(
    workspace: Path, settings: Settings, branch_export_zip: Path
) -> None:
    import_chatgpt_zip(branch_export_zip, workspace, settings)
    with connection(workspace) as conn:
        user = conn.execute(
            """
            SELECT message_id, source_node_id, source_message_id, parent_source_node_id,
                   resolved_parent_message_id
            FROM messages
            WHERE source_node_id = ?
            """,
            ("node-user-1",),
        ).fetchone()
        assistant = conn.execute(
            """
            SELECT message_id, source_node_id, source_message_id, parent_source_node_id,
                   resolved_parent_message_id
            FROM messages
            WHERE source_node_id = ?
            """,
            ("node-assistant-1",),
        ).fetchone()
    assert user["source_message_id"] == "message-user-1"
    assert user["parent_source_node_id"] == "node-root"
    assert user["resolved_parent_message_id"] is None
    assert assistant["source_message_id"] == "message-assistant-1"
    assert assistant["parent_source_node_id"] == "node-user-1"
    assert assistant["resolved_parent_message_id"] == user["message_id"]


def test_current_node_missing_and_cycle_warnings(
    workspace: Path, settings: Settings, tmp_path: Path
) -> None:
    missing = {
        "id": "missing-current",
        "current_node": "missing-node",
        "mapping": {"node-user": make_record("node-user", "message-user", "user", "hello", None)},
    }
    cycle_a = make_record("cycle-a", "message-a", "user", "cycle user", "cycle-b")
    cycle_b = make_record("cycle-b", "message-b", "assistant", "cycle assistant", "cycle-a")
    cycle = {
        "id": "cycle-current",
        "current_node": "cycle-a",
        "mapping": {"cycle-a": cycle_a, "cycle-b": cycle_b},
    }
    path = tmp_path / "warnings.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps([missing, cycle]))
    import_chatgpt_zip(path, workspace, settings)
    with connection(workspace) as conn:
        warnings = [
            row["message"] for row in conn.execute("SELECT message FROM import_warnings").fetchall()
        ]
        assert conn.execute("SELECT count(*) AS count FROM messages").fetchone()["count"] == 3
    assert any("missing node" in warning for warning in warnings)
    assert any("Cycle detected" in warning for warning in warnings)


def test_empty_messages_skipped(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    with connection(workspace) as conn:
        assert (
            conn.execute("SELECT count(*) AS count FROM messages WHERE text = ''").fetchone()[
                "count"
            ]
            == 0
        )


def test_transaction_rollback(workspace: Path) -> None:
    with pytest.raises(sqlite3.IntegrityError), connection(workspace) as conn, conn:
        conn.execute(
            """
            INSERT INTO conversations(source_conversation_id, import_id, title, content_hash)
            VALUES ('x', 999, 'x', 'x')
            """
        )
    with connection(workspace) as conn:
        assert conn.execute("SELECT count(*) AS count FROM conversations").fetchone()["count"] == 0


def make_record(
    node_id: str, message_id: str, role: str, text: str, parent: str | None
) -> dict[str, object]:
    return {
        "id": node_id,
        "parent": parent,
        "children": [],
        "message": {
            "id": message_id,
            "author": {"role": role},
            "content": {"parts": [text]},
        },
    }
