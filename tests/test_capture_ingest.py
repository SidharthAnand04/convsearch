"""Coverage for `convsearch.capture.ingest`, focused on timestamp handling.

The extension currently has no reliable way to read a conversation's true creation time
from the DOM, so it may send `created_at: null`. Ingest must faithfully persist whatever
ISO timestamps it IS given rather than substituting wall-clock "now", and must not choke
when they are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

from convsearch.capture.ingest import capture_conversations, parse_capture_payload
from convsearch.config.settings import Settings
from convsearch.storage.database import connection


def test_capture_ingest_stores_provided_iso_timestamps(workspace: Path, settings: Settings) -> None:
    body = json.dumps(
        {
            "conversations": [
                {
                    "source_conversation_id": "conv-a",
                    "title": "Live capture",
                    "created_at": "2024-05-01T12:00:00.000Z",
                    "updated_at": "2024-05-01T12:05:00.000Z",
                    "messages": [
                        {
                            "source_message_id": "m1",
                            "role": "user",
                            "text": "Let's use gRPC.",
                            "order": 0,
                            "created_at": "2024-05-01T12:00:00.000Z",
                        },
                        {
                            "source_message_id": "m2",
                            "role": "assistant",
                            "text": "Sounds good, we decided to use gRPC.",
                            "order": 1,
                            "created_at": "2024-05-01T12:01:00.000Z",
                        },
                    ],
                }
            ]
        }
    ).encode("utf-8")
    payload = parse_capture_payload(body)

    result = capture_conversations(workspace, settings, payload)
    assert result.conversations_written == 1

    with connection(workspace) as conn:
        conv_row = conn.execute(
            "SELECT created_at, updated_at FROM conversations WHERE source_conversation_id = ?",
            ("conv-a",),
        ).fetchone()
        message_rows = conn.execute(
            "SELECT source_message_id, created_at FROM messages "
            "WHERE conversation_id = (SELECT conversation_id FROM conversations "
            "WHERE source_conversation_id = 'conv-a') ORDER BY source_order"
        ).fetchall()

    assert conv_row["created_at"] == "2024-05-01T12:00:00.000Z"
    assert conv_row["updated_at"] == "2024-05-01T12:05:00.000Z"
    assert [dict(row) for row in message_rows] == [
        {"source_message_id": "m1", "created_at": "2024-05-01T12:00:00.000Z"},
        {"source_message_id": "m2", "created_at": "2024-05-01T12:01:00.000Z"},
    ]


def test_capture_ingest_leaves_timestamps_null_when_not_sent(
    workspace: Path, settings: Settings
) -> None:
    """A capture with no timestamp information (the current real-world extension shape)

    is stored with NULL created_at rather than a fabricated wall-clock value.
    """
    body = json.dumps(
        {
            "conversations": [
                {
                    "source_conversation_id": "conv-b",
                    "title": "Untimed capture",
                    "created_at": None,
                    "updated_at": None,
                    "messages": [
                        {
                            "source_message_id": "m1",
                            "role": "user",
                            "text": "hello",
                            "order": 0,
                            "created_at": None,
                        }
                    ],
                }
            ]
        }
    ).encode("utf-8")
    payload = parse_capture_payload(body)

    capture_conversations(workspace, settings, payload)

    with connection(workspace) as conn:
        conv_row = conn.execute(
            "SELECT created_at FROM conversations WHERE source_conversation_id = ?",
            ("conv-b",),
        ).fetchone()
        message_row = conn.execute(
            "SELECT created_at FROM messages WHERE source_message_id = 'm1'"
        ).fetchone()

    assert conv_row["created_at"] is None
    assert message_row["created_at"] is None
