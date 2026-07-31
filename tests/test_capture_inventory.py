"""Coverage for `convsearch.capture.inventory.list_captures`.

Builds a small hand-seeded database with one live-captured and one imported conversation,
one indexed and one not, one segmented and one not, one with memories and one without, so
every flag and warning can be checked against ground truth rather than a mock.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from convsearch.capture.inventory import list_captures
from convsearch.capture.state import CAPTURE_SOURCE_HASH, CAPTURE_SOURCE_PATH, set_index_stale
from convsearch.config.settings import database_path
from convsearch.storage.database import connect


def _seed(workspace: Path) -> None:
    with closing(connect(database_path(workspace))) as conn, conn:
        # A regular export import, owning a fully-processed conversation.
        export_import_id = conn.execute(
            "INSERT INTO imports(source_path, source_hash, status, warning_count,"
            " metadata_json) VALUES ('export.zip', 'export-hash', 'complete', 0, '{}')"
        ).lastrowid

        # The synthetic live-capture import, owning a bare conversation with no downstream
        # processing at all.
        live_import_id = conn.execute(
            "INSERT INTO imports(source_path, source_hash, status, warning_count,"
            " metadata_json) VALUES (?, ?, 'complete', 0, '{}')",
            (CAPTURE_SOURCE_PATH, CAPTURE_SOURCE_HASH),
        ).lastrowid

        # -- Imported conversation: indexed, segmented, has a memory. --------------------
        imported_cid = conn.execute(
            "INSERT INTO conversations(source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES"
            " ('11111111-1111-1111-1111-111111111111', ?, 'Imported convo', '2024-01-01',"
            " '2024-01-02', 'chash-imported')",
            (export_import_id,),
        ).lastrowid
        imported_mid = conn.execute(
            "INSERT INTO messages(source_message_id, conversation_id, parent_message_id, role,"
            " created_at, source_order, is_primary_path, text, content_hash) VALUES"
            " ('m-imported-1', ?, NULL, 'user', '2024-01-01', 0, 1, 'hello', 'mhash-1')",
            (imported_cid,),
        ).lastrowid
        imported_pid = conn.execute(
            "INSERT INTO passages(conversation_id, message_id, passage_order, text,"
            " start_offset, end_offset, word_count, content_hash) VALUES"
            " (?, ?, 0, 'hello', 0, 5, 1, 'phash-1')",
            (imported_cid, imported_mid),
        ).lastrowid
        conn.execute(
            "INSERT INTO embedding_records(passage_id, vector_id, model_id,"
            " embedding_dimension, content_hash) VALUES (?, 0, 'test-model', 8, 'phash-1')",
            (imported_pid,),
        )
        conn.execute(
            "INSERT INTO segments(conversation_id, segment_order, start_message_id,"
            " end_message_id, title, summary, boundary_confidence, segmentation_version,"
            " content_hash) VALUES (?, 0, ?, ?, 'Segment', 'summary', 1.0, 'v1', 'shash-1')",
            (imported_cid, imported_mid, imported_mid),
        )
        conn.execute(
            "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
            " task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json) VALUES ('decision', 'subject', 'Use SQLite',"
            " 'active', 0.9, NULL, NULL, ?, ?, '2024-01-01', 'v1', 'memhash-1', '{}')",
            (imported_cid, imported_mid),
        )

        # -- Live-captured conversation: no messages processed further, no source id shape. -
        live_cid = conn.execute(
            "INSERT INTO conversations(source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES ('not-a-real-uuid', ?, 'Live convo',"
            " '2024-02-01', '2024-02-02', 'chash-live')",
            (live_import_id,),
        ).lastrowid
        conn.execute(
            "INSERT INTO messages(source_message_id, conversation_id, parent_message_id, role,"
            " created_at, source_order, is_primary_path, text, content_hash) VALUES"
            " ('m-live-1', ?, NULL, 'user', '2024-02-01', 0, 1, 'hi', 'mhash-2')",
            (live_cid,),
        )
        # No passages, no segments, no memories for this one: everything downstream of
        # ingest is untouched, which is exactly the state a fresh capture is in.

        set_index_stale(conn, True)


def test_list_captures_flags_and_counts(workspace: Path) -> None:
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    assert inventory.total == 2
    assert inventory.live_captured == 1
    assert inventory.imported == 1
    assert inventory.stale_index is True

    by_title = {item.title: item for item in inventory.items}

    imported_item = by_title["Imported convo"]
    assert imported_item.source == "export-import"
    assert imported_item.indexed is True
    assert imported_item.segmented is True
    assert imported_item.memories_extracted is True
    assert imported_item.passage_count == 1
    assert imported_item.memory_count == 1
    assert imported_item.warnings == ()
    assert imported_item.source_url == "https://chatgpt.com/c/11111111-1111-1111-1111-111111111111"

    live_item = by_title["Live convo"]
    assert live_item.source == "live-capture"
    assert live_item.indexed is False
    assert live_item.segmented is False
    assert live_item.memories_extracted is False
    assert live_item.passage_count == 0
    assert live_item.memory_count == 0
    # No passages at all, so "not indexed" would be noise; only real gaps are reported.
    assert "not indexed" not in live_item.warnings
    assert "not segmented" in live_item.warnings
    assert "no memories extracted" in live_item.warnings
    # The synthetic id the browser made up is not a ChatGPT conversation id.
    assert live_item.source_url is None


def test_list_captures_ordering_is_newest_first(workspace: Path) -> None:
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    assert [item.title for item in inventory.items] == ["Live convo", "Imported convo"]


def test_list_captures_source_filter(workspace: Path) -> None:
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        live_only = list_captures(conn, source="live")
        import_only = list_captures(conn, source="import")

    assert [item.title for item in live_only.items] == ["Live convo"]
    assert [item.title for item in import_only.items] == ["Imported convo"]
    # The aggregate counts describe the whole inbox, not just the filtered slice.
    assert live_only.live_captured == 1
    assert live_only.imported == 1


def test_list_captures_only_problems(workspace: Path) -> None:
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn, only_problems=True)

    assert [item.title for item in inventory.items] == ["Live convo"]
    assert inventory.total == 1


def test_list_captures_rejects_unknown_source(workspace: Path) -> None:
    with closing(connect(database_path(workspace))) as conn:
        try:
            list_captures(conn, source="bogus")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for an unknown source filter")


def test_list_captures_limit(workspace: Path) -> None:
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn, limit=1)

    assert len(inventory.items) == 1
    assert inventory.total == 2


def test_list_captures_date_source_created(workspace: Path) -> None:
    """Both seeded conversations have a real `created_at`, so both resolve to 'created'."""
    _seed(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    by_title = {item.title: item for item in inventory.items}
    assert by_title["Imported convo"].date_source == "created"
    assert by_title["Imported convo"].captured_at == "2024-01-01"
    assert by_title["Live convo"].date_source == "created"
    assert by_title["Live convo"].captured_at == "2024-02-01"


def _seed_no_created_at(workspace: Path) -> None:
    """A live-captured conversation with `created_at = NULL`, `updated_at` set -- the actual
    shape of every conversation the extension captures (see the bug this module fixes).

    Reuses whichever import already owns `CAPTURE_SOURCE_HASH` (there can only be one --
    `source_hash` is UNIQUE) rather than inserting a second one.
    """
    with closing(connect(database_path(workspace))) as conn, conn:
        row = conn.execute(
            "SELECT import_id FROM imports WHERE source_hash = ?", (CAPTURE_SOURCE_HASH,)
        ).fetchone()
        if row is not None:
            live_import_id = row["import_id"]
        else:
            live_import_id = conn.execute(
                "INSERT INTO imports(source_path, source_hash, status, warning_count,"
                " metadata_json) VALUES (?, ?, 'complete', 0, '{}')",
                (CAPTURE_SOURCE_PATH, CAPTURE_SOURCE_HASH),
            ).lastrowid
        conn.execute(
            "INSERT INTO conversations(source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES ('not-a-real-uuid-2', ?, 'No creation date',"
            " NULL, '2024-03-03', 'chash-no-created')",
            (live_import_id,),
        )


def test_list_captures_date_source_captured(workspace: Path) -> None:
    """`created_at IS NULL` with `updated_at` set resolves to 'captured', not 'unknown'."""
    _seed_no_created_at(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    item = next(i for i in inventory.items if i.title == "No creation date")
    assert item.date_source == "captured"
    assert item.captured_at == "2024-03-03"


def _seed_no_timestamps_at_all(workspace: Path) -> None:
    """A conversation with neither `created_at` nor `updated_at` -- only the import's
    `imported_at` (always populated) is available as a last-resort fallback."""
    with closing(connect(database_path(workspace))) as conn, conn:
        import_id = conn.execute(
            "INSERT INTO imports(source_path, source_hash, status, warning_count,"
            " metadata_json, imported_at) VALUES ('export2.zip', 'export-hash-2', 'complete',"
            " 0, '{}', '2024-04-04')"
        ).lastrowid
        conn.execute(
            "INSERT INTO conversations(source_conversation_id, import_id, title, created_at,"
            " updated_at, content_hash) VALUES ('22222222-2222-2222-2222-222222222222', ?,"
            " 'No timestamps at all', NULL, NULL, 'chash-no-ts')",
            (import_id,),
        )


def test_list_captures_date_source_imported(workspace: Path) -> None:
    """With no `created_at` and no `updated_at`, `captured_at` falls back to the import's
    `imported_at`, labeled 'imported' -- never mislabeled 'captured'."""
    _seed_no_timestamps_at_all(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    item = next(i for i in inventory.items if i.title == "No timestamps at all")
    assert item.date_source == "imported"
    assert item.captured_at == "2024-04-04"


def test_list_captures_ordering_uses_effective_timestamp(workspace: Path) -> None:
    """A conversation with only a capture-time fallback must still sort by that resolved
    date among conversations that have a real `created_at`."""
    _seed(workspace)
    _seed_no_created_at(workspace)
    with closing(connect(database_path(workspace))) as conn:
        inventory = list_captures(conn)

    # Live convo: 2024-02-01, No creation date: 2024-03-03 (captured), Imported convo:
    # 2024-01-01 -- newest first regardless of which field the date came from.
    assert [item.title for item in inventory.items] == [
        "No creation date",
        "Live convo",
        "Imported convo",
    ]
