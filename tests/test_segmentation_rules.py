from __future__ import annotations

from pathlib import Path

from convsearch.config.settings import Settings
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.segmentation.build import rebuild_segments
from convsearch.storage.database import connection


def test_rule_segmentation_assigns_primary_passages(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    count = rebuild_segments(workspace, settings)
    assert count >= 1
    with connection(workspace) as conn:
        unsegmented_primary = conn.execute(
            """
            SELECT count(*) AS count
            FROM passages p
            JOIN messages m ON m.message_id = p.message_id
            WHERE m.is_primary_path = 1 AND p.segment_id IS NULL
            """
        ).fetchone()["count"]
        segment_fts_count = conn.execute("SELECT count(*) AS count FROM segment_fts").fetchone()[
            "count"
        ]
    assert unsegmented_primary == 0
    assert segment_fts_count == count
