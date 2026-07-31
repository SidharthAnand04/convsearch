"""Startup catch-up scheduling for the server's AutoIndexer.

The stale-index flag is a live-capture signal; a CLI `import` that was never followed by an
`index` leaves passages with no embedding while that flag stays False. These tests pin the
behaviour that such a workspace still gets an indexing pass scheduled the moment the server
starts, and that a fully-indexed, non-stale workspace does NOT.
"""

from __future__ import annotations

from pathlib import Path

from convsearch.capture.state import read_index_stale, set_index_stale
from convsearch.config.settings import Settings
from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.indexes.build import _pending_passage_count, build_indexes
from convsearch.server.app import AutoIndexer
from convsearch.storage.database import connection


def _make_indexer(workspace: Path, settings: Settings) -> AutoIndexer:
    # A long delay keeps a scheduled pass from actually firing during the test: we only assert
    # that one was queued (`busy`), never run it, so the provider factory is never invoked.
    return AutoIndexer(
        workspace,
        settings,
        lambda: DeterministicEmbeddingProvider(),
        delay=1000.0,
    )


def test_start_schedules_pass_for_imported_but_unindexed_workspace(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    # Imported, never indexed: passages exist, nothing is embedded, and the stale flag was
    # never set. This is the common state after a bare CLI `import`.
    with connection(workspace) as conn, conn:
        set_index_stale(conn, False)
    with connection(workspace) as conn:
        assert read_index_stale(conn) is False
        assert _pending_passage_count(conn, settings.embedding_model) > 0

    indexer = _make_indexer(workspace, settings)
    try:
        indexer.start()
        # Scheduled purely off the pending-passage count, with the stale flag still False.
        assert indexer.busy is True
    finally:
        indexer.stop()


def test_start_does_not_schedule_for_fully_indexed_non_stale_workspace(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    # Index with the deterministic provider and point the indexer's settings at that same
    # model id, so `_pending_passage_count` sees zero work left.
    provider = DeterministicEmbeddingProvider()
    build_indexes(workspace, settings, provider)
    indexer_settings = settings.model_copy(update={"embedding_model": provider.model_id})
    with connection(workspace) as conn, conn:
        set_index_stale(conn, False)
    with connection(workspace) as conn:
        assert _pending_passage_count(conn, indexer_settings.embedding_model) == 0

    indexer = _make_indexer(workspace, indexer_settings)
    try:
        indexer.start()
        assert indexer.busy is False
    finally:
        indexer.stop()
