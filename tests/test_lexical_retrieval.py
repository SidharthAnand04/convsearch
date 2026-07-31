from __future__ import annotations

import json
import zipfile
from pathlib import Path

from convsearch.config.settings import Settings
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.retrieval.lexical import lexical_search, title_search
from convsearch.storage.database import connection
from tests.conftest import index_with_test_embeddings, make_message


def test_lexical_identifier_query(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    with connection(workspace) as conn:
        hits = lexical_search(conn, "FTS5 FAISS", 10, settings=settings)
    assert hits
    assert hits[0].lexical_score is not None


def _identifier_export_zip(tmp_path: Path, cases: dict[str, str]) -> Path:
    """Build a ChatGPT-style export where each case is its own conversation.

    `cases` maps a short conversation slug to the assistant-message text that should carry a
    dotted/slashed/camelCase identifier verbatim. Each conversation is a linear
    system -> user -> assistant thread with the assistant node as the selected path, so the
    identifier lands on an indexed primary-path passage.
    """
    data = []
    for slug, text in cases.items():
        root = make_message(f"{slug}-root", "system", "", None)
        user = make_message(f"{slug}-user", "user", f"Question about {slug}.", f"{slug}-root")
        assistant = make_message(f"{slug}-asst", "assistant", text, f"{slug}-user")
        root["children"] = [f"{slug}-user"]
        user["children"] = [f"{slug}-asst"]
        data.append(
            {
                "id": f"conv-{slug}",
                "title": f"Identifier {slug}",
                "create_time": 1_700_000_000,
                "update_time": 1_700_000_100,
                "current_node": f"{slug}-asst",
                "mapping": {
                    f"{slug}-root": root,
                    f"{slug}-user": user,
                    f"{slug}-asst": assistant,
                },
            }
        )
    path = tmp_path / "chatgpt-identifiers.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(data))
    return path


_IDENTIFIER_CASES = {
    "file": "The importer reads conversations.json straight from the export archive.",
    "node": "Every message row stores a source_node_id copied from the export.",
    "parent": "Branches are stitched together through the parent_source_node_id link.",
    "model": "Passage embeddings are produced by the BAAI/bge-small-en-v1.5 encoder.",
    "index": "The dense vector store is backed by a faiss IndexFlatIP structure.",
}


def _lexical_texts(workspace: Path, settings: Settings, query: str) -> list[str]:
    with connection(workspace) as conn:
        hits = lexical_search(conn, query, 10, settings=settings)
    return [hit.text for hit in hits]


def test_dotted_and_slashed_identifiers_survive_as_units(
    workspace: Path, settings: Settings, tmp_path: Path
) -> None:
    export = _identifier_export_zip(tmp_path, _IDENTIFIER_CASES)
    import_chatgpt_zip(export, workspace, settings)
    index_with_test_embeddings(workspace, settings)

    # Each identifier must be findable as a whole, not just as one of its split-out word pieces.
    for identifier in (
        "conversations.json",
        "source_node_id",
        "parent_source_node_id",
        "BAAI/bge-small-en-v1.5",
        "IndexFlatIP",
    ):
        texts = _lexical_texts(workspace, settings, identifier)
        assert any(identifier in text for text in texts), f"{identifier!r} was not found as a unit"


def test_identifier_query_does_not_spuriously_match_a_different_identifier(
    workspace: Path, settings: Settings, tmp_path: Path
) -> None:
    export = _identifier_export_zip(tmp_path, _IDENTIFIER_CASES)
    import_chatgpt_zip(export, workspace, settings)
    index_with_test_embeddings(workspace, settings)

    # The phrase-adjacency path must require the FULL identifier's tokens in order: querying
    # the longer parent_source_node_id must not drag in the passage that only mentions the
    # shorter source_node_id (it lacks the leading "parent" adjacency).
    parent_texts = _lexical_texts(workspace, settings, "parent_source_node_id")
    assert any("parent_source_node_id" in text for text in parent_texts)
    assert not any(
        "source_node_id" in text and "parent_source_node_id" not in text for text in parent_texts
    )

    # The model id and the FAISS index type share no tokens, so neither query should surface
    # the other's passage.
    model_texts = _lexical_texts(workspace, settings, "BAAI/bge-small-en-v1.5")
    assert any("BAAI/bge-small-en-v1.5" in text for text in model_texts)
    assert not any("IndexFlatIP" in text for text in model_texts)

    index_texts = _lexical_texts(workspace, settings, "IndexFlatIP")
    assert any("IndexFlatIP" in text for text in index_texts)
    assert not any("BAAI/bge-small-en-v1.5" in text for text in index_texts)


def test_title_search_is_separate_channel(
    workspace: Path, settings: Settings, export_zip: Path
) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    with connection(workspace) as conn:
        hits = title_search(conn, "Conversation Search Architecture", 10)
    assert hits
    assert hits[0].channels == ("title",)
