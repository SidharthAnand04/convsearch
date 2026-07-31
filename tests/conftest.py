from __future__ import annotations

import os

# Force Rich to emit plain text, before anything imports the CLI.
#
# `src/convsearch/cli/app.py` builds a module-level `Console` at import time, so
# this cannot be a fixture -- by the time fixtures run, the Console already
# exists and has decided whether to use colour. Module-level code in conftest
# runs before pytest imports any test module, which is early enough.
#
# Rich honours FORCE_COLOR/COLORTERM even when its output is captured, and it
# auto-highlights numbers. With colour on, a workspace path containing a
# migration name such as `stuck-at-006-review` comes back with ANSI codes
# wrapped around the `006`, so CLI tests asserting on plain substrings -- the
# exact `convsearch migrate -w <path>` remediation line, `pinned False -> True`,
# purge counts -- fail for reasons unrelated to the code.
#
# CI does not set these variables, so this only ever broke on a developer
# machine whose shell exported FORCE_COLOR. Pinning it makes the suite
# deterministic regardless of the shell it is run from.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
for _colour_var in ("FORCE_COLOR", "COLORTERM", "CLICOLOR_FORCE"):
    os.environ.pop(_colour_var, None)

import json  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from convsearch.config.settings import Settings  # noqa: E402
from convsearch.embeddings.sentence_transformers import (  # noqa: E402
    DeterministicEmbeddingProvider,
)
from convsearch.indexes.build import build_indexes  # noqa: E402
from convsearch.storage.database import initialize_database  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(
        embedding_batch_size=4,
        passage_target_words=24,
        passage_overlap_words=5,
        lexical_candidate_limit=20,
        semantic_candidate_limit=20,
    )


@pytest.fixture
def workspace(tmp_path: Path, settings: Settings) -> Path:
    root = tmp_path / "workspace"
    for child in ["database", "imports", "indexes", "cache", "logs"]:
        (root / child).mkdir(parents=True, exist_ok=True)
    settings.write(root)
    initialize_database(root)
    return root


def make_message(
    node_id: str,
    role: str,
    text: str,
    parent: str | None = None,
    message_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "message": {
            "id": message_id or node_id,
            "author": {"role": role},
            "create_time": 1_700_000_000,
            "content": {"content_type": "text", "parts": [text]},
        },
        "parent": parent,
        "children": [],
    }


@pytest.fixture
def export_zip(tmp_path: Path) -> Path:
    root = make_message("root", "system", "", None)
    user = make_message("u1", "user", "Let us keep local conversation indexes in SQLite.", "root")
    assistant = make_message(
        "a1",
        "assistant",
        "The local-first architecture uses FTS5 and FAISS for hybrid search.",
        "u1",
    )
    alt = make_message("a2", "assistant", "An alternate branch with cloud storage.", "u1")
    root["children"] = ["u1"]
    user["children"] = ["a1", "a2"]
    data = [
        {
            "id": "conv-1",
            "title": "Conversation Search Architecture",
            "create_time": 1_700_000_000,
            "update_time": 1_700_000_100,
            "current_node": "a1",
            "mapping": {"root": root, "u1": user, "a1": assistant, "a2": alt},
        }
    ]
    path = tmp_path / "chatgpt-export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(data))
    return path


@pytest.fixture
def branch_export_zip(tmp_path: Path) -> Path:
    root = make_message("node-root", "system", "", None, "message-root")
    user = make_message(
        "node-user-1",
        "user",
        "Which vector storage should this local search tool use?",
        "node-root",
        "message-user-1",
    )
    primary = make_message(
        "node-assistant-1",
        "assistant",
        "Use SQLite and FAISS locally.",
        "node-user-1",
        "message-assistant-1",
    )
    alternate = make_message(
        "node-assistant-2",
        "assistant",
        "Use Pinecone Cloud Vector Enterprise.",
        "node-user-1",
        "message-assistant-2",
    )
    root["children"] = ["node-user-1"]
    user["children"] = ["node-assistant-1", "node-assistant-2"]
    data = [
        {
            "id": "conv-branches",
            "title": "Branch Routing",
            "create_time": 1_700_000_000,
            "update_time": 1_700_000_100,
            "current_node": "node-assistant-1",
            "mapping": {
                "node-root": root,
                "node-user-1": user,
                "node-assistant-1": primary,
                "node-assistant-2": alternate,
            },
        }
    ]
    path = tmp_path / "chatgpt-branches.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(data))
    return path


def index_with_test_embeddings(workspace: Path, settings: Settings) -> None:
    build_indexes(workspace, settings, DeterministicEmbeddingProvider())
