from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from convsearch.cli.app import app, parse_since
from convsearch.config.settings import Settings, database_path
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.storage.database import connect
from tests.conftest import index_with_test_embeddings
from tests.test_database_cli import _stamp_workspace_through

runner = CliRunner()


def test_parse_since_accepts_days_hours_minutes() -> None:
    for value in ("7d", "24h", "30m"):
        result = parse_since(value)
        assert result is not None


def test_parse_since_rejects_bad_input() -> None:
    import typer

    with pytest.raises(typer.BadParameter):
        parse_since("nonsense")


def _seed_project_memories(workspace: Path) -> None:
    """Insert a task memory and a decision memory so tasks/timeline/export have content."""
    with closing(connect(database_path(workspace))) as conn, conn:
        cid = conn.execute("SELECT conversation_id FROM conversations LIMIT 1").fetchone()[0]
        mid = conn.execute(
            "SELECT message_id FROM messages WHERE conversation_id = ? LIMIT 1", (cid,)
        ).fetchone()[0]

        def add_memory(kind: str, subject: str, statement: str, task_state: str | None) -> int:
            cur = conn.execute(
                "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
                " task_state, conversation_id, message_id, created_at, extraction_version,"
                " content_hash, metadata_json)"
                " VALUES (?, ?, ?, 'active', 0.9, 'Alpha', ?, ?, ?, '2024-01-01', 'v1', ?, '{}')",
                (kind, subject, statement, task_state, cid, mid, f"hash-{subject}"),
            )
            new_id = int(cur.lastrowid or 0)
            conn.execute(
                "INSERT INTO memory_fts(rowid, statement, kind, project, status)"
                " VALUES (?, ?, ?, 'Alpha', 'active')",
                (new_id, statement, kind),
            )
            return new_id

        add_memory("task", "task/index", "Build the FAISS index", "open")
        add_memory("decision", "arch/store", "Use SQLite for local storage", None)


@pytest.fixture
def cli_workspace(tmp_path: Path, export_zip: Path, settings: Settings) -> Path:
    workspace = tmp_path / "cli-views"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    _seed_project_memories(workspace)
    return workspace


def test_tasks_list_shows_open_task(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["tasks", "list", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Build the FAISS index" in result.output


def test_tasks_list_rejects_bad_since(cli_workspace: Path) -> None:
    result = runner.invoke(
        app, ["tasks", "list", "--workspace", str(cli_workspace), "--since", "nonsense"]
    )
    assert result.exit_code != 0


def test_tasks_list_empty_workspace_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["tasks", "list", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_timeline_shows_matched_node(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["timeline", "SQLite", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Use SQLite for local storage" in result.output


def test_timeline_empty_workspace_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["timeline", "anything", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_captures_list_shows_imported_conversation(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["captures", "list", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "export-import" in result.output


def test_captures_list_empty_workspace_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["captures", "list", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_projects_export_to_stdout(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["projects", "export", "Alpha", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert result.output.startswith("# Alpha")


def test_projects_export_to_file(cli_workspace: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "alpha.md"
    result = runner.invoke(
        app,
        ["projects", "export", "Alpha", "--workspace", str(cli_workspace), "--out", str(out_file)],
    )
    assert result.exit_code == 0
    assert "Wrote" in result.output
    assert out_file.read_text(encoding="utf-8").startswith("# Alpha")


def test_projects_export_missing_project(cli_workspace: Path) -> None:
    result = runner.invoke(
        app, ["projects", "export", "DoesNotExist", "--workspace", str(cli_workspace)]
    )
    assert result.exit_code == 1


def test_status_shows_summary(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["status", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "convsearch status" in result.output
    assert "workspace" in result.output


def test_status_on_empty_workspace_does_not_crash(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["status", "--workspace", str(workspace)])
    assert result.exit_code == 0


def test_status_empty_workspace_shows_import_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["status", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_status_shows_real_data_unchanged(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["status", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "convsearch import" not in result.output


@pytest.fixture
def imported_workspace(tmp_path: Path, export_zip: Path, settings: Settings) -> Path:
    """A workspace with conversations imported but no memories ever extracted."""
    workspace = tmp_path / "imported"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    import_chatgpt_zip(export_zip, workspace, settings)
    return workspace


def test_status_imported_not_extracted_shows_extract_hint(imported_workspace: Path) -> None:
    result = runner.invoke(app, ["status", "--workspace", str(imported_workspace)])
    assert result.exit_code == 0
    assert "convsearch memories extract" in result.output
    assert "convsearch import" not in result.output


# ---------------------------------------------------------------------------
# status / tasks list / digest on a workspace stuck before migration 009
#
# Regression coverage for the exact bug that shipped broken: migration 009 added
# memories.task_state_changed_at and task_state_history, and list_tasks/_completed_tasks
# started reading them unconditionally. A workspace stamped only through 006 -- i.e. every
# workspace that predates tonight, since nothing auto-applies new migrations to an
# already-initialized one -- raised a raw sqlite3.OperationalError from all three commands
# instead of a remediation. See `require_current_schema` in cli/app.py.
# ---------------------------------------------------------------------------


def _seed_bare_conversation(workspace: Path) -> None:
    """Insert just enough rows that `_empty_workspace_hint` doesn't short-circuit before the
    query under test even runs -- `tasks list`/`digest` bail out early (exit 0, an import
    hint) on a workspace with zero conversations, which would make the stale-schema crash
    this test exists to catch unreachable."""
    with closing(connect(database_path(workspace))) as conn, conn:
        conn.execute(
            "INSERT INTO imports(import_id, source_path, source_hash, status)"
            " VALUES (1, '/tmp/x.zip', 'h1', 'imported')"
        )
        conn.execute(
            "INSERT INTO conversations"
            "(conversation_id, source_conversation_id, import_id, title, content_hash)"
            " VALUES (1, 'src-conv-1', 1, 'Untitled', 'ch1')"
        )
        conn.execute(
            "INSERT INTO messages"
            "(message_id, source_message_id, conversation_id, role,"
            " source_order, is_primary_path, text, content_hash)"
            " VALUES (1, 'msg-1', 1, 'user', 0, 1, 'hello', 'mh1')"
        )


def test_status_on_pre_009_workspace_degrades_instead_of_crashing(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006-status"
    _stamp_workspace_through(workspace, "006_interactions")
    _seed_bare_conversation(workspace)

    result = runner.invoke(app, ["status", "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert result.exception is None
    # The table still renders, with the one unavailable metric named as such.
    assert "convsearch status" in result.output
    assert "open tasks" in result.output
    assert "(needs schema update)" in result.output
    # A leaked `None`/`NaN` from an unavailable metric would be a bug in itself, distinct
    # from the crash this test primarily guards against -- pin both.
    assert "None" not in result.output
    assert "NaN" not in result.output
    collapsed = result.output.replace("\n", "")
    assert f"convsearch migrate -w {workspace}" in collapsed


def test_tasks_list_on_pre_009_workspace_shows_remediation(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006-tasks"
    _stamp_workspace_through(workspace, "006_interactions")
    _seed_bare_conversation(workspace)

    result = runner.invoke(app, ["tasks", "list", "--workspace", str(workspace)])

    assert result.exit_code == 1
    # A clean `typer.Exit(1)` from the guard, not an unhandled OperationalError bubbling up
    # as a raw traceback.
    assert isinstance(result.exception, SystemExit)
    collapsed = result.output.replace("\n", "")
    assert f"convsearch migrate -w {workspace}" in collapsed


def test_digest_on_pre_009_workspace_shows_remediation(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006-digest"
    _stamp_workspace_through(workspace, "006_interactions")
    _seed_bare_conversation(workspace)

    result = runner.invoke(app, ["digest", "--workspace", str(workspace)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    collapsed = result.output.replace("\n", "")
    assert f"convsearch migrate -w {workspace}" in collapsed


def test_status_tasks_list_digest_normal_on_fully_migrated_workspace(
    cli_workspace: Path,
) -> None:
    """Same three commands, same seeded data, on a workspace that has every migration
    applied -- the guard/degradation path above must never trigger here."""
    status_result = runner.invoke(app, ["status", "--workspace", str(cli_workspace)])
    assert status_result.exit_code == 0
    assert "(needs schema update)" not in status_result.output
    assert "needs a schema update" not in status_result.output

    tasks_result = runner.invoke(app, ["tasks", "list", "--workspace", str(cli_workspace)])
    assert tasks_result.exit_code == 0
    assert "needs a schema update" not in tasks_result.output

    digest_result = runner.invoke(app, ["digest", "--workspace", str(cli_workspace)])
    assert digest_result.exit_code == 0
    assert "needs a schema update" not in digest_result.output


# ---------------------------------------------------------------------------
# every workspace-taking command, enumerated -- the invariant that would have caught both
# the migration-007 (`pinned`/`reviewed_at`) and migration-009 (`task_state_changed_at`,
# `task_state_history`) regressions before they shipped. Each command is introspected off
# the live Typer app rather than hardcoded, so a newly added command is covered
# automatically: it must either fail closed with the remediation on a stale-schema
# workspace, or be named on the exemption list below with a reason.
# ---------------------------------------------------------------------------

# Kept in sync with the "No require_current_schema() here on purpose" comments in cli/app.py:
# init (creates/upgrades the workspace), migrate (must open a stale workspace to fix it),
# serve (already auto-migrates on startup), status/doctor (diagnostics that must keep working,
# and already degrade correctly, on a stale workspace).
_SCHEMA_GUARD_EXEMPT_COMMANDS = {
    ("init",),
    ("migrate",),
    ("serve",),
    ("status",),
    ("doctor",),
}


def _iter_leaf_commands(group, prefix=()):  # type: ignore[no-untyped-def]
    """Walk a Typer/Click command group, yielding (path, command) for every leaf command."""
    for name, cmd in group.commands.items():
        path = (*prefix, name)
        if hasattr(cmd, "commands"):
            yield from _iter_leaf_commands(cmd, path)
        else:
            yield path, cmd


def _has_workspace_option(cmd) -> bool:  # type: ignore[no-untyped-def]
    return any(getattr(param, "opts", None) and "--workspace" in param.opts for param in cmd.params)


def _cli_args_for(path: tuple, cmd, workspace: Path) -> list:  # type: ignore[no-untyped-def,type-arg]
    """Build a minimal, type-valid argv for `cmd`: the command path, a dummy value for every
    other required (positional) argument, then --workspace. The dummy values never need to
    resolve to real data -- every command under test must exit before touching them, at the
    schema check that runs immediately after `ensure_workspace()`."""
    args: list = list(path)
    for param in cmd.params:
        if getattr(param, "opts", None) and "--workspace" in param.opts:
            continue
        if type(param).__name__ == "TyperArgument" and param.required:
            type_name = getattr(param.type, "name", "")
            args.append("1" if type_name == "integer" else "dummy")
    args += ["--workspace", str(workspace)]
    return args


def test_every_workspace_command_fails_closed_or_is_exempt_on_stale_schema(
    tmp_path: Path,
) -> None:
    import typer.main

    workspace = tmp_path / "stuck-at-006-enumeration"
    _stamp_workspace_through(workspace, "006_interactions")

    root = typer.main.get_command(app)
    checked = 0
    for path, cmd in _iter_leaf_commands(root):
        if not _has_workspace_option(cmd):
            # e.g. `eval synthetic`, which never opens a workspace at all.
            continue
        if path in _SCHEMA_GUARD_EXEMPT_COMMANDS:
            continue
        checked += 1
        result = runner.invoke(app, _cli_args_for(path, cmd, workspace))
        name = " ".join(path)
        assert result.exit_code == 1, (
            f"convsearch {name} did not fail closed on a stale-schema workspace "
            f"(exit={result.exit_code}); wire it through require_current_schema() or add it "
            f"to the documented exemption list. Output: {result.output!r}"
        )
        collapsed = result.output.replace("\n", "")
        assert f"convsearch migrate -w {workspace}" in collapsed, (
            f"convsearch {name} exited 1 but did not print the schema-update remediation"
        )

    # Meaningless if the walk silently found nothing -- pin a floor so a refactor that breaks
    # command discovery is caught here rather than by every command quietly passing.
    assert checked >= 20


# ---------------------------------------------------------------------------
# memories extract --rebuild
# ---------------------------------------------------------------------------


def _memory_ids_by_subject(workspace: Path) -> dict[str, int]:
    with closing(connect(database_path(workspace))) as conn:
        rows = conn.execute("SELECT subject_key, memory_id FROM memories").fetchall()
    return {row["subject_key"]: row["memory_id"] for row in rows}


def _memory_count(workspace: Path) -> int:
    with closing(connect(database_path(workspace))) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])


def test_memories_extract_without_rebuild_does_not_purge(cli_workspace: Path) -> None:
    before_ids = set(_memory_ids_by_subject(cli_workspace).values())
    result = runner.invoke(app, ["memories", "extract", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Purged" not in result.output
    after_ids = set(_memory_ids_by_subject(cli_workspace).values())
    # Nothing removed -- the seeded memories are untouched (extraction only inserts new
    # candidates, deduped by content hash).
    assert before_ids <= after_ids


def test_memories_extract_rebuild_requires_confirmation_with_counts(cli_workspace: Path) -> None:
    before = _memory_count(cli_workspace)
    result = runner.invoke(
        app, ["memories", "extract", "--workspace", str(cli_workspace), "--rebuild"]
    )
    assert result.exit_code != 0
    # The prompt must name concrete counts, not a vague "are you sure".
    # Counts are pluralised properly ("1 memory" / "N memories"), never "N memory(s)".
    expected_deleted = f"{before} memory" if before == 1 else f"{before} memories"
    assert f"delete {expected_deleted}" in result.output
    assert "0 memories will be kept" in result.output  # none of these are curated
    assert _memory_count(cli_workspace) == before  # declining makes no changes


def test_memories_extract_rebuild_with_yes_purges_and_preserves_pinned(
    cli_workspace: Path,
) -> None:
    ids = _memory_ids_by_subject(cli_workspace)
    task_id = ids["task/index"]
    decision_id = ids["arch/store"]

    pin_result = runner.invoke(
        app, ["memories", "pin", str(decision_id), "--workspace", str(cli_workspace)]
    )
    assert pin_result.exit_code == 0

    result = runner.invoke(
        app,
        ["memories", "extract", "--workspace", str(cli_workspace), "--rebuild", "--yes"],
    )
    assert result.exit_code == 0
    assert "Purged:    1 (preserved 1 memories curated by you" in result.output

    with closing(connect(database_path(cli_workspace))) as conn:
        remaining_ids = {r["memory_id"] for r in conn.execute("SELECT memory_id FROM memories")}
        pinned_row = conn.execute(
            "SELECT pinned FROM memories WHERE memory_id = ?", (decision_id,)
        ).fetchone()

    assert task_id not in remaining_ids
    assert decision_id in remaining_ids
    assert pinned_row["pinned"] == 1

    # No orphaned dependent rows left behind for the purged memory.
    with closing(connect(database_path(cli_workspace))) as conn:
        orphan_evidence = conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id = ?", (task_id,)
        ).fetchone()[0]
        orphan_fts = conn.execute(
            "SELECT COUNT(*) FROM memory_fts WHERE rowid = ?", (task_id,)
        ).fetchone()[0]
    assert orphan_evidence == 0
    assert orphan_fts == 0


# ---------------------------------------------------------------------------
# memories review
# ---------------------------------------------------------------------------


def test_memories_review_empty_workspace_shows_import_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["memories", "review", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output
    assert "Nothing needs review" not in result.output


def test_memories_review_imported_not_extracted_shows_extract_hint(
    imported_workspace: Path,
) -> None:
    result = runner.invoke(app, ["memories", "review", "--workspace", str(imported_workspace)])
    assert result.exit_code == 0
    assert "convsearch memories extract" in result.output
    assert "Nothing needs review" not in result.output


def test_memories_review_with_data_shows_nothing_needs_review(cli_workspace: Path) -> None:
    # cli_workspace's seeded memories are active/non-contested, so the review queue
    # is genuinely empty -- that IS good news here, unlike the two states above.
    result = runner.invoke(app, ["memories", "review", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Nothing needs review." in result.output
    assert "convsearch import" not in result.output
    assert "convsearch memories extract" not in result.output


# ---------------------------------------------------------------------------
# projects list
# ---------------------------------------------------------------------------


def test_projects_list_empty_workspace_shows_import_hint(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["projects", "list", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_projects_list_imported_not_extracted_shows_extract_hint(
    imported_workspace: Path,
) -> None:
    result = runner.invoke(app, ["projects", "list", "--workspace", str(imported_workspace)])
    assert result.exit_code == 0
    assert "convsearch memories extract" in result.output


def test_projects_list_shows_real_data_unchanged(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["projects", "list", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Alpha" in result.output


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------


def test_digest_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    result = runner.invoke(app, ["digest", "--workspace", str(workspace)])
    assert result.exit_code == 0
    assert "convsearch import" in result.output


def test_digest_shows_real_data(cli_workspace: Path) -> None:
    # The seeded memories/conversation are dated 2023-11/2024-01; a wide window guarantees
    # they fall inside it regardless of when the test suite runs.
    result = runner.invoke(app, ["digest", "--workspace", str(cli_workspace), "--since", "3650d"])
    assert result.exit_code == 0
    assert "Build the FAISS index" in result.output
    assert "Use SQLite for local storage" in result.output


def test_digest_caveat_appears_exactly_once(cli_workspace: Path) -> None:
    result = runner.invoke(app, ["digest", "--workspace", str(cli_workspace), "--since", "3650d"])
    assert result.exit_code == 0
    # The provenance phrase must be stated once, not once per section plus the headline.
    assert result.output.count("dated by capture time") <= 1
    assert result.output.count("not creation time") <= 1


def test_digest_rejects_bad_since(cli_workspace: Path) -> None:
    result = runner.invoke(
        app, ["digest", "--workspace", str(cli_workspace), "--since", "nonsense"]
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# memories review / confirm / invalidate / pin
# ---------------------------------------------------------------------------


def _seed_review_memory(
    workspace: Path, *, status: str = "proposed", confidence: float = 0.9
) -> int:
    """Insert a memory that qualifies for the review queue (proposed, or low confidence)."""
    with closing(connect(database_path(workspace))) as conn, conn:
        cid = conn.execute("SELECT conversation_id FROM conversations LIMIT 1").fetchone()[0]
        mid = conn.execute(
            "SELECT message_id FROM messages WHERE conversation_id = ? LIMIT 1", (cid,)
        ).fetchone()[0]
        cur = conn.execute(
            "INSERT INTO memories(kind, subject_key, statement, status, confidence, project,"
            " task_state, conversation_id, message_id, created_at, extraction_version,"
            " content_hash, metadata_json)"
            " VALUES ('decision', 'review/needs-check', 'Needs a human look', ?, ?, 'Alpha',"
            " NULL, ?, ?, '2024-01-01', 'v1', 'hash-review-item', '{}')",
            (status, confidence, cid, mid),
        )
        new_id = int(cur.lastrowid or 0)
        conn.execute(
            "INSERT INTO memory_fts(rowid, statement, kind, project, status)"
            " VALUES (?, 'Needs a human look', 'decision', 'Alpha', ?)",
            (new_id, status),
        )
        return new_id


def test_memories_review_shows_reason(cli_workspace: Path) -> None:
    _seed_review_memory(cli_workspace)
    result = runner.invoke(app, ["memories", "review", "--workspace", str(cli_workspace)])
    assert result.exit_code == 0
    assert "Needs a human look" in result.output
    assert "Why:" in result.output
    assert "never been confirmed" in result.output


def test_memories_confirm_happy_path(cli_workspace: Path) -> None:
    memory_id = _seed_review_memory(cli_workspace)
    result = runner.invoke(
        app, ["memories", "confirm", str(memory_id), "--workspace", str(cli_workspace)]
    )
    assert result.exit_code == 0
    assert "proposed -> active" in result.output


def test_memories_invalidate_requires_confirmation(cli_workspace: Path) -> None:
    memory_id = _seed_review_memory(cli_workspace)
    result = runner.invoke(
        app, ["memories", "invalidate", str(memory_id), "--workspace", str(cli_workspace)]
    )
    assert result.exit_code != 0

    result = runner.invoke(
        app,
        [
            "memories",
            "invalidate",
            str(memory_id),
            "--workspace",
            str(cli_workspace),
            "--yes",
        ],
    )
    assert result.exit_code == 0
    assert "proposed -> invalidated" in result.output


def test_memories_pin_happy_path(cli_workspace: Path) -> None:
    memory_id = _seed_review_memory(cli_workspace)
    result = runner.invoke(
        app, ["memories", "pin", str(memory_id), "--workspace", str(cli_workspace)]
    )
    assert result.exit_code == 0
    assert "pinned False -> True" in result.output

    result = runner.invoke(
        app,
        ["memories", "pin", str(memory_id), "--unpin", "--workspace", str(cli_workspace)],
    )
    assert result.exit_code == 0
    assert "pinned True -> False" in result.output


@pytest.mark.parametrize("action", ["confirm", "invalidate", "pin"])
def test_memories_mutators_unknown_id_clean_error(cli_workspace: Path, action: str) -> None:
    args = ["memories", action, "999999", "--workspace", str(cli_workspace)]
    if action == "invalidate":
        args.append("--yes")
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "Memory not found" in result.output
