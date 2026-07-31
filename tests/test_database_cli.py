from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from convsearch.cli.app import app
from convsearch.config.settings import Settings, database_path
from convsearch.diagnostics.doctor import run_doctor, stats
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.storage.database import (
    apply_pending_migrations,
    connection,
    initialize_database,
    pending_migrations,
    verify_fts5,
)
from convsearch.storage.migrations import MIGRATIONS_DIR, migration_files
from tests.conftest import index_with_test_embeddings


def test_migration_creation(workspace: Path) -> None:
    # Assert against the discovered migration count rather than a hardcoded number, so
    # adding a migration does not require editing this test.
    expected = len(migration_files())
    with connection(workspace) as conn:
        assert (
            conn.execute("SELECT count(*) AS count FROM schema_migrations").fetchone()["count"]
            == expected
        )
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "segments" in tables


def test_fts5_availability(workspace: Path) -> None:
    with connection(workspace) as conn:
        verify_fts5(conn)


def test_referential_integrity(workspace: Path) -> None:
    with connection(workspace) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_doctor_and_stats(workspace: Path, settings: Settings, export_zip: Path) -> None:
    import_chatgpt_zip(export_zip, workspace, settings)
    index_with_test_embeddings(workspace, settings)
    checks = run_doctor(workspace, settings)
    assert any(check.name == "sqlite_fts5" and check.ok for check in checks)
    values = stats(workspace)
    assert values["conversations"] == 1


def test_cli_workflow(tmp_path: Path, export_zip: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "cli-workspace"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    assert (
        runner.invoke(app, ["import", str(export_zip), "--workspace", str(workspace)]).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["index", "--workspace", str(workspace), "--test-embeddings"]).exit_code
        == 0
    )
    search = runner.invoke(
        app,
        [
            "search",
            "local conversation indexes",
            "--workspace",
            str(workspace),
            "--test-embeddings",
            "--explain",
        ],
    )
    assert search.exit_code == 0
    assert "Conversation Search Architecture" in search.output
    assert runner.invoke(app, ["doctor", "--workspace", str(workspace)]).exit_code == 0
    assert runner.invoke(app, ["stats", "--workspace", str(workspace)]).exit_code == 0

    # memories / projects / plan smoke tests
    assert runner.invoke(app, ["memories", "extract", "--workspace", str(workspace)]).exit_code == 0
    assert runner.invoke(app, ["memories", "list", "--workspace", str(workspace)]).exit_code == 0
    assert runner.invoke(app, ["projects", "list", "--workspace", str(workspace)]).exit_code == 0
    assert (
        runner.invoke(
            app,
            [
                "plan",
                "what did we decide",
                "--workspace",
                str(workspace),
                "--test-embeddings",
            ],
        ).exit_code
        == 0
    )


def test_cli_include_branches(tmp_path: Path, branch_export_zip: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "cli-branches"
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
    assert (
        runner.invoke(
            app, ["import", str(branch_export_zip), "--workspace", str(workspace)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(app, ["index", "--workspace", str(workspace), "--test-embeddings"]).exit_code
        == 0
    )
    default = runner.invoke(
        app,
        [
            "search",
            "Pinecone Cloud Vector Enterprise",
            "--workspace",
            str(workspace),
            "--test-embeddings",
        ],
    )
    assert default.exit_code == 0
    assert "Pinecone Cloud Vector Enterprise" not in default.output
    assert "alternate branch" not in default.output

    included = runner.invoke(
        app,
        [
            "search",
            "Pinecone Cloud Vector Enterprise",
            "--workspace",
            str(workspace),
            "--test-embeddings",
            "--include-branches",
            "--explain",
        ],
    )
    assert included.exit_code == 0
    assert "Pinecone Cloud Vector Enterprise" in included.output
    assert "alternate branch" in included.output


def test_old_workspace_applies_message_routing_migration(tmp_path: Path) -> None:
    workspace = tmp_path / "old-workspace"
    (workspace / "database").mkdir(parents=True)
    db_path = database_path(workspace)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.executescript((MIGRATIONS_DIR / "001_initial.sql").read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations(version) VALUES ('001_initial')")
        conn.execute(
            """
            INSERT INTO imports(import_id, source_path, source_hash, status)
            VALUES (1, 'old.zip', 'hash', 'complete')
            """
        )
        conn.execute(
            """
            INSERT INTO conversations(
                conversation_id, source_conversation_id, import_id, title, content_hash
            )
            VALUES (1, 'old-conv', 1, 'Old', 'conv-hash')
            """
        )
        conn.execute(
            """
            INSERT INTO messages(
                message_id, source_message_id, conversation_id, parent_message_id, role,
                source_order, is_primary_path, text, content_hash
            )
            VALUES (1, 'old-message', 1, NULL, 'user', 1, 1, 'old text', 'message-hash')
            """
        )
    initialize_database(workspace)
    initialize_database(workspace)
    with connection(workspace) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        message = conn.execute("SELECT * FROM messages WHERE message_id = 1").fetchone()
        migrations = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert {"source_node_id", "parent_source_node_id", "resolved_parent_message_id"}.issubset(
        columns
    )
    assert message["source_node_id"] == "old-message"
    # An old workspace must end up with every migration applied, in filename order.
    # Derived from the migrations directory so new migrations do not break this test.
    assert [row["version"] for row in migrations] == [path.stem for path in migration_files()]


def _stamp_workspace_through(workspace: Path, last_version_stem: str) -> list[str]:
    """Build a database stamped only through `last_version_stem` (inclusive).

    Mirrors the real bug this covers: `initialize_database` (run only by `convsearch init`)
    is the only thing that ever applied migrations, so a workspace initialized before a
    migration existed is permanently stuck at whatever version it started at. Returns the
    versions NOT applied (i.e. still pending), in order.
    """
    (workspace / "database").mkdir(parents=True)
    db_path = database_path(workspace)
    all_migrations = migration_files()
    cutoff = next(i for i, p in enumerate(all_migrations) if p.stem == last_version_stem) + 1
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        for migration in all_migrations[:cutoff]:
            conn.executescript(migration.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (migration.stem,))
    return [path.stem for path in all_migrations[cutoff:]]


def test_pending_migrations_detected_applied_and_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006"
    expected_pending = _stamp_workspace_through(workspace, "006_interactions")
    assert expected_pending  # this test is meaningless if 006 is already the newest migration

    with connection(workspace) as conn:
        assert pending_migrations(conn) == expected_pending

    applied = apply_pending_migrations(workspace)
    assert applied == expected_pending

    with connection(workspace) as conn:
        assert pending_migrations(conn) == []

    # A second run against an already-current workspace must be a no-op.
    assert apply_pending_migrations(workspace) == []


def test_doctor_warns_on_pending_migrations_then_stops(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006-doctor"
    _stamp_workspace_through(workspace, "006_interactions")
    settings = Settings.default()

    before = next(check for check in run_doctor(workspace, settings) if check.name == "migrations")
    assert before.ok is False
    assert f"convsearch migrate -w {workspace}" in before.detail

    apply_pending_migrations(workspace)

    after = next(check for check in run_doctor(workspace, settings) if check.name == "migrations")
    assert after.ok is True


def test_cli_migrate_applies_pending_and_is_idempotent(tmp_path: Path) -> None:
    workspace = tmp_path / "stuck-at-006-cli"
    expected_pending = _stamp_workspace_through(workspace, "006_interactions")
    runner = CliRunner()

    result = runner.invoke(app, ["migrate", "-w", str(workspace)])
    assert result.exit_code == 0
    collapsed = result.stdout.replace("\n", "")
    for version in expected_pending:
        assert version in collapsed

    result_again = runner.invoke(app, ["migrate", "-w", str(workspace)])
    assert result_again.exit_code == 0
    assert "Already up to date" in result_again.stdout


def test_cli_memories_review_fails_gracefully_on_unmigrated_workspace(tmp_path: Path) -> None:
    # Migration 007 adds `memories.pinned`; a workspace stuck at 006 must not surface a raw
    # sqlite3.OperationalError from a review command -- it must name the fix.
    workspace = tmp_path / "stuck-at-006-review"
    _stamp_workspace_through(workspace, "006_interactions")
    runner = CliRunner()

    result = runner.invoke(app, ["memories", "review", "-w", str(workspace)])
    assert result.exit_code == 1
    # Rich may hard-wrap the line across columns in a narrow test terminal (inserting a bare
    # newline mid-path, no space) -- drop newlines rather than collapsing all whitespace, or
    # the path itself would get corrupted with an inserted space.
    collapsed = result.stdout.replace("\n", "")
    assert f"convsearch migrate -w {workspace}" in collapsed


def test_init_force_preserves_existing_config(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    runner = CliRunner()
    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0

    config_path = workspace / "config.yaml"
    original = config_path.read_text(encoding="utf-8")
    edited = original.replace(
        "embedding_model: BAAI/bge-small-en-v1.5", "embedding_model: custom-hand-edited-model"
    )
    assert edited != original  # sanity: the replacement actually matched something
    config_path.write_text(edited, encoding="utf-8")

    result = runner.invoke(app, ["init", str(workspace), "--force"])
    assert result.exit_code == 0
    assert "Kept existing config.yaml" in result.stdout

    after = config_path.read_text(encoding="utf-8")
    assert "custom-hand-edited-model" in after
    assert "BAAI/bge-small-en-v1.5" not in after
