from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from convsearch.config.settings import Settings, database_path
from convsearch.diagnostics.doctor import run_doctor
from convsearch.diagnostics.doctor import stats as collect_stats
from convsearch.digest.build import parse_duration
from convsearch.embeddings.sentence_transformers import (
    EmbeddingModelError,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from convsearch.importers.chatgpt import import_chatgpt_zip
from convsearch.indexes.build import build_indexes
from convsearch.indexes.locking import IndexLockTimeout
from convsearch.indexes.vector import VectorIndexError
from convsearch.retrieval.service import search_conversations, search_segments
from convsearch.segmentation.build import rebuild_segments
from convsearch.storage.database import initialize_database
from convsearch.utils import format_dated, pluralize

app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
memories_app = typer.Typer(no_args_is_help=True)
projects_app = typer.Typer(no_args_is_help=True)
learn_app = typer.Typer(no_args_is_help=True)
tasks_app = typer.Typer(no_args_is_help=True)
captures_app = typer.Typer(no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(inspect_app, name="inspect")
app.add_typer(memories_app, name="memories")
app.add_typer(projects_app, name="projects")
app.add_typer(learn_app, name="learn")
app.add_typer(tasks_app, name="tasks")
app.add_typer(captures_app, name="captures")
console = Console(width=120)


def parse_since(value: str) -> datetime:
    """Parse a compact duration (`7d`, `24h`, `30m`, `2w`) into a past datetime.

    Thin wrapper over `digest.build.parse_duration` -- the one place duration-string
    parsing lives -- that adapts it to the CLI idiom: a `typer.BadParameter` naming the
    accepted forms instead of a raw `ValueError`, and a past `datetime` (what every other
    `--since`-taking command wants) instead of a bare `timedelta`.
    """
    try:
        delta = parse_duration(value)
    except ValueError as exc:
        raise typer.BadParameter(f"--since must look like 7d, 24h, or 30m (got {value!r})") from exc
    return datetime.now() - delta


def _conversation_count(workspace: Path) -> int:
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        return int(conn.execute("SELECT count(*) FROM conversations").fetchone()[0])


def _memory_count(workspace: Path) -> int:
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        return int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])


def _empty_workspace_hint(workspace: Path, *, check_memories: bool = False) -> str | None:
    """A next-action hint when the workspace has nothing usable yet, or None.

    Bare (`check_memories=False`, the default) this only distinguishes "nothing
    imported" -- the right check for commands that work straight off conversations
    (tasks, captures, digest, timeline). With `check_memories=True` it also catches
    the "imported but never extracted" state: conversations exist, but zero memories
    have been derived from them, so the next action is `memories extract`, not
    `import`. Callers whose usefulness depends on memories (review, status,
    projects) should pass this.
    """
    if _conversation_count(workspace) == 0:
        return "No conversations yet. Run: convsearch import <export.zip> -w " + str(workspace)
    if check_memories and _memory_count(workspace) == 0:
        return "No memories extracted yet. Run: convsearch memories extract -w " + str(workspace)
    return None


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Workspace path")],
    force: Annotated[
        bool, typer.Option("--force", help="Allow a non-empty existing workspace")
    ] = False,
) -> None:
    # No ensure_workspace()/require_current_schema() here on purpose: this command creates
    # (or, via `initialize_database` below, upgrades) the workspace, so there is nothing to
    # check yet -- the database may not even exist.
    if path.exists() and any(path.iterdir()) and not force:
        raise typer.BadParameter("Workspace exists and is not empty. Use --force to continue.")
    for child in ["database", "imports", "indexes", "cache", "logs"]:
        (path / child).mkdir(parents=True, exist_ok=True)
    # `--force` governs reusing a non-empty *workspace directory*; it must never be read as
    # permission to destroy `config.yaml`, the only place hand-edited settings (llm.backend,
    # ollama_host, ...) live. Only write defaults when no config exists yet -- re-running
    # `init --force` on an existing workspace (currently the only documented way to pick up
    # new migrations, before `convsearch migrate` existed) used to silently revert any
    # override back to defaults. `Settings.load` already fills in defaults for fields added
    # since the file was written (pydantic defaults missing keys), so there is no need to
    # rewrite an existing file to "add" new fields -- doing so would risk reordering/dropping
    # whatever the user hand-edited.
    config_path = path / "config.yaml"
    if config_path.exists():
        console.print("Kept existing config.yaml", markup=False)
    else:
        Settings.default().write(path)
    initialize_database(path)
    console.print(f"Initialized workspace at {path}")


@app.command("import")
def import_command(
    export_zip: Annotated[Path, typer.Argument(help="ChatGPT export ZIP")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    import_id = import_chatgpt_zip(export_zip, workspace, settings)
    console.print(f"Imported export as import #{import_id}")


@app.command()
def index(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    test_embeddings: Annotated[
        bool, typer.Option("--test-embeddings", help="Use deterministic local vectors")
    ] = False,
) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    provider: EmbeddingProvider
    try:
        if test_embeddings:
            from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider

            provider = DeterministicEmbeddingProvider()
        else:
            provider = SentenceTransformerEmbeddingProvider(
                settings.embedding_model, settings.embedding_device
            )
        count = build_indexes(workspace, settings, provider)
    except (EmbeddingModelError, IndexLockTimeout, VectorIndexError) as exc:
        # These carry a message that names the fix (another process is already indexing this
        # workspace, the model will not load, the disk is full). A traceback would bury it.
        console.print(f"[red]Indexing failed:[/red] {exc}", markup=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(
            f"[red]Indexing failed:[/red] {exc}\nThe previous index is intact. "
            "This is usually a full disk or a file locked by another program.",
            markup=True,
        )
        raise typer.Exit(code=1) from exc
    # No clear_index_stale() here: build_indexes derives the flag from what is actually
    # embedded, so forcing it False would hide a conversation captured by a live server while
    # this rebuild was running.
    console.print(f"Indexed {count} passages")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    profile: Annotated[str, typer.Option("--profile")] = "balanced",
    show_passages: Annotated[int, typer.Option("--show-passages")] = 2,
    explain: Annotated[bool, typer.Option("--explain")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    test_embeddings: Annotated[bool, typer.Option("--test-embeddings")] = False,
    test_reranker: Annotated[bool, typer.Option("--test-reranker")] = False,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    level: Annotated[str, typer.Option("--level")] = "conversation",
    include_branches: Annotated[
        bool,
        typer.Option(
            "--include-branches",
            help="Include passages from alternate conversation branches",
        ),
    ] = False,
) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    provider: EmbeddingProvider
    if test_embeddings:
        from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider

        provider = DeterministicEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model, settings.embedding_device
        )
    if level not in {"conversation", "segment", "passage"}:
        raise typer.BadParameter("--level must be conversation, segment, or passage")
    if level == "segment":
        segment_results = search_segments(
            workspace,
            query,
            settings,
            limit=limit or settings.final_result_limit,
            include_branches=include_branches,
        )
        render_segment_results(segment_results, explain=explain, debug=debug)
        return
    results = search_conversations(
        workspace,
        query,
        settings,
        provider,
        limit=limit or settings.final_result_limit,
        profile=profile,
        show_passages=show_passages,
        include_branches=include_branches,
        rerank=rerank,
        test_reranker=test_reranker,
    )
    if level == "passage":
        render_passage_results(results, explain=explain, debug=debug)
    else:
        render_results(results, explain=explain, debug=debug)


@app.command()
def migrate(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Apply pending schema migrations to an existing workspace. Safe to run anytime; a
    second run against an up-to-date workspace is a no-op."""
    ensure_workspace(workspace)
    # No require_current_schema() here on purpose: this is the command that FIXES a stale
    # schema, so it must be able to open one.
    from convsearch.storage.database import apply_pending_migrations

    applied = apply_pending_migrations(workspace)
    if not applied:
        console.print("Already up to date -- no pending migrations.", markup=False)
        return
    console.print(
        f"Applied {pluralize(len(applied), 'migration')}: {', '.join(applied)}",
        markup=False,
    )


@app.command()
def doctor(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    load_model: Annotated[bool, typer.Option("--load-model")] = False,
    load_reranker: Annotated[bool, typer.Option("--load-reranker")] = False,
) -> None:
    # No ensure_workspace()/require_current_schema() here on purpose: doctor is a diagnostic
    # command and must keep working on a stale (or even missing) workspace -- it reports
    # pending migrations as a check (see run_doctor's "migrations" entry) rather than
    # refusing to run.
    settings = Settings.load(workspace)
    table = Table(title="convsearch doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in run_doctor(workspace, settings, load_model=load_model):
        table.add_row(check.name, "ok" if check.ok else "fail", check.detail)
    if load_reranker:
        try:
            from convsearch.retrieval.reranking import make_reranker

            make_reranker(settings.reranking)
            table.add_row("reranker_load", "ok", settings.reranking.model)
        except Exception as exc:
            table.add_row("reranker_load", "fail", str(exc))
    console.print(table)


@app.command()
def serve(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    host: Annotated[str, typer.Option("--host", help="Loopback only by default")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8756,
    test_embeddings: Annotated[
        bool, typer.Option("--test-embeddings", help="Use deterministic local vectors")
    ] = False,
    auto_index: Annotated[
        bool,
        typer.Option(
            "--auto-index/--no-auto-index",
            help="Index captured conversations automatically so they become searchable",
        ),
    ] = True,
    auto_index_delay: Annotated[
        float,
        typer.Option("--auto-index-delay", help="Seconds to coalesce captures before indexing"),
    ] = 3.0,
) -> None:
    """Serve a local JSON search API for the browser extension."""
    ensure_workspace(workspace)
    # No require_current_schema() here on purpose: the server already auto-migrates the
    # workspace on startup (see server.app.serve), so a stale schema is not an error state.
    settings = Settings.load(workspace)
    from convsearch.server.app import serve as run_server

    def provider_factory() -> EmbeddingProvider:
        if test_embeddings:
            from convsearch.embeddings.sentence_transformers import (
                DeterministicEmbeddingProvider,
            )

            return DeterministicEmbeddingProvider()
        return SentenceTransformerEmbeddingProvider(
            settings.embedding_model, settings.embedding_device
        )

    run_server(
        workspace,
        settings,
        provider_factory,
        host=host,
        port=port,
        auto_index=auto_index,
        auto_index_delay=auto_index_delay,
    )


@app.command()
def stats(workspace: Annotated[Path, typer.Option("--workspace", "-w")]) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    values = collect_stats(workspace)
    table = Table(title="convsearch stats")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def status(workspace: Annotated[Path, typer.Option("--workspace", "-w")]) -> None:
    """One-screen health summary: the CLI counterpart of the dashboard."""
    ensure_workspace(workspace)
    # No require_current_schema() here on purpose: status is a diagnostic command and must
    # keep working on a stale workspace -- it already degrades correctly (see the open-tasks
    # try/except below) and still prints the migrate hint via run_doctor's "migrations" check.
    settings = Settings.load(workspace)
    from convsearch.capture.inventory import list_captures
    from convsearch.diagnostics.llm_readiness import probe_llm_readiness
    from convsearch.storage.database import connection
    from convsearch.tasks.query import list_tasks

    values = collect_stats(workspace)
    checks = run_doctor(workspace, settings)
    readiness = probe_llm_readiness(settings)
    with connection(workspace) as conn:
        memory_count = int(conn.execute("SELECT count(*) FROM memories").fetchone()[0])
        project_count = int(
            conn.execute(
                "SELECT count(DISTINCT project) FROM memories "
                "WHERE project IS NOT NULL AND project != ''"
            ).fetchone()[0]
        )
        # `status` is the diagnostic command -- it must report everything it CAN compute even
        # when the schema is stale, rather than let one unavailable metric (open tasks, which
        # needs migration 009's task_state_changed_at) take down the whole report. The
        # "Schema update available" line below (driven by run_doctor's own migrations check,
        # independent of this query) still fires and names the fix.
        try:
            open_tasks: int | None = list_tasks(
                conn, state="open", limit=1, include_evidence=False
            ).total_open
        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "no such column" not in message and "no such table" not in message:
                raise
            open_tasks = None
        captures = list_captures(conn, limit=1)

    stale_check = next((c for c in checks if c.name == "stale_vector_index"), None)
    stale = stale_check is not None and not stale_check.ok
    last_capture = None
    if captures.items:
        last_capture = captures.items[0].updated_at or captures.items[0].captured_at

    table = Table(title="convsearch status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("workspace", str(workspace.resolve()))
    table.add_row("conversations", str(values.get("conversations", 0)))
    table.add_row("messages", str(values.get("messages", 0)))
    table.add_row("passages", str(values.get("passages", 0)))
    table.add_row("segments", str(values.get("segments", 0)))
    table.add_row("indexed passages", str(values.get("embedded_passages", 0)))
    table.add_row("stale index", "yes" if stale else "no")
    table.add_row("memories", str(memory_count))
    table.add_row(
        "open tasks",
        str(open_tasks) if open_tasks is not None else "(needs schema update)",
    )
    table.add_row("projects", str(project_count))
    table.add_row("last capture", last_capture or "(none)")
    table.add_row("llm backend", readiness.backend or "(none available)")
    table.add_row("llm ready", "yes" if readiness.ready else "no")
    console.print(table)

    migrations_check = next((c for c in checks if c.name == "migrations"), None)
    if migrations_check is not None and not migrations_check.ok:
        console.print(
            f"\nSchema update available -- run: convsearch migrate -w {workspace}",
            style="yellow",
            markup=False,
        )

    console.print(readiness.summary, markup=False)
    if not readiness.ready and readiness.remediation:
        console.print("\nTo enable the local/cloud model, run:", markup=False)
        for line in readiness.remediation:
            console.print(f"  {line}", markup=False)

    # Surface the single most useful next action, not a wall of hints. Priority: get
    # something imported, then extracted, then indexed -- each is a prerequisite for
    # the next, so whichever is missing first is the one thing worth telling the user.
    next_action = _empty_workspace_hint(workspace, check_memories=True)
    if next_action is None:
        passages = values.get("passages", 0)
        indexed = values.get("embedded_passages", 0)
        if isinstance(passages, int) and isinstance(indexed, int) and passages > 0 and indexed == 0:
            next_action = f"No passages indexed yet. Run: convsearch index -w {workspace}"
    if next_action:
        console.print(f"\n{next_action}", markup=False)


@app.command()
def timeline(
    topic: Annotated[str, typer.Argument(help="Topic to build a decision timeline for")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    project: Annotated[str | None, typer.Option("--project")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 40,
    evidence: Annotated[bool, typer.Option("--evidence")] = False,
) -> None:
    """Build a topic-scoped decision timeline: how an idea changed over time."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    hint = _empty_workspace_hint(workspace)
    if hint:
        console.print(hint)
        return
    from convsearch.storage.database import connection
    from convsearch.timeline.build import build_timeline

    with connection(workspace) as conn:
        result = build_timeline(
            conn, topic, project=project, limit=limit, include_evidence=evidence
        )
    if not result.nodes:
        console.print(f"No memories matched '{topic}'.", markup=False)
        console.print(f"Try: convsearch memories search {topic!r} -w {workspace}", markup=False)
        return
    console.print(f"Timeline: {result.topic}", markup=False)
    first_seen_label = format_dated(result.first_seen, result.first_seen_source or "unknown")
    last_seen_label = format_dated(result.last_seen, result.last_seen_source or "unknown")
    console.print(
        f"Matched {result.matched_count} memories | "
        f"first seen {first_seen_label} | last seen {last_seen_label}"
    )
    console.print(
        f"active={len(result.active)} superseded={len(result.superseded)} "
        f"contested={len(result.contested)} rejected={len(result.rejected)}"
    )
    table = Table()
    table.add_column("date")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("statement")
    table.add_column("reasons")
    for node in result.nodes:
        table.add_row(
            format_dated(node.created_at, node.date_source),
            node.kind,
            node.status,
            trim(node.statement, 70),
            "; ".join(node.reasons),
        )
    console.print(table)
    if evidence:
        console.print("\nEvidence:", markup=False)
        for node in result.nodes:
            if not node.evidence:
                continue
            console.print(f"  [{node.memory_id}] {trim(node.statement, 70)}", markup=False)
            for ev in node.evidence:
                console.print(
                    f'    "{trim(ev.quote, 160)}" '
                    f"— {ev.conversation_title or '(untitled)'} ({ev.timestamp or 'unknown date'})",
                    markup=False,
                )


@app.command()
def digest(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    since: Annotated[
        str, typer.Option("--since", help="Compact duration, e.g. 7d, 24h, 30m, 2w")
    ] = "7d",
    limit_per_section: Annotated[
        int, typer.Option("--limit-per-section", help="Max items shown per section")
    ] = 5,
) -> None:
    """Summarize what changed in the memory system recently: no LLM, fully deterministic."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    hint = _empty_workspace_hint(workspace)
    if hint:
        console.print(hint)
        return
    since_dt = parse_since(since)
    from convsearch.digest.build import build_digest
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        result = build_digest(conn, since=since_dt, limit_per_section=limit_per_section)

    console.print(result.headline, markup=False)
    if result.caveat:
        console.print(result.caveat, style="dim", markup=False)
    if result.is_empty:
        return
    for section in result.sections:
        console.print(f"\n{section.title} ({section.count}):", markup=False)
        for item in section.items:
            console.print(f"  - {item}", markup=False)
        remaining = section.count - len(section.items)
        if remaining > 0:
            console.print(f"  ... ({remaining} more)", markup=False)


@app.command()
def segment(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    rebuild: Annotated[bool, typer.Option("--rebuild")] = False,
    test_embeddings: Annotated[
        bool, typer.Option("--test-embeddings", help="Use deterministic local vectors")
    ] = False,
) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    seg_provider: EmbeddingProvider | None = None
    if settings.segmentation.strategy in ("semantic", "hybrid"):
        if test_embeddings:
            from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider

            seg_provider = DeterministicEmbeddingProvider()
        else:
            seg_provider = SentenceTransformerEmbeddingProvider(
                settings.embedding_model, settings.embedding_device
            )
    count = rebuild_segments(workspace, settings, provider=seg_provider)
    action = "Rebuilt" if rebuild else "Built"
    console.print(f"{action} {count} segments")


@inspect_app.command("segment")
def inspect_segment(
    segment_id: Annotated[int, typer.Argument(help="Segment ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        row = conn.execute(
            """
            SELECT s.segment_id, s.title, s.boundary_confidence, s.metadata_json,
                   c.title AS conversation_title,
                   sm.source_order AS start_order, em.source_order AS end_order
            FROM segments s
            JOIN conversations c ON c.conversation_id = s.conversation_id
            JOIN messages sm ON sm.message_id = s.start_message_id
            JOIN messages em ON em.message_id = s.end_message_id
            WHERE s.segment_id = ?
            """,
            (segment_id,),
        ).fetchone()
        if row is None:
            raise typer.BadParameter("Segment not found")
        messages = conn.execute(
            """
            SELECT role, text, is_primary_path
            FROM messages
            WHERE conversation_id = (
                SELECT conversation_id FROM segments WHERE segment_id = ?
            )
            AND source_order BETWEEN ? AND ?
            ORDER BY source_order
            """,
            (segment_id, row["start_order"], row["end_order"]),
        ).fetchall()
    console.print(f"{row['conversation_title']} / {row['title']}", markup=False)
    console.print(f"Boundary confidence: {float(row['boundary_confidence']):.2f}")
    console.print(f"Reasons: {row['metadata_json']}", markup=False)
    for message in messages:
        branch = "selected path" if bool(message["is_primary_path"]) else "alternate branch"
        console.print(f"[{message['role']} | {branch}]", markup=False)
        console.print(trim(str(message["text"]), 500), markup=False)


@eval_app.command("synthetic")
def eval_synthetic(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("./data/synthetic"),
    run_root: Annotated[Path | None, typer.Option("--run-root")] = None,
    keep_run: Annotated[bool, typer.Option("--keep-run")] = False,
    real_model: Annotated[bool, typer.Option("--real-model")] = False,
    report: Annotated[str, typer.Option("--report")] = "console",
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression/--no-fail-on-regression"),
    ] = True,
) -> None:
    from convsearch.evaluation.models import EvaluationOptions
    from convsearch.evaluation.reporting import print_report
    from convsearch.evaluation.runner import default_run_root, run_synthetic_evaluation

    options = EvaluationOptions(
        data_dir=data_dir,
        run_root=run_root or default_run_root(),
        keep_run=keep_run,
        real_model=real_model,
        report="json" if report == "json" else "console",
        fail_on_regression=fail_on_regression,
    )
    evaluation_report, passed = run_synthetic_evaluation(options)
    print_report(evaluation_report, as_json=options.report == "json")
    if keep_run:
        console.print(f"Run directory: {evaluation_report.run_dir}", markup=False)
    if fail_on_regression and not passed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# memories sub-app
# ---------------------------------------------------------------------------


@memories_app.command("extract")
def memories_extract(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Purge existing memories first, then extract fresh (destructive).",
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm --rebuild's purge without prompting")
    ] = False,
    llm: Annotated[
        bool,
        typer.Option(
            "--llm/--no-llm",
            help="Also run the opt-in LLM extractor after the rules pass, storing its "
            "accepted proposals through the same dedup/supersession path.",
        ),
    ] = False,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="LLM backend for --llm: auto, ollama, or anthropic. Ignored without --llm.",
        ),
    ] = "auto",
) -> None:
    """Extract memories from all imported conversations and store them.

    Without --rebuild: incremental, content-hash deduped, exactly as before -- existing
    memories are left alone and only new candidates are inserted.

    With --rebuild: purges memories first (preserving anything curated by you -- pinned,
    reviewed, manually re-statused, or with a task state you set) so the current extraction
    rules and quality filter apply to ALL data, not just newly imported messages. This
    destroys unreviewed extracted memories, so it requires --yes or an interactive
    confirmation naming the actual counts.
    """
    ensure_workspace(workspace)
    require_current_schema(workspace)
    if backend not in {"auto", "ollama", "anthropic"}:
        raise typer.BadParameter("--backend must be one of: auto, ollama, anthropic")
    from convsearch.memory.store import clear_memories, extract_and_store_memories, preview_purge
    from convsearch.storage.database import connection

    purge_summary = None
    if rebuild:
        with connection(workspace) as conn:
            preview = preview_purge(conn)
        if preview.deleted == 0 and preview.preserved == 0:
            console.print("Nothing to purge: no memories exist yet.")
        elif not yes:
            console.print(
                f"--rebuild will permanently delete "
                f"{pluralize(preview.deleted, 'memory', 'memories')} "
                f"(and their evidence, status history, and relations), then re-extract "
                f"from all conversations. "
                f"{pluralize(preview.preserved, 'memory', 'memories')} will be "
                f"kept because they were curated by you (pinned, reviewed, manually "
                f"re-statused, or with a task state you set) -- these are never purged. "
                f"Re-run with --yes to proceed.",
                markup=False,
            )
            raise typer.Abort()

        with connection(workspace) as conn:
            purge_summary = clear_memories(conn)

    with connection(workspace) as conn:
        summary = extract_and_store_memories(conn)

    llm_note: str | None = None
    llm_summary = None
    if llm:
        from convsearch.config.settings import Settings
        from convsearch.memory.llm_extract import propose_memories
        from convsearch.memory.store import store_extracted_memories

        settings = Settings.load(workspace)
        with connection(workspace) as conn:
            proposal = propose_memories(conn, settings=settings, backend=backend)
            if proposal.accepted:
                llm_summary = store_extracted_memories(conn, proposal.accepted)
        if proposal.llm_calls == 0:
            llm_note = "LLM backend unavailable; kept rules-only results"

    if purge_summary is not None:
        console.print(
            f"Purged:    {purge_summary.deleted} (preserved {purge_summary.preserved} "
            f"memories curated by you, kept as-is)"
        )
    console.print(f"Extracted: {summary.extracted}")
    console.print(f"Inserted:  {summary.inserted}")
    console.print(f"Filtered:  {summary.rejected}")
    if summary.rejected_by_reason:
        for reason, count in sorted(summary.rejected_by_reason.items()):
            console.print(f"  - {reason}: {count}", markup=False)
    console.print(f"Superseded: {summary.superseded}")
    console.print(f"Contested: {summary.contested}")
    console.print(f"Entities:  {summary.entities}")

    if llm:
        accepted = llm_summary.extracted if llm_summary is not None else 0
        inserted = llm_summary.inserted if llm_summary is not None else 0
        console.print(f"LLM accepted: {accepted}")
        console.print(f"LLM inserted: {inserted}")
        if llm_note is not None:
            console.print(llm_note, markup=False)


@memories_app.command("list")
def memories_list(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    status: Annotated[str | None, typer.Option("--status")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 50,
) -> None:
    """List memories with optional filters."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.search import list_memories
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        records = list_memories(conn, kind=kind, status=status, project=project, limit=limit)
    _render_memory_table(records)


@memories_app.command("search")
def memories_search(
    query: Annotated[str, typer.Argument(help="Search query")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    status: Annotated[str | None, typer.Option("--status")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """Search memories using full-text search."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.search import search_memories as _search_memories
    from convsearch.storage.database import connection

    kinds = [kind] if kind else None
    statuses = [status] if status else None
    with connection(workspace) as conn:
        records = _search_memories(
            conn, query, kinds=kinds, statuses=statuses, project=project, limit=limit
        )
    _render_memory_table(records)


@memories_app.command("show")
def memories_show(
    memory_id: Annotated[int, typer.Argument(help="Memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Show detailed information about a single memory."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.search import get_memory
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        record = get_memory(conn, memory_id)
        if record is None:
            console.print(f"Memory {memory_id} not found.", markup=False)
            raise typer.Exit(1)

        console.print(f"Statement: {record.statement}", markup=False)
        console.print(f"Kind:      {record.kind}", markup=False)
        console.print(f"Status:    {record.status}", markup=False)
        console.print(f"Subject:   {record.subject_key}", markup=False)
        console.print(f"Project:   {record.project or '(none)'}", markup=False)
        console.print(f"Confidence: {record.confidence:.2f}", markup=False)
        console.print(f"Conversation: {record.conversation_title or '(unknown)'}", markup=False)
        created_label = format_dated(record.created_at, record.date_source)
        console.print(f"Created:   {created_label}", markup=False)

        if record.evidence:
            console.print("\nEvidence:", markup=False)
            for ev in record.evidence:
                console.print(
                    f"  [{ev.evidence_id}] message_id={ev.message_id} "
                    f"passage_id={ev.passage_id} "
                    f"offsets={ev.start_offset}:{ev.end_offset}",
                    markup=False,
                )
                console.print(f"  Quote: {trim(ev.quote, 200)}", markup=False)

        if record.relations:
            console.print("\nRelations:", markup=False)
            for rel in record.relations:
                console.print(
                    f"  [{rel.direction}] {rel.relation} → "
                    f"memory {rel.other_memory_id}: {trim(rel.other_statement, 120)}",
                    markup=False,
                )

        history_rows = conn.execute(
            """
            SELECT old_status, new_status, reason, changed_at
            FROM memory_status_history
            WHERE memory_id = ?
            ORDER BY history_id
            """,
            (memory_id,),
        ).fetchall()
        if history_rows:
            console.print("\nStatus history:", markup=False)
            for row in history_rows:
                reason_str = f" ({row['reason']})" if row["reason"] else ""
                console.print(
                    f"  {row['changed_at']}: {row['old_status']} → {row['new_status']}{reason_str}",
                    markup=False,
                )


@memories_app.command("review")
def memories_review(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    limit: Annotated[int, typer.Option("--limit")] = 30,
    kind: Annotated[str | None, typer.Option("--kind")] = None,
    project: Annotated[str | None, typer.Option("--project")] = None,
    include_reviewed: Annotated[
        bool, typer.Option("--include-reviewed", help="Include already-reviewed memories")
    ] = False,
) -> None:
    """List memories needing human attention: contested, conflicting, or low-confidence."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.review import build_review_queue
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        queue = build_review_queue(
            conn, limit=limit, kind=kind, project=project, include_reviewed=include_reviewed
        )
    # Only worth distinguishing "genuinely nothing to review" from "there's no data to
    # review yet" once we know the schema itself is healthy -- a stale-schema workspace
    # must still surface via `require_current_schema` above, not this hint.
    hint = _empty_workspace_hint(workspace, check_memories=True)
    if hint:
        console.print(hint)
        return
    console.print(
        f"Pending: {queue.total_pending}  Pinned: {queue.total_pinned}  "
        f"Contested: {queue.total_contested}  Invalidated: {queue.total_invalidated}"
    )
    if not queue.items:
        console.print("Nothing needs review.")
        return
    for item in queue.items:
        console.print(f"\n[{item.memory_id}] {trim(item.statement, 90)}", markup=False)
        console.print(
            f"  kind={item.kind} status={item.status} project={item.project or '(none)'} "
            f"confidence={item.confidence:.2f}",
            markup=False,
        )
        # The whole point of the queue: why this item needs a human, in plain language.
        console.print(f"  Why: {item.review_reason}", markup=False)
        if item.conflicts:
            console.print("  Conflicts with:", markup=False)
            for conflict in item.conflicts:
                console.print(
                    f"    [{conflict.memory_id}] {trim(conflict.statement, 80)} "
                    f"(status={conflict.status})",
                    markup=False,
                )
        if item.superseded_by:
            console.print("  Superseded by:", markup=False)
            for sup in item.superseded_by:
                console.print(f"    [{sup.memory_id}] {trim(sup.statement, 80)}", markup=False)


def _memory_state(conn, memory_id: int) -> tuple[str, bool] | None:  # type: ignore[no-untyped-def]
    row = conn.execute(
        "SELECT status, pinned FROM memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if row is None:
        return None
    return row["status"], bool(row["pinned"])


@memories_app.command("confirm")
def memories_confirm(
    memory_id: Annotated[int, typer.Argument(help="Memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    """Confirm a memory: status -> active, and stamp it as reviewed."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.review import confirm_memory
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        before = _memory_state(conn, memory_id)
        try:
            confirm_memory(conn, memory_id, reason=reason)
        except ValueError as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc
        after = _memory_state(conn, memory_id)
    old_status = before[0] if before else "?"
    new_status = after[0] if after else "?"
    console.print(f"Memory {memory_id}: {old_status} -> {new_status}", markup=False)


@memories_app.command("invalidate")
def memories_invalidate(
    memory_id: Annotated[int, typer.Argument(help="Memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm marking this memory as invalidated")
    ] = False,
) -> None:
    """Mark a memory as wrong: status -> invalidated. Requires --yes."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    if not yes:
        raise typer.Abort()
    from convsearch.memory.review import invalidate_memory
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        before = _memory_state(conn, memory_id)
        try:
            invalidate_memory(conn, memory_id, reason=reason)
        except ValueError as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc
        after = _memory_state(conn, memory_id)
    old_status = before[0] if before else "?"
    new_status = after[0] if after else "?"
    console.print(f"Memory {memory_id}: {old_status} -> {new_status}", markup=False)


@memories_app.command("pin")
def memories_pin(
    memory_id: Annotated[int, typer.Argument(help="Memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    unpin: Annotated[
        bool, typer.Option("--unpin", help="Clear the pin instead of setting it")
    ] = False,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Not persisted -- there is no pin-history table"),
    ] = None,
) -> None:
    """Pin (or with --unpin, unpin) a memory, excluding/including it from the review queue."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.review import set_memory_pinned
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        before = _memory_state(conn, memory_id)
        try:
            set_memory_pinned(conn, memory_id, not unpin, reason=reason)
        except ValueError as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc
        after = _memory_state(conn, memory_id)
    old_pinned = before[1] if before else False
    new_pinned = after[1] if after else False
    console.print(f"Memory {memory_id}: pinned {old_pinned} -> {new_pinned}", markup=False)


def _render_memory_table(records: list) -> None:  # type: ignore[type-arg]
    if not records:
        console.print("No memories found.")
        return
    table = Table()
    table.add_column("id")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("project")
    table.add_column("statement")
    table.add_column("created_at")
    for rec in records:
        table.add_row(
            str(rec.memory_id),
            rec.kind,
            rec.status,
            rec.project or "",
            trim(rec.statement, 80),
            rec.created_at or "",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# projects sub-app
# ---------------------------------------------------------------------------


@projects_app.command("list")
def projects_list(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """List all projects extracted from memories."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    hint = _empty_workspace_hint(workspace, check_memories=True)
    if hint:
        console.print(hint)
        return
    from convsearch.projects.reconstruct import list_projects
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        summaries = list_projects(conn)
    if not summaries:
        # Conversations and memories both exist, but none named a project -- a
        # legitimate outcome, not a sign anything is broken or missing.
        console.print(
            "No projects identified in your memories. This is expected if none of "
            "them were tagged with a project.",
            markup=False,
        )
        return
    table = Table()
    table.add_column("name")
    table.add_column("memories")
    table.add_column("conversations")
    table.add_column("decisions")
    table.add_column("open tasks")
    table.add_column("last activity")
    for s in summaries:
        table.add_row(
            s.name,
            str(s.memory_count),
            str(s.conversation_count),
            str(s.decision_count),
            str(s.open_task_count),
            format_dated(s.last_activity, s.date_source),
        )
    console.print(table)


@projects_app.command("show")
def projects_show(
    name: Annotated[str, typer.Argument(help="Project name")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Show a full project report."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.projects.reconstruct import reconstruct_project
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        report = reconstruct_project(conn, name)
    if report is None:
        console.print(f"Project '{name}' not found.", markup=False)
        raise typer.Exit(1)

    console.print(f"Project: {report.name}", markup=False)
    console.print(f"Summary: {report.summary}", markup=False)
    console.print(f"Evidence count: {report.evidence_count}", markup=False)

    if report.decisions:
        console.print("\nDecisions:", markup=False)
        for item in report.decisions:
            console.print(f"  [{item.status}] {item.statement}", markup=False)
            for ev in item.evidence:
                console.print(
                    f"    Evidence: {trim(ev.quote, 120)} "
                    f"[conv {ev.conversation_id}, msg {ev.message_id}]",
                    markup=False,
                )

    if report.superseded_decisions:
        console.print("\nSuperseded decisions:", markup=False)
        for item in report.superseded_decisions:
            console.print(f"  [{item.status}] {item.statement}", markup=False)

    if report.rejected_alternatives:
        console.print("\nRejected alternatives:", markup=False)
        for alt in report.rejected_alternatives:
            console.print(f"  - {alt}", markup=False)

    if report.open_tasks:
        console.print("\nOpen tasks:", markup=False)
        for item in report.open_tasks:
            console.print(f"  [ ] {item.statement}", markup=False)

    if report.completed_tasks:
        console.print("\nCompleted tasks:", markup=False)
        for item in report.completed_tasks:
            console.print(f"  [x] {item.statement}", markup=False)

    if report.risks:
        console.print("\nRisks:", markup=False)
        for item in report.risks:
            console.print(f"  [!] {item.statement}", markup=False)

    if report.timeline:
        console.print("\nTimeline (compact):", markup=False)
        for entry in report.timeline[:10]:
            date_str = format_dated(entry.created_at, entry.date_source)
            console.print(
                f"  {date_str} [{entry.kind}/{entry.status}] {trim(entry.statement, 80)}",
                markup=False,
            )
        if len(report.timeline) > 10:
            console.print(f"  ... ({len(report.timeline) - 10} more)", markup=False)

    if report.conversations:
        console.print("\nConversations:", markup=False)
        for conv_id, conv_title in report.conversations:
            console.print(f"  [{conv_id}] {conv_title or '(untitled)'}", markup=False)


@projects_app.command("export")
def projects_export(
    name: Annotated[str, typer.Argument(help="Project name")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    out: Annotated[Path | None, typer.Option("--out", help="Write to this file")] = None,
    no_evidence: Annotated[
        bool, typer.Option("--no-evidence", help="Omit the evidence appendix")
    ] = False,
) -> None:
    """Export a project report as a self-contained Markdown document."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.projects.export import render_project_markdown
    from convsearch.projects.reconstruct import reconstruct_project
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        report = reconstruct_project(conn, name)
    if report is None:
        console.print(f"Project '{name}' not found.", markup=False)
        raise typer.Exit(1)

    markdown = render_project_markdown(report, include_evidence=not no_evidence)
    if out is not None:
        data = markdown.encode("utf-8")
        out.write_bytes(data)
        console.print(f"Wrote {len(data)} bytes to {out}")
    else:
        # Written straight to stdout (not through Rich) so long lines are never wrapped and
        # the output stays pipeable byte-for-byte, e.g. `convsearch projects export X | less`.
        typer.echo(markdown, nl=False)


# ---------------------------------------------------------------------------
# tasks sub-app
# ---------------------------------------------------------------------------


@tasks_app.command("list")
def tasks_list(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    state: Annotated[str, typer.Option("--state")] = "open",
    project: Annotated[str | None, typer.Option("--project")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 50,
    since: Annotated[
        str | None, typer.Option("--since", help="Compact duration, e.g. 7d, 24h, 30m")
    ] = None,
    evidence: Annotated[bool, typer.Option("--evidence")] = False,
) -> None:
    """List tasks (memories of kind=task) as an inbox."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    if state not in {"open", "completed", "all"}:
        raise typer.BadParameter("--state must be open, completed, or all")
    since_dt = parse_since(since) if since else None
    hint = _empty_workspace_hint(workspace)
    if hint:
        console.print(hint)
        return
    from convsearch.storage.database import connection
    from convsearch.tasks.query import list_tasks

    with connection(workspace) as conn:
        result = list_tasks(
            conn,
            state=state,
            project=project,
            limit=limit,
            since=since_dt,
            include_evidence=evidence,
        )
    console.print(
        f"Open: {result.total_open}  Completed: {result.total_completed}  "
        f"Projects: {', '.join(result.projects) or '(none)'}"
    )
    if not result.items:
        console.print("No tasks match the given filters.")
        return
    table = Table()
    table.add_column("id")
    table.add_column("state")
    table.add_column("project")
    table.add_column("statement")
    table.add_column("conversation")
    table.add_column("created_at")
    for item in result.items:
        marker = "" if item.has_evidence else " (no evidence)"
        table.add_row(
            str(item.memory_id),
            item.task_state or "",
            item.project or "",
            trim(item.statement, 70) + marker,
            item.conversation_title or "",
            format_dated(item.created_at, item.date_source),
        )
    console.print(table)
    if evidence:
        console.print("\nEvidence:", markup=False)
        for item in result.items:
            if not item.evidence:
                continue
            console.print(f"  [{item.memory_id}] {trim(item.statement, 70)}", markup=False)
            for ev in item.evidence:
                console.print(
                    f'    "{trim(ev.quote, 160)}" '
                    f"— {ev.conversation_title or '(untitled)'} ({ev.timestamp or 'unknown date'})",
                    markup=False,
                )


def _task_state_value(conn, memory_id: int) -> str | None:  # type: ignore[no-untyped-def]
    row = conn.execute(
        "SELECT task_state FROM memories WHERE memory_id = ? AND kind = 'task'", (memory_id,)
    ).fetchone()
    return row["task_state"] if row is not None else None


@tasks_app.command("complete")
def tasks_complete(
    memory_id: Annotated[int, typer.Argument(help="Task memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    """Mark a task complete: task_state -> completed, recording the transition."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.store import set_task_state
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        before = _task_state_value(conn, memory_id)
        try:
            set_task_state(conn, memory_id, "completed", reason=reason)
            conn.commit()
        except ValueError as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc
        after = _task_state_value(conn, memory_id)
    console.print(f"Task {memory_id}: {before or '(none)'} -> {after or '(none)'}", markup=False)


@tasks_app.command("reopen")
def tasks_reopen(
    memory_id: Annotated[int, typer.Argument(help="Task memory ID")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
) -> None:
    """Reopen a task: task_state -> open, recording the transition."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.memory.store import set_task_state
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        before = _task_state_value(conn, memory_id)
        try:
            set_task_state(conn, memory_id, "open", reason=reason)
            conn.commit()
        except ValueError as exc:
            console.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc
        after = _task_state_value(conn, memory_id)
    console.print(f"Task {memory_id}: {before or '(none)'} -> {after or '(none)'}", markup=False)


# ---------------------------------------------------------------------------
# captures sub-app
# ---------------------------------------------------------------------------


@captures_app.command("list")
def captures_list(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    source: Annotated[str, typer.Option("--source")] = "all",
    limit: Annotated[int, typer.Option("--limit")] = 50,
    problems: Annotated[bool, typer.Option("--problems")] = False,
) -> None:
    """List captured conversations and how far each got through the pipeline."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    if source not in {"all", "live", "import"}:
        raise typer.BadParameter("--source must be all, live, or import")
    hint = _empty_workspace_hint(workspace)
    if hint:
        console.print(hint)
        return
    from convsearch.capture.inventory import list_captures
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        inventory = list_captures(conn, source=source, limit=limit, only_problems=problems)
    console.print(
        f"Total: {inventory.total}  Live: {inventory.live_captured}  "
        f"Imported: {inventory.imported}  Not indexed: {inventory.not_indexed}  "
        f"Not segmented: {inventory.not_segmented}  Stale index: {inventory.stale_index}"
    )
    if not inventory.items:
        console.print("No captures match the given filters.")
        return
    table = Table()
    table.add_column("id")
    table.add_column("source")
    table.add_column("title")
    table.add_column("captured")
    table.add_column("messages")
    table.add_column("indexed")
    table.add_column("segmented")
    table.add_column("memories")
    table.add_column("warnings")
    for item in inventory.items:
        table.add_row(
            item.conversation_id,
            item.source,
            trim(item.title, 60),
            format_dated(item.captured_at, item.date_source),
            str(item.message_count),
            "yes" if item.indexed else "no",
            "yes" if item.segmented else "no",
            str(item.memory_count),
            ", ".join(item.warnings),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# learn sub-app (self-improvement loop)
# ---------------------------------------------------------------------------


@learn_app.command("run")
def learn_run(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Skip the LLM; use the deterministic heuristic summary"),
    ] = False,
) -> None:
    """Turn logged interactions into learned preference notes."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    from convsearch.feedback.learn import run_self_improvement
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        summary = run_self_improvement(conn, settings, use_llm=not no_llm)
    console.print(f"Events read:   {summary.events_read}")
    console.print(f"Notes written: {summary.notes_written}")
    console.print(f"Backend:       {summary.backend}:{summary.model}")
    if summary.notes:
        console.print("\nNotes:", markup=False)
        for note in summary.notes:
            console.print(f"  - {note}", markup=False)


@learn_app.command("show")
def learn_show(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """List learned preference notes, newest first."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.feedback.learn import list_learned_preferences
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        rows = list_learned_preferences(conn, limit=limit)
    if not rows:
        console.print("No learned preferences.")
        return
    table = Table(title="learned preferences")
    table.add_column("id")
    table.add_column("weight")
    table.add_column("note")
    table.add_column("created_at")
    for pref_id, note, weight, created_at in rows:
        table.add_row(str(pref_id), f"{weight:.2f}", trim(note, 80), created_at)
    console.print(table)


@learn_app.command("stats")
def learn_stats(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
) -> None:
    """Show interaction-log statistics."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    from convsearch.feedback.store import interaction_stats
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        values = interaction_stats(conn)
    table = Table(title="interaction stats")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


@learn_app.command("clear")
def learn_clear(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm deletion of all interactions")
    ] = False,
) -> None:
    """Delete all logged interactions (privacy/reset)."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    if not yes:
        raise typer.Abort()
    from convsearch.feedback.store import clear_interactions
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        deleted = clear_interactions(conn)
    console.print(f"Deleted {pluralize(deleted, 'interaction')}.")


# ---------------------------------------------------------------------------
# plan command
# ---------------------------------------------------------------------------


@app.command()
def plan(
    query: Annotated[str, typer.Argument(help="Planning query")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    test_embeddings: Annotated[
        bool, typer.Option("--test-embeddings", help="Use deterministic local vectors")
    ] = False,
) -> None:
    """Execute a deterministic query plan and display findings."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    provider: EmbeddingProvider
    if test_embeddings:
        from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider

        provider = DeterministicEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model, settings.embedding_device
        )

    from convsearch.planner.planner import execute_plan
    from convsearch.planner.tools import PlannerContext
    from convsearch.storage.database import connection

    with connection(workspace) as conn:
        ctx = PlannerContext(
            workspace=workspace,
            settings=settings,
            provider=provider,
            conn=conn,
        )
        answer = execute_plan(ctx, query)

    if answer.answer:
        console.print("Answer:", markup=False)
        console.print(answer.answer, markup=False)
        console.print("", markup=False)
    console.print(f"Intent: {answer.intent}", markup=False)
    console.print("\nSteps:", markup=False)
    for step in answer.steps:
        console.print(f"  {step.order}. [{step.tool}] {step.rationale}", markup=False)

    console.print("\nTool calls:", markup=False)
    for call in answer.calls:
        console.print(
            f"  {call.tool}: {pluralize(call.result_count, 'result')} — {call.result_summary}",
            markup=False,
        )

    console.print("\nFindings:", markup=False)
    for line in answer.findings:
        console.print(f"  {line}", markup=False)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about your past conversations")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    limit: Annotated[int, typer.Option("--limit", help="Conversations to retrieve")] = 5,
    passages: Annotated[int, typer.Option("--passages", help="Passages per conversation")] = 3,
    backend: Annotated[str | None, typer.Option("--backend", help="auto|ollama|anthropic")] = None,
    test_embeddings: Annotated[bool, typer.Option("--test-embeddings")] = False,
) -> None:
    """Answer a question using retrieval over your imported conversations (RAG)."""
    ensure_workspace(workspace)
    require_current_schema(workspace)
    settings = Settings.load(workspace)
    if backend is not None:
        if backend not in {"auto", "ollama", "anthropic"}:
            raise typer.BadParameter("--backend must be auto, ollama, or anthropic")
        settings = settings.model_copy(
            update={"llm": settings.llm.model_copy(update={"backend": backend})}
        )
    provider: EmbeddingProvider
    if test_embeddings:
        from convsearch.embeddings.sentence_transformers import DeterministicEmbeddingProvider

        provider = DeterministicEmbeddingProvider()
    else:
        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model, settings.embedding_device
        )

    from convsearch.answer.answer import answer_question

    result = answer_question(
        workspace,
        question,
        settings,
        provider,
        limit=limit,
        passages_per_conversation=passages,
    )
    render_answer(result)


def ensure_workspace(workspace: Path) -> None:
    if not database_path(workspace).exists():
        raise typer.BadParameter(f"Workspace database not found at {database_path(workspace)}")


_SCHEMA_UPDATE_LINE = (
    "this workspace needs a schema update -- run: convsearch migrate -w {workspace}"
)


def require_current_schema(workspace: Path) -> None:
    """Exit 1 with the standard remediation if `workspace` is behind on schema migrations.

    This is the blanket guard called right after `ensure_workspace()` at the top of every
    command that reads or writes the database -- the same place two prior regressions
    (migration 007's `pinned`/`reviewed_at`, migration 009's `task_state_changed_at` and
    `task_state_history`) each shipped broken, because safety depended on the author
    remembering to hand-wrap the affected command. A stale workspace used to raise a raw
    `sqlite3.OperationalError: no such column/table: ...` from deep inside whatever module
    happened to touch the new column first. Checking `pending_migrations()` once, up front,
    makes "does this command need a current schema" the default answer instead of an opt-in.

    Cheap by construction: one query against `schema_migrations` (see
    `storage.database.pending_migrations`), never per-column introspection.

    Exempted on purpose, each with its own reason at the call site (or lack of one):
    `init` (creates/upgrades the workspace -- nothing to check yet), `migrate` (must open a
    stale workspace in order to fix it), `serve` (already auto-migrates on startup), and
    `status`/`doctor` (diagnostic commands that must keep working, and already degrade
    correctly, on a stale workspace -- exiting 1 here would remove the only way a user can
    see what is wrong).
    """
    from convsearch.storage.database import connection, pending_migrations

    with connection(workspace) as conn:
        pending = pending_migrations(conn)
    if pending:
        console.print(_SCHEMA_UPDATE_LINE.format(workspace=workspace), markup=False)
        raise typer.Exit(1)


def render_results(results, *, explain: bool, debug: bool) -> None:  # type: ignore[no-untyped-def]
    if not results:
        console.print("No results.")
        return
    for index, result in enumerate(results, start=1):
        title = f"{index}. {result.title}"
        if debug:
            title += f" (conversation_id={result.conversation_id})"
        body = [f"Score: {result.score:.4f}"]
        if result.created_at:
            body.append(f"Date: {result.created_at[:10]}")
        channels = sorted({channel for hit in result.best_passages for channel in hit.channels})
        if channels:
            body.append("Match reasons: " + ", ".join(channels))
        body.append("")
        body.append("Supporting passages:")
        for hit in result.best_passages:
            branch = "selected path" if hit.is_primary_path else "alternate branch"
            prefix = f"[{hit.role} | {branch}]"
            if debug:
                prefix += f" passage_id={hit.passage_id}"
            body.append(prefix)
            body.append(f'"{trim(hit.text)}"')
            if explain:
                body.append(
                    "lexical_rank="
                    f"{hit.lexical_rank} semantic_rank={hit.semantic_rank} "
                    f"title_rank={hit.title_rank} reranker_rank={hit.reranker_rank} "
                    f"lexical_score={hit.lexical_score} semantic_score={hit.semantic_score} "
                    f"title_score={hit.title_score} reranker_score={hit.reranker_score} "
                    f"fused={hit.fused_score:.4f} final={hit.final_score} "
                    f"channels={','.join(hit.channels)} "
                    f"branch={'primary' if hit.is_primary_path else 'alternate'}"
                )
                if hit.segment_title:
                    body.append(f"segment={hit.segment_title}")
            body.append("")
        if explain:
            from convsearch.retrieval.explain import build_reason

            body.append(
                f"conversation_score={result.score:.4f} "
                f"distinct_messages={result.distinct_message_count}"
            )
            body.append(build_reason(result))
        console.print(title, markup=False)
        console.print("\n".join(body), markup=False)


def render_segment_results(results, *, explain: bool, debug: bool) -> None:  # type: ignore[no-untyped-def]
    if not results:
        console.print("No results.")
        return
    for index, result in enumerate(results, start=1):
        title = result.title or "Untitled segment"
        prefix = f"{index}. {title}"
        if debug:
            prefix += f" (segment_id={result.segment_id})"
        console.print(prefix, markup=False)
        console.print(f"Conversation: {result.conversation_title}", markup=False)
        console.print(f"Score: {result.score:.4f}", markup=False)
        for hit in result.best_passages:
            branch = "selected path" if hit.is_primary_path else "alternate branch"
            console.print(f"[{hit.role} | {branch}]", markup=False)
            console.print(f'"{trim(hit.text)}"', markup=False)
            if explain:
                console.print(
                    f"lexical_score={hit.lexical_score} title_score={hit.title_score} "
                    f"channels={','.join(hit.channels)} branch={branch}",
                    markup=False,
                )


def render_passage_results(results, *, explain: bool, debug: bool) -> None:  # type: ignore[no-untyped-def]
    hits = [hit for result in results for hit in result.best_passages]
    hits.sort(
        key=lambda hit: hit.final_score if hit.final_score is not None else hit.fused_score,
        reverse=True,
    )
    if not hits:
        console.print("No results.")
        return
    for index, hit in enumerate(hits, start=1):
        title = f"{index}. {hit.title}"
        if debug:
            title += f" (passage_id={hit.passage_id})"
        console.print(title, markup=False)
        branch = "selected path" if hit.is_primary_path else "alternate branch"
        segment = f" | {hit.segment_title}" if hit.segment_title else ""
        console.print(f"[{hit.role} | {branch}{segment}]", markup=False)
        console.print(f'"{trim(hit.text)}"', markup=False)
        if explain:
            console.print(
                f"lexical_score={hit.lexical_score} title_score={hit.title_score} "
                f"fused={hit.fused_score:.4f} final={hit.final_score} "
                f"channels={','.join(hit.channels)}",
                markup=False,
            )


def render_answer(result) -> None:  # type: ignore[no-untyped-def]
    console.print(result.answer, markup=False)
    if result.sources:
        console.print("")
        console.print("Sources:", markup=False)
        for source in result.sources:
            date = source.date or "unknown date"
            console.print(
                f"[{source.index}] {source.title} ({date}) [{source.role}]",
                markup=False,
            )
            console.print(f"    {trim(source.quote)}", markup=False)
    console.print(f"answered by {result.backend}:{result.model}", style="dim", markup=False)


def trim(text: str, limit: int = 420) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."
