"""Local-only HTTP JSON API over the existing retrieval service.

The server exists so browser clients (the Chrome extension in `extension/`) can query a
workspace. It intentionally uses only the standard library, binds to the loopback
interface, and never calls a cloud service.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

from convsearch.answer.answer import answer_question
from convsearch.capture.ingest import (
    MAX_CAPTURE_BYTES,
    CaptureValidationError,
    capture_conversations,
    parse_capture_payload,
)
from convsearch.capture.inventory import list_captures
from convsearch.capture.state import (
    count_captured_conversations,
    read_index_stale,
    set_index_stale,
)
from convsearch.config.settings import Settings, database_path, faiss_index_path, vector_map_path
from convsearch.diagnostics.doctor import run_doctor
from convsearch.diagnostics.llm_readiness import probe_llm_readiness
from convsearch.domain.models import ConversationResult, PassageHit
from convsearch.embeddings.sentence_transformers import EmbeddingModelError, EmbeddingProvider
from convsearch.feedback.learn import list_learned_preferences, run_self_improvement
from convsearch.feedback.models import InteractionEvent
from convsearch.feedback.store import (
    apply_click_boost,
    interaction_stats,
    popular_queries,
    recent_queries,
    record_event,
)
from convsearch.indexes.build import _pending_passage_count, build_indexes, update_indexes
from convsearch.indexes.locking import IndexLockTimeout
from convsearch.indexes.vector import VectorIndexError
from convsearch.memory.review import (
    _conflict_ids,
    _load_conflicts_batch,
    _load_evidence_batch,
    _load_superseded_by_batch,
    _review_reason,
    build_review_queue,
    confirm_memory,
    invalidate_memory,
    set_memory_pinned,
)
from convsearch.memory.search import get_memory, list_memories, search_memories
from convsearch.memory.store import set_task_state
from convsearch.planner.planner import execute_plan
from convsearch.planner.tools import PlannerContext
from convsearch.projects.export import render_project_markdown
from convsearch.projects.reconstruct import list_projects, reconstruct_project
from convsearch.retrieval.service import search_conversations, search_segments
from convsearch.server import serializers
from convsearch.storage.database import apply_pending_migrations, connection
from convsearch.tasks.query import get_task, list_tasks
from convsearch.timeline.build import build_timeline

_SINCE_RE = re.compile(r"^(\d+)(d|h|m)$")

# Sentinel returned by `_optional_reason` when validation already answered the request,
# distinct from `None` (a valid, absent `reason`).
_INVALID = object()

ALLOWED_ORIGIN_PREFIXES = ("chrome-extension://", "moz-extension://")
ALLOWED_ORIGINS = frozenset(
    {
        "https://chatgpt.com",
        "https://chat.openai.com",
    }
)


def _allowed_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    if origin in ALLOWED_ORIGINS or origin.startswith(ALLOWED_ORIGIN_PREFIXES):
        return origin
    return None


def _passage_payload(hit: PassageHit) -> dict[str, Any]:
    return {
        "passage_id": hit.passage_id,
        "message_id": hit.message_id,
        "role": hit.role,
        "text": hit.text,
        "created_at": hit.created_at,
        "is_primary_path": hit.is_primary_path,
        "branch": "selected path" if hit.is_primary_path else "alternate branch",
        "segment_id": hit.segment_id,
        "segment_title": hit.segment_title,
        "channels": list(hit.channels),
        "score": hit.final_score if hit.final_score is not None else hit.fused_score,
    }


def _source_ids(workspace: Path, conversation_ids: list[int]) -> dict[int, str]:
    """Map local conversation rows back to ChatGPT ids so the UI can deep-link."""
    if not conversation_ids:
        return {}
    placeholders = ",".join("?" * len(conversation_ids))
    with connection(workspace) as conn:
        rows = conn.execute(
            f"SELECT conversation_id, source_conversation_id FROM conversations "
            f"WHERE conversation_id IN ({placeholders})",
            conversation_ids,
        ).fetchall()
    return {row["conversation_id"]: row["source_conversation_id"] for row in rows}


def _result_payload(results: list[ConversationResult], source_ids: dict[int, str]) -> Any:
    payload = []
    for result in results:
        source_id = source_ids.get(result.conversation_id)
        payload.append(
            {
                "conversation_id": result.conversation_id,
                "source_conversation_id": source_id,
                "url": f"https://chatgpt.com/c/{source_id}" if source_id else None,
                "title": result.title,
                "created_at": result.created_at,
                "updated_at": result.updated_at,
                "score": result.score,
                "distinct_message_count": result.distinct_message_count,
                "features": result.features,
                "passages": [_passage_payload(hit) for hit in result.best_passages],
            }
        )
    return payload


def _workspace_status(workspace: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "workspace": str(workspace.resolve()),
        "database": database_path(workspace).exists(),
        "indexed": vector_map_path(workspace).exists(),
        "conversations": 0,
        "messages": 0,
        "captured_conversations": 0,
        "stale_index": False,
    }
    if not status["database"]:
        return status
    try:
        with connection(workspace) as conn:
            status["conversations"] = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[
                0
            ]
            status["messages"] = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            status["captured_conversations"] = count_captured_conversations(conn)
            status["stale_index"] = read_index_stale(conn)
    except sqlite3.DatabaseError:
        pass
    return status


def _workspace_problem(workspace: Path) -> str | None:
    """Why this workspace cannot be served right now, phrased for the user, or None.

    A workspace can vanish underneath a running server — moved, renamed, deleted, or on an
    unmounted drive. Every route would then fail deep inside sqlite3 with "unable to open
    database file" and a 500, which tells the user nothing about what happened or where.
    """
    db_path = database_path(workspace)
    if not db_path.exists():
        return (
            f"the workspace database is gone: {db_path} no longer exists. The workspace was "
            "moved, renamed or deleted while the server was running. Restore it, or stop the "
            "server and restart it with `convsearch serve -w <workspace>`."
        )
    return None


def _is_loopback_host(host: str) -> bool:
    """Whether a bind address is loopback-only, for the privacy report."""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not a literal IP (e.g. a hostname or "0.0.0.0" alias) -- treat as non-loopback
        # since we cannot prove it is confined to this machine.
        return False


def _privacy_counts(workspace: Path) -> dict[str, int]:
    counts: dict[str, int] = {"conversations": 0, "messages": 0, "memories": 0}
    if not database_path(workspace).exists():
        return counts
    try:
        with connection(workspace) as conn:
            counts["conversations"] = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[
                0
            ]
            counts["messages"] = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            counts["memories"] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    except sqlite3.DatabaseError:
        pass
    return counts


def _first_int(values: list[str] | None, default: int, *, low: int, high: int) -> int:
    if not values:
        return default
    try:
        return max(low, min(high, int(values[0])))
    except ValueError:
        return default


def _first_bool(values: list[str] | None, default: bool) -> bool:
    if not values:
        return default
    return values[0].lower() in {"1", "true", "yes", "on"}


def _parse_since(value: str) -> datetime | None:
    """Parse a compact duration (`7d`, `24h`, `30m`) into a past datetime, or None if invalid."""
    match = _SINCE_RE.match(value.strip())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        delta = timedelta(days=amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    else:
        delta = timedelta(minutes=amount)
    return datetime.now() - delta


class AutoIndexer:
    """Keeps the vector index current after captures, off the request thread.

    A capture arrives while the user is browsing, so `POST /capture` must return at once.
    Indexing is therefore deferred to a single worker thread and debounced: opening several
    conversations in a row coalesces into one pass rather than one pass each.
    """

    def __init__(
        self,
        workspace: Path,
        settings: Settings,
        provider_factory: Callable[[], EmbeddingProvider],
        lock: threading.Lock | None = None,
        *,
        delay: float = 3.0,
    ) -> None:
        self._workspace = workspace
        self._settings = settings
        self._provider_factory = provider_factory
        self.lock = lock or threading.Lock()
        self._delay = delay
        self._state = threading.Condition()
        self._due_at: float | None = None
        self._running = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        # Constructing a SentenceTransformer provider LOADS the model from disk (~2.7s
        # warm, ~13s cold). Building one per pass dominated the cost of indexing a couple
        # of passages, so it is created once and reused. `provider()` is only ever called
        # while holding `lock`, which is also what keeps the model single-threaded.
        self._provider: EmbeddingProvider | None = None

    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = self._provider_factory()
        return self._provider

    @property
    def busy(self) -> bool:
        """True when a pass is running or one is scheduled."""
        with self._state:
            return self._running or self._due_at is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="convsearch-autoindex", daemon=True)
        self._thread.start()
        if self._stale_on_startup():
            # The previous process may have died between a capture and its debounced pass —
            # a crash, an update, a reboot. Without this the workspace stays stale until the
            # *next* capture arrives, so a user who just reopens the popup and searches gets
            # an incomplete index and nothing ever fixes it on its own.
            print("[convsearch] index is stale at startup: scheduling a catch-up pass")
            self.schedule()
        else:
            # The stale flag is a live-capture signal; a CLI `import` that was never followed by
            # `index` leaves passages with no embedding while the flag stays False, so the stale
            # check above misses the most common "nothing is searchable yet" case. Count the
            # unindexed passages directly (same query update_indexes uses to size its work) and
            # schedule a catch-up when any exist, so opening the popup after a bare import is
            # enough to make the archive searchable without a manual reindex.
            pending = self._pending_on_startup()
            if pending > 0:
                print(
                    f"[convsearch] {pending} unindexed passage(s) at startup: "
                    f"scheduling a catch-up pass"
                )
                self.schedule()

    def _stale_on_startup(self) -> bool:
        try:
            with connection(self._workspace) as conn:
                return read_index_stale(conn)
        except Exception as exc:
            print(f"[convsearch] could not read the stale-index flag: {type(exc).__name__}: {exc}")
            return False

    def _pending_on_startup(self) -> int:
        """Unindexed passages for the configured model, or 0 if the count cannot be taken.

        Reuses `indexes.build._pending_passage_count`, the same query `update_indexes` /
        `_sync_stale_flag` use to size the work, rather than inventing a second notion of
        "pending". Must never raise: like `_stale_on_startup`, a failure here degrades to
        "assume nothing pending" instead of taking down server startup.
        """
        try:
            with connection(self._workspace) as conn:
                return _pending_passage_count(conn, self._settings.embedding_model)
        except Exception as exc:
            print(f"[convsearch] could not count unindexed passages: {type(exc).__name__}: {exc}")
            return 0

    def stop(self) -> None:
        with self._state:
            self._stopping = True
            self._state.notify_all()

    def schedule(self) -> None:
        """Ask for an indexing pass, pushing back any already-pending one."""
        with self._state:
            self._due_at = time.monotonic() + self._delay
            self._state.notify_all()

    def _loop(self) -> None:
        while True:
            with self._state:
                while not self._stopping and self._due_at is None:
                    self._state.wait()
                if self._stopping:
                    return
                assert self._due_at is not None
                remaining = self._due_at - time.monotonic()
                if remaining > 0:
                    # A capture arriving during this wait moves `_due_at` and we re-check,
                    # which is what makes the debounce coalesce.
                    self._state.wait(remaining)
                    continue
                self._due_at = None
                self._running = True
            try:
                self._run_once()
            except Exception as exc:  # a failed pass must never kill the server
                print(f"[convsearch] auto-index failed: {type(exc).__name__}: {exc}")
                # Leave the workspace marked stale so the popup can offer a manual rebuild.
                try:
                    with connection(self._workspace) as conn, conn:
                        set_index_stale(conn, True)
                except Exception as flag_exc:
                    # The workspace itself is unreachable (deleted, unmounted). Say so:
                    # swallowing this silently hid the actual problem behind a stale flag.
                    print(
                        f"[convsearch] could not mark the index stale: "
                        f"{type(flag_exc).__name__}: {flag_exc}"
                    )
            finally:
                with self._state:
                    self._running = False

    def _run_once(self) -> None:
        # The same lock searches take: never let a query read a half-written index.
        with self.lock:
            update = update_indexes(self._workspace, self._settings, self.provider())
        if update.pending:
            # A capture committed while this pass was encoding, so its passages are not in
            # the index. /capture also schedules, but relying on that alone loses the work if
            # this process dies first: the follow-up would never be queued.
            print(f"[convsearch] {update.pending} passage(s) arrived mid-pass: rescheduling")
            self.schedule()
        if update.mode == "noop":
            return
        detail = f" ({update.reason})" if update.reason else ""
        print(
            f"[convsearch] auto-index {update.mode}: encoded {update.encoded}, "
            f"{update.total} total{detail}"
        )


def make_handler(
    workspace: Path,
    settings: Settings,
    provider_factory: Callable[[], EmbeddingProvider],
    *,
    auto_indexer: AutoIndexer | None = None,
) -> type[BaseHTTPRequestHandler]:
    # `lock` guards the embedding model and reranker, which are not thread safe, and is
    # therefore also what keeps a reindex from running underneath an in-flight search.
    lock = auto_indexer.lock if auto_indexer is not None else threading.Lock()
    # Captures are cheap SQLite writes and must not queue behind a slow search, so they take
    # their own lock; SQLite itself serialises them against the reindex transaction.
    write_lock = threading.Lock()
    provider_cell: list[EmbeddingProvider] = []

    def get_provider() -> EmbeddingProvider:
        # Reuse the auto-indexer's provider when there is one: the embedding model is the
        # largest thing in the process, and two copies would double memory for no gain.
        # Both callers hold `lock`, which is the same lock either way.
        if auto_indexer is not None:
            return auto_indexer.provider()
        if not provider_cell:
            provider_cell.append(provider_factory())
        return provider_cell[0]

    class Handler(BaseHTTPRequestHandler):
        server_version = "convsearch"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[convsearch] {self.address_string()} {fmt % args}")

        def handle_one_request(self) -> None:
            # Browsers drop keep-alive connections without ceremony (closing a tab, or
            # Chromium tearing down a context). socketserver would print a traceback for
            # each one, which reads like a server fault but is normal client behaviour.
            #
            # Catch ConnectionError, not the individual subclasses: Windows raises
            # ConnectionAbortedError (WinError 10053) where POSIX raises ConnectionResetError,
            # and an earlier version of this handler listed only the POSIX pair — so closing the
            # popup mid-response took the whole server down.
            try:
                super().handle_one_request()
            except ConnectionError:
                self.close_connection = True

        def _cors(self) -> None:
            origin = _allowed_origin(self.headers.get("Origin"))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _respond(self, status: int, body: Any, *, close: bool = False) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            if close:
                self.close_connection = True
                self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.wfile.write(encoded)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/health", "/"}:
                problem = _workspace_problem(workspace)
                payload: dict[str, Any] = {
                    # Still 200 with status "ok" so existing clients keep working; `problem`
                    # is additive and says what is wrong when something is.
                    "status": "ok",
                    **_workspace_status(workspace),
                    "indexing": auto_indexer.busy if auto_indexer is not None else False,
                    "auto_index": auto_indexer is not None,
                }
                if problem is not None:
                    payload["problem"] = problem
                self._respond(200, payload)
                return
            if parsed.path == "/search":
                self._search(parse_qs(parsed.query))
                return
            if parsed.path == "/ask":
                self._ask(parse_qs(parsed.query))
                return
            if parsed.path == "/memories":
                self._memories(parse_qs(parsed.query))
                return
            if parsed.path == "/memories/review":
                # Must be matched before the generic `/memories/{id}` prefix below, or
                # "review" would be parsed as a memory id.
                self._memories_review(parse_qs(parsed.query))
                return
            if parsed.path.startswith("/memories/"):
                self._memory_detail(parsed.path)
                return
            if parsed.path == "/projects":
                self._projects()
                return
            if parsed.path.startswith("/projects/") and parsed.path.endswith("/export"):
                self._project_export(parsed.path)
                return
            if parsed.path.startswith("/projects/"):
                self._project_detail(parsed.path)
                return
            if parsed.path == "/tasks":
                self._tasks(parse_qs(parsed.query))
                return
            if parsed.path == "/timeline":
                self._timeline(parse_qs(parsed.query))
                return
            if parsed.path == "/captures":
                self._captures(parse_qs(parsed.query))
                return
            if parsed.path == "/diagnostics":
                self._diagnostics()
                return
            if parsed.path == "/privacy":
                self._privacy()
                return
            if parsed.path.startswith("/conversation/"):
                self._conversation(parsed.path)
                return
            if parsed.path == "/suggestions":
                self._suggestions(parse_qs(parsed.query))
                return
            if parsed.path == "/plan":
                self._plan(parse_qs(parsed.query))
                return
            if parsed.path == "/learn/stats":
                self._learn_stats()
                return
            if parsed.path == "/learn/preferences":
                self._learn_preferences(parse_qs(parsed.query))
                return
            self._respond(404, {"error": "not found", "path": parsed.path})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/capture":
                self._capture()
                return
            if parsed.path == "/reindex":
                self._reindex()
                return
            if parsed.path == "/feedback":
                self._feedback()
                return
            if parsed.path == "/learn":
                self._learn()
                return
            if parsed.path.startswith("/memories/") and parsed.path.endswith("/confirm"):
                self._memory_confirm(parsed.path)
                return
            if parsed.path.startswith("/memories/") and parsed.path.endswith("/invalidate"):
                self._memory_invalidate(parsed.path)
                return
            if parsed.path.startswith("/memories/") and parsed.path.endswith("/pin"):
                self._memory_pin(parsed.path)
                return
            if parsed.path.startswith("/tasks/") and parsed.path.endswith("/complete"):
                self._task_complete(parsed.path)
                return
            if parsed.path.startswith("/tasks/") and parsed.path.endswith("/reopen"):
                self._task_reopen(parsed.path)
                return
            self._respond(404, {"error": "not found", "path": parsed.path})

        def _read_body(self) -> bytes | None:
            """Return the request body, or None after answering with 413/400."""
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length else 0
            except ValueError:
                self._respond(400, {"error": "invalid Content-Length header"}, close=True)
                return None
            if length > MAX_CAPTURE_BYTES:
                # Do not drain a body we are refusing; drop the connection instead.
                message = f"body exceeds {MAX_CAPTURE_BYTES} bytes"
                self._respond(413, {"error": message, "limit": MAX_CAPTURE_BYTES}, close=True)
                return None
            return self.rfile.read(length) if length else b""

        def _unavailable(self) -> bool:
            """Answer 503 with a specific reason when the workspace itself is unusable."""
            problem = _workspace_problem(workspace)
            if problem is None:
                return False
            self._respond(503, {"error": problem, "workspace": str(workspace)})
            return True

        def _capture(self) -> None:
            body = self._read_body()
            if body is None:
                return
            if self._unavailable():
                return
            try:
                payload = parse_capture_payload(body)
            except CaptureValidationError as exc:
                self._respond(400, {"error": str(exc)})
                return
            try:
                with write_lock:
                    result = capture_conversations(workspace, settings, payload)
            except sqlite3.DatabaseError as exc:
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if result.conversations_written and auto_indexer is not None:
                # Returns immediately; the pass runs on the indexer thread after a debounce.
                auto_indexer.schedule()
            try:
                with connection(workspace) as conn:
                    stale = read_index_stale(conn)
            except sqlite3.DatabaseError:
                # The write above succeeded, so report the capture rather than failing it.
                stale = True
            self._respond(200, result.payload(stale_index=stale))

        def _reindex(self) -> None:
            if self._read_body() is None:
                return
            if self._unavailable():
                return
            try:
                # Held for the whole rebuild: searches queue rather than read a half-written
                # vector index, and the embedding provider stays single-threaded.
                with lock:
                    # build_indexes derives the stale flag from what is actually embedded.
                    # Forcing it False here would clobber a capture that committed while the
                    # rebuild was encoding, hiding data that is not in the index.
                    indexed = build_indexes(workspace, settings, get_provider())
            except (EmbeddingModelError, IndexLockTimeout, VectorIndexError) as exc:
                # These already carry a message naming the fix, so pass it through verbatim
                # rather than burying it behind a generic "500 RuntimeError".
                self._respond(503, {"error": str(exc)})
                return
            except OSError as exc:
                self._respond(
                    507,
                    {
                        "error": (
                            f"writing the index failed: {exc}. The previous index is intact. "
                            "This is usually a full disk or a locked file."
                        )
                    },
                )
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            try:
                with connection(workspace) as conn:
                    stale = read_index_stale(conn)
            except sqlite3.DatabaseError:
                stale = False
            self._respond(200, {"indexed_passages": indexed, "stale_index": stale})

        def _run_conversation_search(
            self,
            query: str,
            limit: int,
            profile: str,
            passages: int,
            branches: bool,
        ) -> list[ConversationResult] | None:
            """Run a conversation search, answering with the right error and returning None."""
            try:
                # The embedding model and reranker are not guaranteed thread safe.
                with lock:
                    return search_conversations(
                        workspace,
                        query,
                        settings,
                        get_provider(),
                        limit=limit,
                        profile=profile,
                        show_passages=passages,
                        include_branches=branches,
                    )
            except (EmbeddingModelError, VectorIndexError) as exc:
                # A missing model or an unreadable index is a server-side precondition the
                # user can fix; 503 plus the message beats a 500 with a stack-trace type name.
                self._respond(503, {"error": str(exc)})
                return None
            except RuntimeError as exc:
                self._respond(409, {"error": str(exc)})
                return None
            except sqlite3.DatabaseError as exc:
                self._respond(
                    503,
                    {
                        "error": (
                            f"the workspace database at {database_path(workspace)} could not "
                            f"be read: {exc}. Check that the workspace still exists and is "
                            "not open in another tool."
                        )
                    },
                )
                return None
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return None

        def _search(self, params: dict[str, list[str]]) -> None:
            query = (params.get("q") or [""])[0].strip()
            if not query:
                self._respond(400, {"error": "missing query parameter `q`"})
                return
            limit = _first_int(params.get("limit"), settings.final_result_limit, low=1, high=50)
            passages = _first_int(params.get("passages"), 2, low=0, high=10)
            profile = (params.get("profile") or ["balanced"])[0]
            branches = _first_bool(params.get("branches"), False)
            level = (params.get("level") or ["conversation"])[0]
            explain = _first_bool(params.get("explain"), False)
            boost = _first_bool(params.get("boost"), True)
            if level not in {"conversation", "segment", "passage"}:
                self._respond(
                    400,
                    {"error": f"unknown level `{level}`; use conversation|segment|passage"},
                )
                return
            if self._unavailable():
                return
            if level == "segment":
                self._search_segments(query, limit, branches, explain)
                return
            results = self._run_conversation_search(query, limit, profile, passages, branches)
            if results is None:
                return
            if boost and level == "conversation":
                # Learned click-boost: prior open/inspect events on token-overlapping queries
                # nudge those conversations up. A pure DB read on its own connection (no model,
                # so no `lock`). With no interactions logged this is a no-op re-sort, so the
                # default (boost=1) leaves the documented ordering unchanged. Passage/segment
                # levels are deliberately untouched.
                try:
                    with connection(workspace) as conn:
                        results = apply_click_boost(results, query, conn)
                except sqlite3.DatabaseError:
                    # Never let an interaction-table hiccup break search; serve the raw ranking.
                    pass
            if level == "passage":
                flat = [hit for result in results for hit in result.best_passages]
                self._respond(
                    200,
                    {
                        "query": query,
                        "level": "passage",
                        "count": len(flat),
                        "results": serializers.flat_passage_payload(flat, explain=explain),
                    },
                )
                return
            source_ids = _source_ids(workspace, [r.conversation_id for r in results])
            if explain:
                result_payload = serializers.build_conversation_result_payload(
                    results, source_ids, explain=True
                )
            else:
                # Byte-compatible with the original response for clients that pass no flags.
                result_payload = _result_payload(results, source_ids)
            self._respond(
                200,
                {
                    "query": query,
                    "profile": profile,
                    "count": len(results),
                    "results": result_payload,
                },
            )

        def _search_segments(self, query: str, limit: int, branches: bool, explain: bool) -> None:
            try:
                results = search_segments(
                    workspace, query, settings, limit=limit, include_branches=branches
                )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(
                200,
                {
                    "query": query,
                    "level": "segment",
                    "count": len(results),
                    "results": [serializers.segment_payload(r, explain=explain) for r in results],
                },
            )

        def _ask(self, params: dict[str, list[str]]) -> None:
            question = (params.get("q") or [""])[0].strip()
            if not question:
                self._respond(400, {"error": "missing query parameter `q`"})
                return
            limit = _first_int(params.get("limit"), 5, low=1, high=50)
            passages = _first_int(params.get("passages"), 3, low=0, high=10)
            ask_settings = settings
            backend_values = params.get("backend")
            if backend_values:
                backend = backend_values[0]
                if backend not in {"auto", "ollama", "anthropic"}:
                    self._respond(
                        400,
                        {"error": f"unknown backend `{backend}`; use auto|ollama|anthropic"},
                    )
                    return
                ask_settings = settings.model_copy(
                    update={"llm": settings.llm.model_copy(update={"backend": backend})}
                )
            if self._unavailable():
                return
            try:
                # answer_question searches (embedding provider) and then calls the LLM; the
                # whole thing must be single-threaded like every other provider user.
                with lock:
                    result = answer_question(
                        workspace,
                        question,
                        ask_settings,
                        get_provider(),
                        limit=limit,
                        passages_per_conversation=passages,
                    )
            except Exception as exc:
                # An unreachable LLM (LLMUnavailableError) or any other generation failure is
                # a precondition the user can fix; never let it crash the server.
                self._respond(
                    503,
                    {
                        "error": "answer generation failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                )
                return
            self._respond(200, serializers.answer_payload(result))

        def _memories(self, params: dict[str, list[str]]) -> None:
            query = (params.get("q") or [""])[0].strip()
            kind = ((params.get("kind") or [""])[0].strip()) or None
            status = ((params.get("status") or [""])[0].strip()) or None
            project = ((params.get("project") or [""])[0].strip()) or None
            limit = _first_int(params.get("limit"), 30, low=1, high=200)
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    if query:
                        records = search_memories(
                            conn,
                            query,
                            kinds=[kind] if kind else None,
                            statuses=[status] if status else None,
                            project=project,
                            limit=limit,
                        )
                    else:
                        records = list_memories(
                            conn, kind=kind, status=status, project=project, limit=limit
                        )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(
                200,
                {
                    "count": len(records),
                    "query": query,
                    "memories": [serializers.memory_list_item(r) for r in records],
                },
            )

        def _memory_detail(self, path: str) -> None:
            raw = path[len("/memories/") :]
            try:
                memory_id = int(unquote(raw))
            except ValueError:
                self._respond(404, {"error": "not found", "path": path})
                return
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    record = get_memory(conn, memory_id)
                    if record is None:
                        self._respond(404, {"error": "memory not found", "memory_id": memory_id})
                        return
                    history = conn.execute(
                        "SELECT old_status, new_status, reason, changed_at "
                        "FROM memory_status_history WHERE memory_id = ? ORDER BY history_id",
                        (memory_id,),
                    ).fetchall()
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.memory_detail_payload(record, history))

        def _memories_review(self, params: dict[str, list[str]]) -> None:
            limit = _first_int(params.get("limit"), 30, low=1, high=200)
            kind = ((params.get("kind") or [""])[0].strip()) or None
            project = ((params.get("project") or [""])[0].strip()) or None
            include_reviewed = _first_bool(params.get("include_reviewed"), False)
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    queue = build_review_queue(
                        conn,
                        limit=limit,
                        kind=kind,
                        project=project,
                        include_reviewed=include_reviewed,
                    )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.review_queue_payload(queue))

        def _review_item_dict(self, conn: sqlite3.Connection, memory_id: int) -> dict[str, Any]:
            """Re-read one memory in the review-item shape after a mutation.

            Built straight from the `memories` table rather than through
            `build_review_queue`, because a memory that was just confirmed, invalidated,
            or pinned no longer qualifies for the pending queue and would otherwise
            disappear from the response entirely.
            """
            row = conn.execute(
                "SELECT m.memory_id, m.kind, m.statement, m.status, m.project, m.confidence,"
                " m.created_at, m.pinned, m.reviewed_at, m.conversation_id,"
                " c.title AS conversation_title FROM memories m "
                "LEFT JOIN conversations c ON c.conversation_id = m.conversation_id "
                "WHERE m.memory_id = ?",
                (memory_id,),
            ).fetchone()
            assert row is not None  # the caller's mutator already proved the memory exists
            has_conflict = memory_id in _conflict_ids(conn)
            reason = _review_reason(row["status"], has_conflict, row["confidence"])
            evidence = _load_evidence_batch(conn, [memory_id]).get(memory_id, ())
            conflicts = _load_conflicts_batch(conn, [memory_id]).get(memory_id, ())
            superseded_by = _load_superseded_by_batch(conn, [memory_id]).get(memory_id, ())
            return serializers.review_state_payload(
                memory_id=row["memory_id"],
                kind=row["kind"],
                statement=row["statement"],
                status=row["status"],
                project=row["project"],
                confidence=row["confidence"],
                created_at=row["created_at"],
                pinned=bool(row["pinned"]),
                reviewed_at=row["reviewed_at"],
                conversation_id=row["conversation_id"],
                conversation_title=row["conversation_title"],
                evidence=evidence,
                conflicts=conflicts,
                superseded_by=superseded_by,
                review_reason=reason,
            )

        def _parse_memory_id(
            self, path: str, suffix: str, *, prefix: str = "/memories/"
        ) -> int | None:
            """Pull the id out of a `{prefix}{id}{suffix}` path, or answer 400.

            Shared by the memory review mutations (`/memories/{id}/confirm|invalidate|pin`)
            and the task mutations (`/tasks/{id}/complete|reopen`) -- both are just "an
            integer id between two fixed path segments".
            """
            inner = path[len(prefix) : -len(suffix)]
            try:
                return int(unquote(inner))
            except ValueError:
                self._respond(400, {"error": f"invalid id in path `{path}`"})
                return None

        def _read_json_object(self) -> dict[str, Any] | None:
            """Read the body and parse it as a JSON object, or answer 400/413 and return None."""
            body = self._read_body()
            if body is None:
                return None
            try:
                payload = json.loads(body) if body else {}
            except (ValueError, UnicodeDecodeError) as exc:
                self._respond(400, {"error": f"invalid JSON body: {exc}"})
                return None
            if not isinstance(payload, dict):
                self._respond(400, {"error": "body must be a JSON object"})
                return None
            return payload

        def _memory_mutation_response(
            self, memory_id: int, mutate: Callable[[sqlite3.Connection], None]
        ) -> None:
            """Run a review mutator and respond with the updated item, mapping errors."""
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    mutate(conn)
                    item = self._review_item_dict(conn, memory_id)
            except ValueError as exc:
                self._respond(404, {"error": str(exc), "memory_id": memory_id})
                return
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, item)

        def _optional_reason(self, payload: dict[str, Any]) -> str | None | object:
            """Validate the optional `reason` field; returns a sentinel on error already sent."""
            reason = payload.get("reason")
            if reason is not None and not isinstance(reason, str):
                self._respond(400, {"error": "`reason` must be a string"})
                return _INVALID
            return reason

        def _memory_confirm(self, path: str) -> None:
            memory_id = self._parse_memory_id(path, "/confirm")
            if memory_id is None:
                return
            payload = self._read_json_object()
            if payload is None:
                return
            reason = self._optional_reason(payload)
            if reason is _INVALID:
                return
            self._memory_mutation_response(
                memory_id,
                lambda conn: confirm_memory(conn, memory_id, reason=cast("str | None", reason)),
            )

        def _memory_invalidate(self, path: str) -> None:
            memory_id = self._parse_memory_id(path, "/invalidate")
            if memory_id is None:
                return
            payload = self._read_json_object()
            if payload is None:
                return
            reason = self._optional_reason(payload)
            if reason is _INVALID:
                return
            self._memory_mutation_response(
                memory_id,
                lambda conn: invalidate_memory(conn, memory_id, reason=cast("str | None", reason)),
            )

        def _memory_pin(self, path: str) -> None:
            memory_id = self._parse_memory_id(path, "/pin")
            if memory_id is None:
                return
            payload = self._read_json_object()
            if payload is None:
                return
            if "pinned" not in payload or not isinstance(payload["pinned"], bool):
                self._respond(400, {"error": "`pinned` must be a boolean (true or false)"})
                return
            pinned = payload["pinned"]
            reason = self._optional_reason(payload)
            if reason is _INVALID:
                return
            self._memory_mutation_response(
                memory_id,
                lambda conn: set_memory_pinned(
                    conn, memory_id, pinned, reason=cast("str | None", reason)
                ),
            )

        def _task_mutation_response(
            self, memory_id: int, mutate: Callable[[sqlite3.Connection], None]
        ) -> None:
            """Run a task-state mutator and respond with the updated task item, mapping errors.

            Mirrors `_memory_mutation_response`: `ValueError` (unknown id, or a memory that
            isn't kind='task') maps to 404, `mutate` is expected to commit its own write
            (`set_task_state` does not commit -- see `_commit_task_state` below).
            """
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    mutate(conn)
                    item = get_task(conn, memory_id)
            except ValueError as exc:
                self._respond(404, {"error": str(exc), "memory_id": memory_id})
                return
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            assert item is not None  # mutate() already proved the task exists
            self._respond(200, serializers.task_item_payload(item))

        def _commit_task_state(
            self, conn: sqlite3.Connection, memory_id: int, new_state: str, reason: str | None
        ) -> None:
            set_task_state(conn, memory_id, new_state, reason=reason)
            conn.commit()

        def _task_complete(self, path: str) -> None:
            memory_id = self._parse_memory_id(path, "/complete", prefix="/tasks/")
            if memory_id is None:
                return
            payload = self._read_json_object()
            if payload is None:
                return
            reason = self._optional_reason(payload)
            if reason is _INVALID:
                return
            self._task_mutation_response(
                memory_id,
                lambda conn: self._commit_task_state(
                    conn, memory_id, "completed", cast("str | None", reason)
                ),
            )

        def _task_reopen(self, path: str) -> None:
            memory_id = self._parse_memory_id(path, "/reopen", prefix="/tasks/")
            if memory_id is None:
                return
            payload = self._read_json_object()
            if payload is None:
                return
            reason = self._optional_reason(payload)
            if reason is _INVALID:
                return
            self._task_mutation_response(
                memory_id,
                lambda conn: self._commit_task_state(
                    conn, memory_id, "open", cast("str | None", reason)
                ),
            )

        def _projects(self) -> None:
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    summaries = list_projects(conn)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(
                200,
                {
                    "count": len(summaries),
                    "projects": [serializers.project_summary_payload(s) for s in summaries],
                },
            )

        def _project_detail(self, path: str) -> None:
            name = unquote(path[len("/projects/") :])
            if not name:
                self._respond(404, {"error": "not found", "path": path})
                return
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    report = reconstruct_project(conn, name)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if report is None:
                self._respond(404, {"error": "project not found", "name": name})
                return
            self._respond(200, serializers.project_report_payload(report))

        def _project_export(self, path: str) -> None:
            inner = path[len("/projects/") : -len("/export")].strip("/")
            name = unquote(inner)
            if not name:
                self._respond(404, {"error": "not found", "path": path})
                return
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    report = reconstruct_project(conn, name)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            if report is None:
                self._respond(404, {"error": "project not found", "name": name})
                return
            markdown = render_project_markdown(report)
            self._respond(200, {"name": name, "markdown": markdown})

        def _tasks(self, params: dict[str, list[str]]) -> None:
            state = (params.get("state") or ["open"])[0]
            if state not in {"open", "completed", "all"}:
                self._respond(400, {"error": f"unknown state `{state}`; use open|completed|all"})
                return
            project = ((params.get("project") or [""])[0].strip()) or None
            limit = _first_int(params.get("limit"), 50, low=1, high=500)
            evidence = _first_bool(params.get("evidence"), False)
            since_raw = (params.get("since") or [""])[0].strip()
            since_dt = None
            if since_raw:
                since_dt = _parse_since(since_raw)
                if since_dt is None:
                    self._respond(
                        400,
                        {"error": f"invalid `since` value `{since_raw}`; use 7d, 24h, or 30m"},
                    )
                    return
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    result = list_tasks(
                        conn,
                        state=state,
                        project=project,
                        limit=limit,
                        since=since_dt,
                        include_evidence=evidence,
                    )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.task_list_payload(result))

        def _timeline(self, params: dict[str, list[str]]) -> None:
            topic = (params.get("q") or [""])[0].strip()
            if not topic:
                self._respond(400, {"error": "missing query parameter `q`"})
                return
            project = ((params.get("project") or [""])[0].strip()) or None
            limit = _first_int(params.get("limit"), 40, low=1, high=200)
            evidence = _first_bool(params.get("evidence"), False)
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    result = build_timeline(
                        conn, topic, project=project, limit=limit, include_evidence=evidence
                    )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.timeline_payload(result))

        def _captures(self, params: dict[str, list[str]]) -> None:
            source = (params.get("source") or ["all"])[0]
            if source not in {"all", "live", "import"}:
                self._respond(400, {"error": f"unknown source `{source}`; use all|live|import"})
                return
            limit = _first_int(params.get("limit"), 50, low=1, high=500)
            problems = _first_bool(params.get("problems"), False)
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    inventory = list_captures(
                        conn, source=source, limit=limit, only_problems=problems
                    )
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.capture_inventory_payload(inventory))

        def _diagnostics(self) -> None:
            if self._unavailable():
                return
            try:
                checks = run_doctor(workspace, settings)
                readiness = probe_llm_readiness(settings)
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.diagnostics_payload(checks, readiness))

        def _privacy(self) -> None:
            if self._unavailable():
                return
            try:
                readiness = probe_llm_readiness(settings)
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            # Reuse the presence check readiness already computed rather than re-reading the
            # environment variable here: one place decides what "configured" means.
            cloud_configured = any(c.name == "anthropic_api_key" and c.ok for c in readiness.checks)
            # ThreadingHTTPServer always binds a TCP address, so this is a (host, port) pair;
            # the stub types `server_address` broadly to also cover AF_UNIX sockets.
            address = cast("tuple[str, int]", self.server.server_address)
            host, port = address
            payload = serializers.privacy_payload(
                workspace_path=str(workspace.resolve()),
                database_path=str(database_path(workspace).resolve()),
                index_path=str(faiss_index_path(workspace).resolve()),
                server_bind=f"{host}:{port}",
                loopback_only=_is_loopback_host(str(host)),
                backend_mode=settings.llm.backend,
                ollama_host=settings.llm.ollama_host,
                cloud_configured=cloud_configured,
                readiness=readiness,
                counts=_privacy_counts(workspace),
            )
            self._respond(200, payload)

        def _conversation(self, path: str) -> None:
            raw = path[len("/conversation/") :]
            try:
                conversation_id = int(unquote(raw))
            except ValueError:
                self._respond(404, {"error": "not found", "path": path})
                return
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    row = conn.execute(
                        "SELECT conversation_id, source_conversation_id, title, created_at, "
                        "updated_at FROM conversations WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    if row is None:
                        self._respond(
                            404,
                            {"error": "conversation not found", "conversation_id": conversation_id},
                        )
                        return
                    messages = conn.execute(
                        "SELECT message_id, role, text, created_at, is_primary_path, source_order "
                        "FROM messages WHERE conversation_id = ? ORDER BY source_order",
                        (conversation_id,),
                    ).fetchall()
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            source_id = row["source_conversation_id"]
            url = f"https://chatgpt.com/c/{source_id}" if source_id else None
            self._respond(200, serializers.conversation_payload(row, messages, url))

        def _feedback(self) -> None:
            body = self._read_body()
            if body is None:
                return
            if self._unavailable():
                return
            try:
                payload = json.loads(body) if body else {}
            except (ValueError, UnicodeDecodeError) as exc:
                self._respond(400, {"error": f"invalid JSON body: {exc}"})
                return
            if not isinstance(payload, dict):
                self._respond(400, {"error": "body must be a JSON object"})
                return
            event_type = payload.get("event_type")
            if event_type not in {"search", "open", "inspect", "ask"}:
                self._respond(
                    400,
                    {"error": "event_type must be one of search|open|inspect|ask"},
                )
                return
            event = InteractionEvent(
                event_type=event_type,
                query=str(payload.get("query") or ""),
                conversation_id=payload.get("conversation_id"),
                passage_id=payload.get("passage_id"),
                segment_id=payload.get("segment_id"),
                position=payload.get("position"),
            )
            try:
                # Pure SQLite write on its own connection; no model, so no `lock`.
                with connection(workspace) as conn:
                    event_id = record_event(conn, event)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, {"ok": True, "event_id": event_id})

        def _suggestions(self, params: dict[str, list[str]]) -> None:
            limit = _first_int(params.get("limit"), 8, low=1, high=50)
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    recent = recent_queries(conn, limit)
                    popular = popular_queries(conn, limit)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.suggestions_payload(recent, popular))

        def _learn_stats(self) -> None:
            if self._unavailable():
                return
            try:
                with connection(workspace) as conn:
                    stats = interaction_stats(conn)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, {"stats": stats})

        def _plan(self, params: dict[str, list[str]]) -> None:
            query = (params.get("q") or [""])[0].strip()
            if not query:
                self._respond(400, {"error": "missing query parameter `q`"})
                return
            if self._unavailable():
                return
            try:
                # The planner runs read-only tools that hit the embedding model, so it must
                # be single-threaded like every other provider user; its DB reads go through a
                # workspace connection opened under the same lock.
                with lock, connection(workspace) as conn:
                    ctx = PlannerContext(
                        workspace=workspace,
                        settings=settings,
                        provider=get_provider(),
                        conn=conn,
                    )
                    answer = execute_plan(ctx, query)
            except (EmbeddingModelError, VectorIndexError) as exc:
                self._respond(503, {"error": str(exc)})
                return
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(200, serializers.plan_payload(query, answer))

        def _learn(self) -> None:
            body = self._read_body()
            if body is None:
                return
            if self._unavailable():
                return
            try:
                payload = json.loads(body) if body else {}
            except (ValueError, UnicodeDecodeError) as exc:
                self._respond(400, {"error": f"invalid JSON body: {exc}"})
                return
            if not isinstance(payload, dict):
                self._respond(400, {"error": "body must be a JSON object"})
                return
            use_llm = bool(payload.get("use_llm", True))
            try:
                # May call the local LLM (Ollama) when use_llm is true, so single-thread it
                # under the shared lock. It degrades to a heuristic if the LLM is unreachable,
                # but any unexpected failure is surfaced as a 503 rather than crashing.
                with lock, connection(workspace) as conn:
                    summary = run_self_improvement(conn, settings, use_llm=use_llm)
            except Exception as exc:
                self._respond(
                    503,
                    {
                        "error": "self-improvement failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                )
                return
            self._respond(
                200,
                {
                    "events_read": summary.events_read,
                    "notes_written": summary.notes_written,
                    "backend": summary.backend,
                    "model": summary.model,
                    "notes": list(summary.notes),
                },
            )

        def _learn_preferences(self, params: dict[str, list[str]]) -> None:
            limit = _first_int(params.get("limit"), 20, low=1, high=200)
            if self._unavailable():
                return
            try:
                # Pure SQLite read on its own connection; no model, so no `lock`.
                with connection(workspace) as conn:
                    prefs = list_learned_preferences(conn, limit)
            except sqlite3.DatabaseError as exc:
                self._respond(503, {"error": f"{type(exc).__name__}: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - surfaced to the client
                self._respond(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._respond(
                200,
                {
                    "preferences": [
                        {
                            "pref_id": pref_id,
                            "note": note,
                            "weight": weight,
                            "created_at": created_at,
                        }
                        for pref_id, note, weight, created_at in prefs
                    ]
                },
            )

    return Handler


def serve(
    workspace: Path,
    settings: Settings,
    provider_factory: Callable[[], EmbeddingProvider],
    *,
    host: str = "127.0.0.1",
    port: int = 8756,
    auto_index: bool = True,
    auto_index_delay: float = 3.0,
) -> None:
    # This is local single-user software -- a user should never need to know migrations
    # exist. Applying them here (idempotent, additive-only) means an existing workspace last
    # touched several releases ago just works when `serve` is started, instead of failing
    # deep inside the memory-review endpoints with a raw `no such column` error. Read-only
    # commands deliberately do NOT do this. Three entry points apply schema changes: `init`
    # (via initialize_database, when creating or upgrading a workspace), `serve` here, and
    # the explicit `migrate` command.
    applied = apply_pending_migrations(workspace)
    if applied:
        print(f"[convsearch] applied {len(applied)} pending migration(s): {', '.join(applied)}")

    auto_indexer: AutoIndexer | None = None
    if auto_index:
        auto_indexer = AutoIndexer(workspace, settings, provider_factory, delay=auto_index_delay)
        auto_indexer.start()
    handler = make_handler(workspace, settings, provider_factory, auto_indexer=auto_indexer)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"[convsearch] serving {workspace} on http://{host}:{port}")
    print(
        "[convsearch] endpoints: GET /health, GET /search?q=..., GET /ask?q=..., "
        "GET /plan?q=..., GET /memories, GET /memories/{id}, GET /memories/review, "
        "GET /projects, "
        "GET /projects/{name}, GET /conversation/{id}, GET /suggestions, "
        "GET /learn/stats, GET /learn/preferences, "
        "GET /tasks, GET /timeline?q=..., GET /captures, GET /projects/{name}/export, "
        "GET /diagnostics, GET /privacy, "
        "POST /capture, POST /reindex, POST /feedback, POST /learn, "
        "POST /memories/{id}/confirm, POST /memories/{id}/invalidate, POST /memories/{id}/pin, "
        "POST /tasks/{id}/complete, POST /tasks/{id}/reopen"
    )
    if auto_indexer is not None:
        print(
            f"[convsearch] auto-index on: captured conversations become searchable "
            f"~{auto_index_delay:g}s after capture"
        )
    else:
        print("[convsearch] auto-index off: use POST /reindex or `convsearch index`")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[convsearch] shutting down")
    finally:
        if auto_indexer is not None:
            auto_indexer.stop()
        httpd.server_close()
