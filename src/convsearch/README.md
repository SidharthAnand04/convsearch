# convsearch engine layout

This directory is the whole engine: import ChatGPT conversations into a local SQLite
workspace, chunk them into passages, index them lexically and semantically, retrieve
across both, and answer questions over the result. Everything runs on the user's machine
against a single workspace directory. There is no server-side component and no shared
state, which is why almost every function here takes an explicit `workspace: Path`
instead of reading a global.

This file is orientation for contributors. It covers how the packages depend on each
other, who owns each pipeline stage, the two invariants that are expensive to break, and
where new code usually belongs.

## Layering

Imports run one way. Reading top to bottom, each tier may import anything above it and
nothing below it:

```
domain, config, utils          value objects, settings, shared SQL/hash helpers
  -> storage                   connections, migration runner
  -> passages, importers, capture, embeddings, llm
  -> segmentation, indexes
  -> retrieval
  -> memory, answer, projects, tasks, feedback
  -> planner, timeline, digest
  -> cli, server               the only entry points
```

The hard rule: **no engine package imports `cli` or `server`.** The only edges into
those two are `cli/app.py:315` importing `server.app.serve` to run the `serve` command,
and `server/__init__.py` re-exporting it. If you find yourself wanting engine code to
reach for a request handler or a Typer callback, the logic belongs in the engine package
and the entry point should call it.

A few edges are worth knowing because they are not obvious from the tier list:

- `capture/ingest.py` imports `importers/chatgpt.py`. Live capture deliberately reuses
  the importer's persistence, so the tiers are peers only in the sense that capture sits
  on top of importers.
- `indexes/build.py` imports `segmentation/build.py` and `capture/state.py`: an index
  pass also refreshes segments and clears the capture staleness flag.
- `memory` imports `retrieval/query.py` (the query parser) but not `retrieval/service.py`.
  Memory search is FTS over its own tables; it does not go through conversation search.
- `planner` imports `memory/search.py`, `projects/reconstruct.py` and
  `retrieval/service.py`, which is what puts it above all three.

## Packages

| Package | What it owns |
| --- | --- |
| `domain/` | Pydantic boundary models (`ImportedConversation`, `ImportedMessage`) and frozen dataclasses for results (`Passage`, `PassageHit`, `ConversationResult`, `SegmentResult`). No behaviour, no IO. |
| `config/` | `Settings` loaded from YAML, plus the path helpers (`database_path`, `faiss_index_path`, `vector_map_path`) that define a workspace's on-disk layout. Every path in the system derives from here. |
| `utils.py` | Shared leaf helpers: `stable_hash` for content-addressed ids, and the `memory_effective_timestamp_*` SQL fragments that several packages splice into queries so "effective date" means the same thing everywhere. |
| `storage/` | `connect`/`connection` context managers and the migration runner. See `migrations/README.md` for the migration contract. |
| `importers/` | Reading a ChatGPT export ZIP and persisting it. `chatgpt.py` holds both the ZIP walk and `persist_conversation`, the single writer of conversation/message/passage rows. |
| `capture/` | Live capture from the browser extension: `ingest.py` normalises the scraped payload into the same domain models and calls `persist_conversation`; `state.py` tracks the stale-index flag and the synthetic capture import row; `inventory.py` reports what capture has actually landed. |
| `passages/` | Pure chunking. Splits message text into overlapping word-budgeted passages with offsets, so a hit can be located back in the original message. No IO. |
| `embeddings/` | The `EmbeddingProvider` protocol plus a lazily loaded sentence-transformers implementation with a bounded cache. Optional dependency; the protocol is what the rest of the code depends on. |
| `llm/` | Text generation. `client.py` wraps the optional Anthropic client behind a `MessagesClient` protocol; `generate.py` is the call site the engine uses, Ollama-first with a local HTTP fallback. |
| `segmentation/` | Grouping a conversation's messages into topical segments. `rules.py`, `semantic.py` and `hybrid.py` are interchangeable providers behind `base.py`; `build.py` drives the rebuild and the `segment_fts` refresh. |
| `indexes/` | Building and maintaining the two search indexes. Carries the hardest correctness invariants in the codebase — see below. `build.py` orchestrates, `lexical.py` handles FTS5, `vector.py` handles the FAISS index and its positional map, `locking.py` provides the cross-process advisory locks. |
| `retrieval/` | Search, one stage per file: `query.py` parses, `lexical.py`/`semantic.py`/`segments.py` are the channels, `fusion.py` merges them by reciprocal rank, `reranking.py` reorders, `aggregation.py` rolls passages up to conversations, `explain.py` renders why a hit matched, `service.py` is the entry point that runs the whole chain. |
| `memory/` | Extracting durable statements (decisions, tasks, preferences, risks, ...) out of message text and reconciling them over time. The project's most distinctive subsystem; it has its own `README.md`. |
| `answer/` | Retrieval-augmented answering: run a search, trim the top passages into sources, build a grounded prompt, call the LLM, return the answer with its citations. |
| `projects/` | `reconstruct.py` assembles a `ProjectReport` from memory rows; `export.py` renders one as evidence-cited Markdown. Both are pure over SQLite reads — `export.py` does no IO at all. |
| `tasks/` | Read-only queries over task-kind memories, filtering out statuses that mean the obligation is dead (invalidated, superseded) regardless of `task_state`. |
| `timeline/` | Chronological view of memory activity, built on `memory/search.py` and the shared effective-timestamp SQL. |
| `digest/` | "What changed recently" summary. Fully deterministic, no LLM: it reads real timestamps from `memories.created_at` and `memory_status_history.changed_at`. |
| `planner/` | Deterministic rule-based query planner. `plan_query` picks a plan from regex rules, `execute_plan` runs the steps through the read-only tool registry in `tools.py`. No memory writes, and every finding must cite ids that appear in tool output. |
| `feedback/` | Logs search interactions and distils them into `learned_preferences` notes, LLM-assisted with a deterministic fallback. |
| `evaluation/` | Synthetic end-to-end evaluation: builds a throwaway workspace, runs the real pipeline, scores retrieval, and stores runs for comparison. |
| `diagnostics/` | `doctor.py` checks a workspace (schema currency, FTS row counts, index files present); `llm_readiness.py` checks whether a generation backend is actually reachable. |
| `server/` | The local HTTP server: a stdlib `ThreadingHTTPServer` with a hand-routed handler in `app.py`, plus `serializers.py` for JSON payload shapes. |
| `cli/` | The Typer app. One module, `app.py`; every command is a thin wrapper over engine functions. |

## The pipeline, end to end

| Stage | Owner |
| --- | --- |
| Import from export ZIP | `importers/chatgpt.py:35 import_chatgpt_zip` -> `:254 persist_conversation` |
| Live capture | `POST /capture` -> `server/app.py:610 _capture` -> `capture/ingest.py:150 capture_conversations` -> the same `persist_conversation` |
| Chunking | `passages/builder.py:24 build_passages_for_message` |
| Segmentation | `segmentation/build.py:33 rebuild_segments`, `:52 rebuild_segments_for_conversations` |
| Indexing | `indexes/build.py:240 update_indexes` (incremental) or `:90 build_indexes` (full) |
| Retrieval | `retrieval/service.py:19 search_conversations` |
| Answering | `answer/answer.py:90 answer_question`, or `planner/planner.py:476 execute_plan` |

The important thing about the first two rows: capture and import converge on
`persist_conversation`. The browser only ever sees the visible linear chain of a
conversation, with no alternate branches; that is the *only* structural difference, and
`capture/ingest.py` handles it by normalising into the same `ImportedConversation` /
`ImportedMessage` models rather than writing its own SQL. If you change how conversations
are stored, there is exactly one place to change it. Keep it that way.

## Two things not to break

**1. Opening a database never migrates it.** `connection()` (`storage/database.py:24`)
just connects and closes; schema mutation lives only in `apply_pending_migrations`
(`:44`). It is reached from `convsearch init` (via `initialize_database`,
`cli/app.py:123`), `convsearch migrate` (`cli/app.py:252`), and `serve` startup
(`server/app.py:1584`). Read-only commands — `search`, `tasks list`, `digest` — must
never rewrite schema out from under the user; they instead check
`pending_migrations()` up front and refuse with a clear message. If you add a command
that needs a new column, guard it centrally with that check rather than migrating on
open.

**2. The FAISS vector map is positional.** `vector_map_path` stores
`{"passage_ids": [...]}` where a list *position* is the FAISS vector id
(`indexes/vector.py:1-18`). Drift between the list and the index does not raise — it
returns the wrong passage, which is a plausible-looking wrong answer rather than a
visible failure. The module defends this with atomic temp-file-plus-`os.replace` writes,
a cross-process `swap` lock held across both replaces, an `ntotal`-versus-map-length
check that forces a full rebuild after a crash between them, and bounds-checking on every
returned vector id. Any change to index writing must preserve the position-to-passage_id
alignment, and must keep those defences intact.

## Where to add things

| You want to... | Do this |
| --- | --- |
| Add a retrieval channel | New module under `retrieval/` returning `list[PassageHit]`, then wire it into `reciprocal_rank_fusion` (`retrieval/fusion.py:7`) and into `service.py` |
| Add a memory kind | A migration widening the `kind` CHECK, then `memory/models.py:5`, then a trigger rule in `memory/extract.py` |
| Add an HTTP route | A branch in `do_GET`/`do_POST` in `server/app.py`, a payload function in `server/serializers.py`, and a case in `tests/test_server.py` |
| Add a CLI command | A function in `cli/app.py` that calls existing engine code; if it needs new logic, that logic goes in an engine package |

## Conventions

- `mypy --strict` over the whole package (`pyproject.toml`), Python 3.12 target.
- `ruff` with `select = ["E", "F", "I", "UP", "B", "SIM", "C4", "RUF"]`, line length 100,
  double quotes.
- `from __future__ import annotations` at the top of every module.
- Frozen dataclasses for internal value objects; pydantic only at boundaries (parsing an
  export payload, loading settings). Do not push pydantic into the retrieval hot path.
- Optional dependencies (`faiss`, `sentence-transformers`, `anthropic`) are imported
  lazily via `importlib` and hidden behind protocols, so the core stays importable
  without them.

One grep trap worth knowing: `server/serializers.py` is imported as
`from convsearch.server import serializers` (`server/app.py:71`) and used at 19 call
sites through the module name. A grep for `server.serializers` or for individual function
names finds nothing. It is not dead code.
