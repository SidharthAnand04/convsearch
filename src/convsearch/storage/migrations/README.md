# Migrations

Plain `.sql` files, applied in filename order, recorded by name. There is no ORM, no
down-migration, and no separate "upgrade" code path — the same runner brings a brand-new
workspace and a three-releases-stale one to the same schema.

## How they run

`migration_files()` returns `sorted(MIGRATIONS_DIR.glob("*.sql"))` — plain lexicographic
order over filenames (`../migrations.py:8`). That is why the numeric prefix must be
zero-padded: without it, `10_x.sql` would sort before `9_x.sql`.

`apply_pending_migrations` (`../database.py:44-75`) creates `schema_migrations` if needed,
reads the set of applied versions, and for each file not in that set runs
`conn.executescript(...)` and inserts `path.stem` as the version. The stem — for example
`005_memory_kinds` — is the identity key. It returns the list of versions it actually
applied, so an empty list means "already current".

Three commands reach it:

| Entry point | Path |
| --- | --- |
| `convsearch init` | `cli/app.py:123` -> `initialize_database` -> `apply_pending_migrations` |
| `convsearch migrate` | `cli/app.py:252` |
| `convsearch serve` startup | `server/app.py:1584` |

Everything else — `search`, `tasks list`, `digest`, `doctor` — is read-only and must stay
that way. A read command that silently rewrote schema would be a nasty surprise on a
workspace the user had not chosen to upgrade. Those commands instead call
`pending_migrations(conn)` once, up front, and refuse with a clear message
(`cli/app.py:1621-1637`); `doctor` reports pending migrations as a check
(`diagnostics/doctor.py:38`). Add your guard there, centrally — never per-column
introspection like "does this table have column X".

## Writing a new one

- Name it `NNN_snake_case.sql` with a zero-padded three-digit prefix, one higher than the
  current maximum.
- **Never edit a shipped migration.** Its stem is already recorded in users'
  `schema_migrations` tables; changing the file changes nothing for them and silently
  diverges their schema from a fresh install's. Add a new file instead.
- Prefer additive `ALTER TABLE ... ADD COLUMN` with a safe default (see 007 and 009).
  SQLite applies these instantly and they cannot fail on existing data.
- `executescript` runs **outside** a transaction. For a single additive statement that is
  fine. For a multi-statement destructive change — anything with a `DROP` — wrap the
  destructive part in an explicit `BEGIN; ... COMMIT;` so a failure halfway through does
  not leave a half-rebuilt table. 005 is the worked example.
- Write a test. `tests/test_database_cli.py` covers the runner itself (pending detection,
  idempotency, double-`initialize_database`); add a case for whatever your migration makes
  possible.
- No guard clause needed inside the migration for "has this already run" — the
  `schema_migrations` gate handles that. `IF NOT EXISTS` on `CREATE` is still worth using
  as belt and braces, and most files here do.

## The nine

| # | File | What it does |
| --- | --- | --- |
| 001 | `initial` | Core schema: `imports`, `conversations`, `messages`, `passages`, `embedding_records`, `index_metadata`, `import_warnings`. Adds the `passage_fts` FTS5 virtual table and three triggers — `passages_ai`, `passages_ad`, `passages_au` (`:74-91`) — that keep it in sync with `passages` on insert, delete and update. Because the triggers join `messages` and `conversations` to denormalise `role` and `title` into the FTS row, the message and conversation must already exist when a passage is inserted. |
| 002 | `message_node_routing` | Adds `source_node_id`, `parent_source_node_id`, `resolved_parent_message_id` to `messages`, backfills them from the existing id columns, and adds a unique index on `(conversation_id, source_node_id)`. This is what makes branch routing possible: an export's message graph is keyed by node id, not by the flat message id 001 assumed. |
| 003 | `segments` | Adds the `segments` table, a nullable `passages.segment_id` (`ON DELETE SET NULL`, so re-segmenting never destroys passages), and the `segment_fts` virtual table. |
| 004 | `memories` | The memory graph: `memories`, `memory_evidence`, `memory_relations`, `memory_status_history`, `entities`, `entity_mentions`, and `memory_fts`. Note `memory_evidence.passage_id` is nullable with `ON DELETE SET NULL` (`:28`) while `message_id` is `NOT NULL` — a memory's evidence must always point at a message, but re-chunking can legitimately delete the passage without invalidating the quote. |
| 005 | `memory_kinds` | Widens the `memories.kind` CHECK to add `constraint` and `open_question`. SQLite cannot alter a CHECK in place, so the table is rebuilt: create, copy, drop, rename. Two details make it safe (`:6-10`): `PRAGMA foreign_keys=OFF` around the rebuild, because dropping the old `memories` would otherwise fire `ON DELETE CASCADE` and wipe every child table; and `memory_id` values copied verbatim, so `memory_fts` rowids and all foreign keys stay valid. The copy/drop/rename is wrapped in an explicit `BEGIN`/`COMMIT` since `executescript` gives it no transaction of its own. |
| 006 | `interactions` | `interactions` (search / open / inspect / ask events) and `learned_preferences`, backing the self-improvement loop in `feedback/`. |
| 007 | `memory_review` | Adds `memories.pinned` and `memories.reviewed_at`, plus a review index. The comment at `:3-5` explains the design call: `pinned` is a column rather than a status value because it is orthogonal to status — a pinned memory can still be active or contested. |
| 008 | `backfill_timestamps` | Data-only, no schema change. Derives missing `created_at`/`updated_at` from related rows: a conversation from its messages, a message from its conversation, a memory from its evidence message then its conversation. Every statement is `WHERE created_at IS NULL`, so it is idempotent by construction (`:9-10`). It explicitly **never invents a date** (`:52-54`) — rows where nothing upstream has a real timestamp, the live-capture case, are left NULL rather than given a plausible guess, because a wrong timestamp would make the decision timeline lie. |
| 009 | `task_state_history` | Adds the `task_state_history` table and `memories.task_state_changed_at`. `old_state` is nullable, unlike `memory_status_history.old_status` (`:14-17`): a task memory can sit at `task_state = NULL` when the extractor matched `kind='task'` but neither the open- nor completed-trigger list, so its *first* real transition is genuinely `NULL -> open` rather than a fabricated prior state. |

Two of these are worth reading in full before writing anything similar: 005 for how to
rebuild a table without cascading its children away, and 008 for how to backfill without
inventing data.

One thing 009 did *not* do, and which is the standing lesson here: it shipped
`task_state_history` without adding it to `_CURATION_PREDICATE` in `memory/store.py`, so
for a while a user-completed task was not protected from a purge. A migration that adds a
table recording deliberate user action almost always has a code-side counterpart. Look for
it.
