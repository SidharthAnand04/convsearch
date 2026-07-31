# memory

Search finds passages. This package tries to answer a different question: *what did I
decide, and is it still true?* It reads the same conversation text and pulls out durable
statements — decisions, tasks, preferences, risks, constraints, open questions — then
maintains them over time, so a decision made in March and reversed in June shows up as
superseded rather than as two equally confident search hits.

That maintenance is what makes this the most distinctive subsystem here, and it is also
where the surprises live. This file explains the model, not just the call graph.

## The pipeline

```
message text
  -> extract.py::extract_from_message      sentence split, trigger rules, label lookahead
  -> quality.py::is_usable_statement       reject-only precision filter
  -> store.py::_insert_memory              content-hash dedup, INSERT OR IGNORE
  -> store.py::reconcile                   supersede / contest, once per batch
```

`_insert_memory` deliberately does not call `reconcile()` or `commit()`. Both run once at
the end of a batch in the caller, so supersession sees every new memory at once instead
of reconciling against a half-written set.

## Kinds and statuses

Seven kinds (`models.py:5-13`): `decision`, `task`, `preference`, `project_state`,
`risk`, `constraint`, `open_question`.

Six statuses (`models.py:14-21`): `proposed`, `active`, `contested`, `superseded`,
`invalidated`, `historical`.

Two of those six are aspirational. `proposed` and `historical` exist in the schema's
CHECK constraint and in `MEMORY_STATUSES`, but nothing in the code ever writes them —
every insert is hardcoded to `'active'` (`store.py:124`), and the only transitions
written are to `contested`, `superseded`, `active` and `invalidated`. Do not write code
that assumes a memory can be in a `proposed` state; nothing puts it there today.

## Dedup

`content_hash = stable_hash(kind, subject_key, statement, message_id)`, inserted with
`INSERT OR IGNORE` (`store.py:116-125`). Re-running extraction over the same corpus is
therefore free and idempotent.

The consequence worth internalising: **an existing memory is never overwritten.** If you
improve the extractor so it produces a better `statement` for the same sentence, the hash
changes and you get a *second* memory, not an updated one. Reprocessing a corpus after an
extractor change means purging first (`clear_memories`, which respects curation — see
below), not re-running and expecting rows to be corrected in place.

## Supersession (`store.py:389-497`)

`reconcile()` runs after every extraction batch. In order:

1. Group memories by `(kind, subject_key, project)`, restricted to
   `kind IN ('decision', 'preference')` and to groups with more than one member. Tasks,
   risks, constraints and open questions are **never** auto-superseded — a second risk
   about the same subject is additional information, not a replacement.
2. Order the group by `(created_at IS NULL), created_at, message_id`. The last row is
   "newest"; the `created_at IS NULL` term sorts undated rows to the end deterministically.
3. If an earlier member's date key *equals* the newest's, they are treated as
   simultaneous: both become `contested` and a `conflicts_with` relation is written
   between them. Nothing is superseded.
4. Otherwise the earlier member becomes `superseded`, the newest is forced back to
   `active` if it was neither active nor contested, and a `supersedes` relation is written
   from the newest to the earlier one.
5. Every transition writes a `memory_status_history` row with **`reason = NULL`**. That
   NULL is load-bearing: it is the only thing distinguishing automatic reconciliation from
   a deliberate human status change, and the curation predicate below depends on it.
   Never pass a reason from `reconcile()`.
6. FTS5 has no in-place UPDATE, so a status change deletes and reinserts the `memory_fts`
   row keyed on `rowid = memory_id` (`store.py:516-531`). Any new writer of
   `memories.status` must go through `_update_memory_fts_status` or the search index
   silently keeps the old status.

### The surprising case: same-conversation decisions come out contested

Steps 3 and 4 hinge on the date key, and the date key is not always what you would guess.

Live-captured conversations carry `created_at = NULL` on every message, because ChatGPT's
DOM exposes no per-message timestamp. Extraction does not store a bare NULL, though: it
resolves an effective timestamp first, via `memory_effective_timestamp_sql("m", "c")`
(`store.py:47`) — the message's own `created_at`, then the conversation's `created_at`,
then the conversation's `updated_at`, which for a captured conversation is capture
wall-clock time.

So every memory extracted from one captured conversation inherits **the same** timestamp:
that conversation's capture time. Two decisions about the same subject in the same
captured conversation therefore have equal date keys, compare equal at step 3, and come
out **contested rather than superseded** — even when the user plainly changed their mind
halfway through the conversation. Message order is only a tiebreaker in the ORDER BY; it
is not consulted when comparing date keys.

If you are debugging "why is this contested instead of superseded", check the source of
the timestamp before checking the algorithm.

## Confidence, and why the review threshold is not arbitrary

The extractor emits exactly two confidence values (`extract.py:293`):

```python
confidence = 0.9 if (from_label or starts_with_trigger or has_identifier) else 0.7
```

0.9 when the statement came from a label lookahead, or the sentence *starts* with the
trigger phrase, or it names an identifier. 0.7 otherwise. There is no continuum.

`LOW_CONFIDENCE_THRESHOLD = 0.75` (`review.py:14`) sits between the two tiers **by
construction**: it means "the 0.7 tier goes to the review queue, the 0.9 tier does not".
It is not a tuned number. Changing either the tiers or the threshold without the other
either floods the review queue or empties it.

The LLM path uses a third fixed value, `_LLM_CONFIDENCE = 0.6` (`llm_extract.py:26`),
deliberately below both rules tiers: a model proposal has no trigger-phrase or identifier
signal, so it has no basis for claiming 0.9.

## The quality filter

`is_usable_statement(text, kind)` (`quality.py:35-77`) returns `(True, None)` or
`(False, reason)`. Five rejections:

| # | Rejects | Why |
| --- | --- | --- |
| 1 | Trailing-colon fragment | "For your strategy, you need to know:" introduces a list; it is a lead-in, not a statement |
| 2 | Table-row debris | A tab, two or more pipes, or a leading bare row number — artifacts of a markdown table surviving sentence splitting |
| 3 | Fewer than 4 words | Below that it is reliably a label ("risk-free rate"), not a thought. Kept low on purpose |
| 4 | Task-only: negation of need | "you do not need to compute delta yourself" asserts a task is *not* required. Narrow phrasing so it misses "do not deploy without X", which is a real constraint |
| 5 | Task-only: pure question | A question is an `open_question`, not a task |

The contract is **reject-only**. This module never rewrites a statement and never
reclassifies a kind — rule 5 rejects rather than switching the kind precisely to stay
inside that contract, since changing `kind` belongs to the caller. When in doubt, keep: a
missed rejection is cheap, a wrongly rejected real memory is not recoverable.

Rejection reasons are counted into `reject_counts` and surfaced in the extraction summary,
so a filter change shows up as a shift in the discard breakdown rather than a silently
smaller number.

## Curation: four signals, one predicate

`clear_memories` must never delete something a person deliberately touched.
`_CURATION_PREDICATE` (`store.py:277-282`) is the single source of truth for what "a
person touched this" means:

| Signal | Set by |
| --- | --- |
| `pinned = 1` | `set_memory_pinned` |
| `reviewed_at IS NOT NULL` | `confirm_memory` / `invalidate_memory` |
| A `memory_status_history` row with non-NULL `reason` | `set_memory_status(..., reason=...)` — a manual status change. `reconcile()` always writes NULL, so ordinary reconciliation does not qualify |
| **Any** `task_state_history` row | `set_task_state`. Unlike status history, every row here is already a deliberate user action; there is no automatic writer, so no reason filter is needed |

This predicate has gone stale once already: migration 009 shipped `task_state_history`
without adding it here, so completing a task did not protect it from a purge. If you add
a new user-facing mutation, add its signal to this constant. Do not add a parallel
curation check somewhere else — that duplication is exactly how it went stale.

## Two extraction paths, one store

| | Rules | LLM (opt-in) |
| --- | --- | --- |
| Entry point | `store.py:38 extract_and_store_memories` | `llm_extract.py:199 propose_memories` -> `store.py:226 store_extracted_memories` |
| `extraction_version` | `rules-v2` | `llm-v1` |
| Confidence | 0.7 / 0.9 | fixed 0.6 |
| Insert path | `_insert_memory` | `_insert_memory` |
| Reconciliation | one `reconcile()` per batch | one `reconcile()` per batch |

`llm_extract.py` imports `_detect_project` and `_subject_key` *from* `extract.py`
(`llm_extract.py:12`). That is intentional: the LLM path extends the rules path, it does
not replace it. Subject keys and project detection must agree across both, or the same
decision extracted twice lands in two different reconciliation groups and never
supersedes.

The LLM path also runs its proposals through `is_usable_statement` and verifies that the
model's `quote` is a verbatim substring of the source text before accepting anything —
same reject-only discipline, plus grounding. Its result object counts each discard reason
separately so a caller can report a discard rate rather than one opaque number.

`extraction_version` is what makes the two paths separable after the fact:
`clear_memories(extraction_version="llm-v1")` removes an LLM experiment without touching
rules-extracted memories.

## Task state

`task_state` is `open`, `completed`, or NULL, and lives on the memory row alongside
`status`. The two are orthogonal: a completed task can still be superseded, and an
invalidated memory is not a live obligation regardless of its `task_state` — which is why
`tasks/query.py` filters on both.

`set_task_state` (`store.py:574`) is the only writer of `task_state_history`, and it
enforces two things:

- the memory must be `kind = 'task'`; letting a decision grow a `task_state` would be a
  data-integrity bug, not a feature.
- a no-op returns early (`:604-606`) without writing a history row or touching
  `task_state_changed_at`, so re-completing an already-completed task does not fabricate
  a transition in the audit trail.

## subject_key

`_subject_key` (`extract.py:190-201`) is what decides which memories are "about the same
thing", and therefore what reconciliation groups together. It tries, in order:

1. the first token matching `IDENTIFIER_RE` (paths, dotted names, ALL_CAPS, camelCase),
   lowercased;
2. otherwise the first three non-stopword alphanumeric terms joined by `-`;
3. otherwise the literal `"general"`.

The fallback is coarse by design. Two decisions phrased differently can land in different
groups and never supersede each other, and many unrelated statements can pile up under
`general`. Tightening this changes which historical memories reconcile against each other,
so it is not a local change.
