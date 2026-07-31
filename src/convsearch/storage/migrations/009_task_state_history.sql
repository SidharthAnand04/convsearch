-- Task completion workflow: a history table for task_state transitions, mirroring
-- memory_status_history (004) for the `status` column.
--
-- Before this migration, `memories.task_state` was set ONLY by the extraction heuristic
-- (memory/extract.py) and never changed again -- there was no way for a user to mark a task
-- done, and no record distinguishing "the extractor guessed completed" from "a user completed
-- this". This table records every task_state transition made via set_task_state() (CLI/API),
-- so the digest can report real completions instead of windowing on the memory's created_at
-- (see digest/build.py _completed_tasks).
--
-- `task_state_changed_at` on memories is additive, mirroring the pinned/reviewed_at columns
-- added in 007: a plain ALTER TABLE is enough, no table rebuild. NULL means task_state has
-- never been changed via set_task_state (i.e. still whatever the extractor set, if anything).
-- `old_state` is nullable, unlike memory_status_history.old_status (always NOT NULL there):
-- a task memory can have task_state = NULL (the extractor matched kind='task' but neither the
-- open- nor completed-trigger word list), so the *first* real transition on such a task is
-- genuinely NULL -> 'open'/'completed', not a fabricated prior state.
CREATE TABLE IF NOT EXISTS task_state_history (
  history_id INTEGER PRIMARY KEY,
  memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  old_state TEXT,
  new_state TEXT NOT NULL,
  reason TEXT,
  changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_task_state_history_memory ON task_state_history(memory_id);

ALTER TABLE memories ADD COLUMN task_state_changed_at TEXT;
