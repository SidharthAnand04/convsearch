-- Human review workflow for memories: pinning and a reviewed_at stamp.
--
-- "pinned" is orthogonal to `status` (a pinned memory can still be active or contested), so
-- it is a new column rather than a new status value. Both columns are additive with safe
-- defaults, so a plain ALTER TABLE is enough -- no table rebuild like 005 needed.
ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN reviewed_at TEXT;

-- Speeds up "what still needs review" queries that filter on status while excluding
-- already-reviewed and pinned rows.
CREATE INDEX IF NOT EXISTS idx_memories_review ON memories(status, pinned, reviewed_at);
