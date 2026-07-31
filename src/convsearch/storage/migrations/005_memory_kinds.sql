-- Widen the memories.kind CHECK constraint to include 'constraint' and 'open_question'.
-- SQLite cannot ALTER a CHECK in place, so the table is rebuilt (create new, copy rows,
-- drop old, rename). memory_id values are preserved verbatim, so memory_fts rowids and
-- every foreign-key reference (memory_evidence, memory_relations, memory_status_history,
-- entity_mentions) remain valid.
--
-- Foreign keys are disabled for the rebuild: dropping the old `memories` table would
-- otherwise fire ON DELETE CASCADE against the child tables and wipe their rows. The
-- PRAGMA takes effect because executescript runs these statements outside a transaction;
-- the actual copy/drop/rename is wrapped in an explicit transaction for atomicity.
PRAGMA foreign_keys=OFF;

BEGIN;

CREATE TABLE memories_new (
  memory_id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (
    kind IN (
      'decision', 'task', 'preference', 'project_state', 'risk',
      'constraint', 'open_question'
    )
  ),
  subject_key TEXT NOT NULL,
  statement TEXT NOT NULL,
  status TEXT NOT NULL CHECK (
    status IN ('proposed', 'active', 'contested', 'superseded', 'invalidated', 'historical')
  ),
  confidence REAL NOT NULL,
  project TEXT,
  task_state TEXT CHECK (task_state IN ('open', 'completed') OR task_state IS NULL),
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  created_at TEXT,
  extracted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  extraction_version TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

INSERT INTO memories_new (
  memory_id, kind, subject_key, statement, status, confidence, project,
  task_state, conversation_id, message_id, created_at, extracted_at,
  extraction_version, content_hash, metadata_json
)
SELECT
  memory_id, kind, subject_key, statement, status, confidence, project,
  task_state, conversation_id, message_id, created_at, extracted_at,
  extraction_version, content_hash, metadata_json
FROM memories;

DROP TABLE memories;
ALTER TABLE memories_new RENAME TO memories;

CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(kind, subject_key);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_conversation ON memories(conversation_id);

COMMIT;

PRAGMA foreign_keys=ON;
