CREATE TABLE IF NOT EXISTS memories (
  memory_id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('decision', 'task', 'preference', 'project_state', 'risk')),
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

CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(kind, subject_key);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_conversation ON memories(conversation_id);

CREATE TABLE IF NOT EXISTS memory_evidence (
  evidence_id INTEGER PRIMARY KEY,
  memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  passage_id INTEGER REFERENCES passages(passage_id) ON DELETE SET NULL,
  message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  quote TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_memory ON memory_evidence(memory_id);

CREATE TABLE IF NOT EXISTS memory_relations (
  relation_id INTEGER PRIMARY KEY,
  from_memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  to_memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  relation TEXT NOT NULL CHECK (
    relation IN ('supersedes', 'conflicts_with', 'relates_to', 'depends_on')
  ),
  reason TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(from_memory_id, to_memory_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_status_history (
  history_id INTEGER PRIMARY KEY,
  memory_id INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
  old_status TEXT NOT NULL,
  new_status TEXT NOT NULL,
  reason TEXT,
  changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entities (
  entity_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL UNIQUE,
  entity_type TEXT NOT NULL DEFAULT 'identifier',
  first_seen_at TEXT
);

CREATE TABLE IF NOT EXISTS entity_mentions (
  mention_id INTEGER PRIMARY KEY,
  entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  memory_id INTEGER REFERENCES memories(memory_id) ON DELETE SET NULL,
  UNIQUE(entity_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);

CREATE VIRTUAL TABLE memory_fts USING fts5(
  statement,
  kind UNINDEXED,
  project UNINDEXED,
  status UNINDEXED
);
