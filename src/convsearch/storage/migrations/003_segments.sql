CREATE TABLE IF NOT EXISTS segments (
  segment_id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  segment_order INTEGER NOT NULL,
  start_message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  end_message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  title TEXT,
  summary TEXT,
  boundary_confidence REAL NOT NULL,
  segmentation_version TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(conversation_id, segment_order)
);

ALTER TABLE passages ADD COLUMN segment_id INTEGER REFERENCES segments(segment_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_segments_conversation_order
ON segments(conversation_id, segment_order);

CREATE INDEX IF NOT EXISTS idx_passages_segment
ON passages(segment_id);

CREATE VIRTUAL TABLE segment_fts USING fts5(
  text,
  title UNINDEXED,
  conversation_title UNINDEXED
);
