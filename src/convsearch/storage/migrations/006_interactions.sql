-- Interaction-logging tables for the self-improvement loop. `interactions` records what the
-- user searches / opens / inspects / asks; prior clicks are later turned into a re-ranking
-- boost. `learned_preferences` is written by a parallel learn job.
CREATE TABLE IF NOT EXISTS interactions (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK (event_type IN ('search', 'open', 'inspect', 'ask')),
  query TEXT NOT NULL DEFAULT '',
  conversation_id INTEGER,
  passage_id INTEGER,
  segment_id INTEGER,
  position INTEGER,
  extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_interactions_event_type ON interactions(event_type);
CREATE INDEX IF NOT EXISTS idx_interactions_query ON interactions(query);

CREATE TABLE IF NOT EXISTS learned_preferences (
  pref_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  note TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  source TEXT
);
