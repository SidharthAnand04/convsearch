CREATE TABLE imports (
  import_id INTEGER PRIMARY KEY,
  source_path TEXT NOT NULL,
  source_hash TEXT NOT NULL UNIQUE,
  imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status TEXT NOT NULL,
  warning_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE conversations (
  conversation_id INTEGER PRIMARY KEY,
  source_conversation_id TEXT NOT NULL UNIQUE,
  import_id INTEGER NOT NULL REFERENCES imports(import_id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at TEXT,
  updated_at TEXT,
  content_hash TEXT NOT NULL
);

CREATE TABLE messages (
  message_id INTEGER PRIMARY KEY,
  source_message_id TEXT NOT NULL UNIQUE,
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  parent_message_id TEXT,
  role TEXT NOT NULL,
  created_at TEXT,
  source_order INTEGER NOT NULL,
  is_primary_path INTEGER NOT NULL,
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL
);

CREATE TABLE passages (
  passage_id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
  message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
  passage_order INTEGER NOT NULL,
  text TEXT NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset INTEGER NOT NULL,
  word_count INTEGER NOT NULL,
  content_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE embedding_records (
  passage_id INTEGER PRIMARY KEY REFERENCES passages(passage_id) ON DELETE CASCADE,
  vector_id INTEGER NOT NULL UNIQUE,
  model_id TEXT NOT NULL,
  embedding_dimension INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE index_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE import_warnings (
  warning_id INTEGER PRIMARY KEY,
  import_id INTEGER NOT NULL REFERENCES imports(import_id) ON DELETE CASCADE,
  context TEXT NOT NULL,
  message TEXT NOT NULL
);

CREATE VIRTUAL TABLE passage_fts USING fts5(
  text,
  role UNINDEXED,
  title UNINDEXED
);

CREATE TRIGGER passages_ai AFTER INSERT ON passages BEGIN
  INSERT INTO passage_fts(rowid, text, role, title)
  SELECT new.passage_id, new.text, m.role, c.title
  FROM messages m JOIN conversations c ON c.conversation_id = new.conversation_id
  WHERE m.message_id = new.message_id;
END;

CREATE TRIGGER passages_ad AFTER DELETE ON passages BEGIN
  DELETE FROM passage_fts WHERE rowid = old.passage_id;
END;

CREATE TRIGGER passages_au AFTER UPDATE ON passages BEGIN
  DELETE FROM passage_fts WHERE rowid = old.passage_id;
  INSERT INTO passage_fts(rowid, text, role, title)
  SELECT new.passage_id, new.text, m.role, c.title
  FROM messages m JOIN conversations c ON c.conversation_id = new.conversation_id
  WHERE m.message_id = new.message_id;
END;
