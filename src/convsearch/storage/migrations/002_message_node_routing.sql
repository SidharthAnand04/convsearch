ALTER TABLE messages ADD COLUMN source_node_id TEXT;
ALTER TABLE messages ADD COLUMN parent_source_node_id TEXT;
ALTER TABLE messages ADD COLUMN resolved_parent_message_id INTEGER REFERENCES messages(message_id);

UPDATE messages
SET source_node_id = source_message_id
WHERE source_node_id IS NULL;

UPDATE messages
SET parent_source_node_id = parent_message_id
WHERE parent_source_node_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conversation_source_node
ON messages(conversation_id, source_node_id);

CREATE INDEX IF NOT EXISTS idx_messages_resolved_parent_message_id
ON messages(resolved_parent_message_id);
