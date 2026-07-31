-- Backfill NULL created_at/updated_at from whatever related evidence can supply them.
--
-- Root cause: conversations/messages captured by the browser extension carry no timestamp
-- (the DOM does not expose one), so those rows were inserted with created_at = NULL. Memories
-- extracted from those messages inherited the same NULL. This migration derives what it can
-- from data that IS present and leaves the rest NULL -- it never invents a date, because a
-- wrong timestamp would make the decision timeline lie.
--
-- All statements are WHERE created_at IS NULL, so this is naturally idempotent: running it
-- again after new NULLs are legitimately backfilled elsewhere is a no-op.

-- Conversations: if every message in a conversation is timestamped but the conversation
-- itself is not (can happen for old rows written before the importer/capture paths set it),
-- take the earliest message time as the conversation's created_at and the latest as
-- updated_at.
UPDATE conversations
SET created_at = (
  SELECT MIN(m.created_at) FROM messages m
  WHERE m.conversation_id = conversations.conversation_id AND m.created_at IS NOT NULL
)
WHERE created_at IS NULL
AND EXISTS (
  SELECT 1 FROM messages m
  WHERE m.conversation_id = conversations.conversation_id AND m.created_at IS NOT NULL
);

UPDATE conversations
SET updated_at = (
  SELECT MAX(m.created_at) FROM messages m
  WHERE m.conversation_id = conversations.conversation_id AND m.created_at IS NOT NULL
)
WHERE updated_at IS NULL
AND EXISTS (
  SELECT 1 FROM messages m
  WHERE m.conversation_id = conversations.conversation_id AND m.created_at IS NOT NULL
);

-- Messages: if a message has no timestamp of its own but its conversation does, inherit the
-- conversation's created_at. This is a coarser date than the true message time but is strictly
-- better than NULL and cannot be more wrong than "sometime during this conversation".
UPDATE messages
SET created_at = (
  SELECT c.created_at FROM conversations c
  WHERE c.conversation_id = messages.conversation_id AND c.created_at IS NOT NULL
)
WHERE created_at IS NULL
AND EXISTS (
  SELECT 1 FROM conversations c
  WHERE c.conversation_id = messages.conversation_id AND c.created_at IS NOT NULL
);

-- Memories: derive from the evidence message's created_at first (most specific), falling back
-- to the conversation's created_at. Rows whose message and conversation are both still NULL
-- (the extension-capture case, where nothing upstream carries a real timestamp) are left NULL.
UPDATE memories
SET created_at = (
  SELECT m.created_at FROM messages m
  WHERE m.message_id = memories.message_id AND m.created_at IS NOT NULL
)
WHERE created_at IS NULL
AND EXISTS (
  SELECT 1 FROM messages m
  WHERE m.message_id = memories.message_id AND m.created_at IS NOT NULL
);

UPDATE memories
SET created_at = (
  SELECT c.created_at FROM conversations c
  WHERE c.conversation_id = memories.conversation_id AND c.created_at IS NOT NULL
)
WHERE created_at IS NULL
AND EXISTS (
  SELECT 1 FROM conversations c
  WHERE c.conversation_id = memories.conversation_id AND c.created_at IS NOT NULL
);
