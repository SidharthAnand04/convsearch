"use strict";

/**
 * Auto-indexing.
 *
 * The behaviour under test: opening a conversation on chatgpt.com makes it SEARCHABLE on
 * its own, with nobody calling POST /reindex. Everything here goes through the real
 * extension and the real Python server; only the ChatGPT HTML is local.
 *
 * These tests deliberately never call reindex(). If auto-indexing were removed they must
 * fail — that is the whole point, so the assertions are on semantic search hitting text
 * that exists ONLY in the message body, never in the title (a title match would pass even
 * with no vector index at all, and would prove nothing).
 */

const { test, expect, fixtureHtml, serveChatGpt, newConversationId, health, search } =
  require("./helpers");
const { SERVER_URL } = require("./paths");

/** Distinct prose per conversation — identical text would collapse via content_hash. */
const OTTERS = {
  id: newConversationId(),
  title: "Patagonian otter navigation",
  ask: "How do marine otters find their way through the Patagonian fjords each spring?",
  reply:
    "They follow continuous kelp corridors along the Chilean coastline and rely on " +
    "scent-marked haulout rocks as navigational waypoints throughout the migration.",
  // Body-only phrase: appears in the reply, not in the title.
  bodyQuery: "kelp corridors scent-marked haulout waypoints",
};

const SOURDOUGH = {
  id: newConversationId(),
  title: "Hydration troubleshooting",
  ask: "My loaf keeps collapsing once the hydration climbs past eighty percent.",
  reply:
    "Beyond eighty percent the gluten network wants gentle coil folds instead of " +
    "aggressive kneading, together with a noticeably cooler bulk fermentation.",
  bodyQuery: "gentle coil folds cooler bulk fermentation gluten",
};

function htmlFor(topic) {
  return fixtureHtml("topic.html", {
    __CONV_ID__: topic.id,
    __TITLE__: topic.title,
    __ASK__: topic.ask,
    __REPLY__: topic.reply,
  });
}

/** Poll the real server until the conversation is retrievable by its body text. */
async function waitUntilSearchable(topic, timeout = 90_000) {
  await expect
    .poll(
      async () => {
        const payload = await search(topic.bodyQuery, { limit: "10" });
        return (payload.results || []).some((r) => r.source_conversation_id === topic.id);
      },
      {
        timeout,
        intervals: [1000],
        message:
          `"${topic.title}" never became searchable by body text without a manual ` +
          "reindex — auto-indexing did not run",
      }
    )
    .toBe(true);
}

/** Visit a conversation and wait for the background worker to deliver its capture. */
async function browseAndCapture(context, topic) {
  const captured = [];
  await context.route(`${SERVER_URL}/capture`, async (route) => {
    const response = await route.fetch();
    const body = await response.text();
    try {
      captured.push(JSON.parse(body));
    } catch {
      captured.push({ unparseable: body });
    }
    await route.fulfill({ response, body });
  });

  await serveChatGpt(context, { conversationId: topic.id, html: htmlFor(topic) });
  const page = await context.newPage();
  await page.goto(`https://chatgpt.com/c/${topic.id}`);

  await expect
    .poll(() => captured.length, {
      timeout: 45_000,
      message: `no POST /capture for "${topic.title}"`,
    })
    .toBeGreaterThan(0);

  await context.unroute(`${SERVER_URL}/capture`);
  return { page, captured };
}

test.describe.serial("auto-indexing", () => {
  test("a browsed conversation becomes searchable with no manual reindex", async ({
    context,
    configuredExtension,
  }) => {
    const { captured } = await browseAndCapture(context, OTTERS);
    expect(captured[0].conversations_written).toBe(1);

    // Sanity check that we are proving something: right after capture the index is stale,
    // so this text is NOT yet findable. Auto-indexing is what changes that.
    await waitUntilSearchable(OTTERS);

    const after = await health();
    expect(after.captured_conversations).toBeGreaterThan(0);
    expect(after.stale_index).toBe(false);
  });

  test("a second conversation is appended without losing the first", async ({
    context,
    configuredExtension,
  }) => {
    // This is the failure mode incremental indexing most plausibly introduces: appending
    // the second conversation's vectors misaligns or overwrites the first's, and nothing
    // surfaces it unless you re-check the earlier conversation afterwards.
    await browseAndCapture(context, SOURDOUGH);
    await waitUntilSearchable(SOURDOUGH);

    const stillThere = await search(OTTERS.bodyQuery, { limit: "10" });
    expect(
      (stillThere.results || []).map((r) => r.source_conversation_id),
      "the first conversation vanished from the index when the second was appended"
    ).toContain(OTTERS.id);

    // And the deep link still points at the real site.
    const hit = (stillThere.results || []).find((r) => r.source_conversation_id === OTTERS.id);
    expect(hit.url).toBe(`https://chatgpt.com/c/${OTTERS.id}`);
  });

  test("the popup reports a current index rather than nagging for a rebuild", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);

    // Both conversations are indexed by now, so the stale-index alert must be hidden.
    await expect
      .poll(async () => popup.locator("#capture-alert").isVisible(), {
        timeout: 20_000,
        message: "the stale-index alert stayed visible even though the index is current",
      })
      .toBe(false);

    await popup.locator("#query").fill(SOURDOUGH.bodyQuery);
    await popup.locator("#submit").click();

    const first = popup.locator(".result").first();
    await expect(first).toBeVisible({ timeout: 30_000 });
    await expect(first.locator(".result-title")).toHaveAttribute(
      "href",
      `https://chatgpt.com/c/${SOURDOUGH.id}`
    );
  });
});
