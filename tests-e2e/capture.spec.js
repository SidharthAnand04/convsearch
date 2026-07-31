"use strict";

/**
 * Layer 2 — the real thing.
 *
 * A real Chromium with the real unpacked extension loaded, visiting the real
 * https://chatgpt.com origin (fulfilled locally), talking to the real Python server.
 * Nothing is stubbed except the ChatGPT HTML itself.
 */

const { test, expect, fixtureHtml, serveChatGpt, newConversationId, health, search, reindex } =
  require("./helpers");
const { SERVER_URL } = require("./paths");

const TITLE = "Live Capture Architecture";

// Shared across the serial describe: the idempotency test must re-visit the SAME
// conversation the capture test stored.
const conversationId = newConversationId();
const html = () => fixtureHtml("conversation.html", { __CONV_ID__: conversationId, __TITLE__: TITLE });

/**
 * Records every POST /capture response body. context.route sees the background service
 * worker's fetches because the context is created with serviceWorkers: "allow".
 */
async function watchCaptures(context) {
  const seen = [];
  await context.route(`${SERVER_URL}/capture`, async (route) => {
    const response = await route.fetch();
    const body = await response.text();
    try {
      seen.push(JSON.parse(body));
    } catch {
      seen.push({ unparseable: body });
    }
    await route.fulfill({ response, body });
  });
  return seen;
}

/** Event-driven, not a sleep: the content script debounces ~1.5s then queues async. */
async function waitForCapture(seen, index = 0) {
  await expect
    .poll(() => seen.length, {
      timeout: 45_000,
      message: "the content script never produced a POST /capture",
    })
    .toBeGreaterThan(index);
  return seen[index];
}

test.describe.serial("live capture", () => {
  test("captures a browsed conversation into the local workspace", async ({
    context,
    configuredExtension,
  }) => {
    const before = await health();
    expect(before.status).toBe("ok");

    const seen = await watchCaptures(context);
    await serveChatGpt(context, { conversationId, html: html() });

    const page = await context.newPage();
    await page.goto(`https://chatgpt.com/c/${conversationId}`);

    const result = await waitForCapture(seen);
    expect(result.conversations_written).toBe(1);
    expect(result.messages_written).toBe(4); // the empty turn is not a message
    expect(result.skipped_unchanged).toBe(0);
    expect(result.stale_index).toBe(true);

    // The server's own view of the world, not just what it echoed back.
    await expect
      .poll(async () => (await health()).captured_conversations, { timeout: 20_000 })
      .toBe(before.captured_conversations + 1);
    const after = await health();
    expect(after.conversations).toBe(before.conversations + 1);
    expect(after.messages).toBe(before.messages + 4);
    expect(after.stale_index).toBe(true);

    // Capture must NOT have embedded inline; the index is built out of band.
    const indexed = await reindex();
    expect(indexed.indexed_passages).toBeGreaterThan(0);
    expect(indexed.stale_index).toBe(false);

    const found = await search("hybrid retrieval reciprocal rank fusion", { limit: "10" });
    const hit = (found.results || []).find(
      (r) => r.url === `https://chatgpt.com/c/${conversationId}`
    );
    expect(
      hit,
      `no result linked back to https://chatgpt.com/c/${conversationId}; got ${JSON.stringify(
        (found.results || []).map((r) => r.url)
      )}`
    ).toBeTruthy();
    expect(hit.title).toBe(TITLE);

    await page.close();
  });

  test("re-visiting an unchanged conversation is skipped, not duplicated", async ({
    context,
    configuredExtension,
  }) => {
    const before = await health();

    const seen = await watchCaptures(context);
    await serveChatGpt(context, { conversationId, html: html() });

    const page = await context.newPage();
    await page.goto(`https://chatgpt.com/c/${conversationId}`);

    const first = await waitForCapture(seen);
    expect(first.skipped_unchanged).toBe(1);
    expect(first.conversations_written).toBe(0);
    expect(first.messages_written).toBe(0);

    // A full reload re-runs the content script from scratch (its in-page dedup cache is
    // gone), so this is a genuine second POST of identical content.
    await page.reload();
    const second = await waitForCapture(seen, 1);
    expect(second.skipped_unchanged).toBe(1);
    expect(second.conversations_written).toBe(0);

    const after = await health();
    expect(after.conversations).toBe(before.conversations);
    expect(after.messages).toBe(before.messages);
    expect(after.captured_conversations).toBe(before.captured_conversations);

    await page.close();
  });

  test("an unfamiliar page shape captures nothing and breaks nothing", async ({
    context,
    configuredExtension,
  }) => {
    const before = await health();
    const errors = [];

    const seen = await watchCaptures(context);
    const strangeId = newConversationId();
    await serveChatGpt(context, {
      conversationId: strangeId,
      html: fixtureHtml("malformed.html"),
    });

    const page = await context.newPage();
    page.on("pageerror", (error) => errors.push(String(error)));
    await page.goto(`https://chatgpt.com/c/${strangeId}`);

    // Give the debounce + queue a real chance to misbehave before declaring silence.
    await page.waitForTimeout(6000);

    expect(seen, "extraction should have declined to capture an unusable page").toEqual([]);
    expect(errors, "the content script must never throw into the host page").toEqual([]);
    const after = await health();
    expect(after.conversations).toBe(before.conversations);

    await page.close();
  });
});
