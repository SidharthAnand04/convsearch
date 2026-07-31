"use strict";

/**
 * Layer 3 — the popup UI, running as a real extension page against the real server.
 *
 * This spec seeds its own conversation through POST /capture so it does not depend on
 * capture.spec.js having run: a failure here means the popup is broken, not the capture
 * path.
 */

const { test, expect, newConversationId, health } = require("./helpers");
const { SERVER_URL } = require("./paths");

const TITLE = "Popup Search Fixture";
const conversationId = newConversationId();

const CONVERSATION = {
  source_conversation_id: conversationId,
  title: TITLE,
  created_at: null,
  updated_at: "2026-07-29T18:00:00Z",
  messages: [
    {
      source_message_id: `${conversationId}-0`,
      role: "user",
      text: "What is the best way to render search results inside a Chrome extension popup?",
      order: 0,
      created_at: null,
    },
    {
      source_message_id: `${conversationId}-1`,
      role: "assistant",
      text:
        "Build the nodes with createElement and textContent rather than innerHTML, because " +
        "conversation text is untrusted markup. Keep the popup narrow and let each result " +
        "link back to the original conversation.",
      order: 1,
      created_at: null,
    },
  ],
};

test.describe.serial("popup", () => {
  test("renders linked search results and supports a manual rebuild", async ({
    context,
    configuredExtension,
  }) => {
    // NOTE: this test used to leave the index stale and drive the "Rebuild index" button as
    // the happy path. Auto-indexing made that unraceable — the server clears staleness a
    // few seconds after capture, so the alert and its button are gone by design. The
    // automatic path is covered in autoindex.spec.js and scenario.spec.js; what is left to
    // cover here is the manual fallback plumbing and result rendering.
    const response = await fetch(`${SERVER_URL}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversations: [CONVERSATION] }),
    });
    expect(response.status).toBe(200);

    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${configuredExtension.extensionId}/popup.html`);

    // The panel paints "checking…" first and fills in from the background worker.
    await expect(popup.locator("#capture-summary")).toContainText("captured live", {
      timeout: 20_000,
    });

    // Exercise the same message the Rebuild index button sends. Clicking the button itself
    // is not reliable here because it lives inside an alert that auto-indexing hides.
    const reindexResult = await popup.evaluate(
      () =>
        new Promise((resolve) =>
          chrome.runtime.sendMessage({ type: "convsearch:reindex" }, resolve)
        )
    );
    expect(reindexResult.ok, `manual reindex failed: ${JSON.stringify(reindexResult)}`).toBe(true);
    expect(reindexResult.indexed_passages).toBeGreaterThan(0);
    expect((await health()).stale_index).toBe(false);

    await popup.locator("#query").fill("rendering search results in a browser extension popup");
    await popup.locator("#submit").click();

    const results = popup.locator(".result");
    await expect(results.first()).toBeVisible({ timeout: 30_000 });
    await expect(popup.locator("#status")).toContainText("result");

    // The link is only given an href when the URL starts with https://chatgpt.com/, so
    // this single assertion also proves the server built the deep link correctly.
    const link = popup.locator(".result-title", { hasText: TITLE });
    await expect(link).toHaveAttribute("href", `https://chatgpt.com/c/${conversationId}`);

    await popup.close();
  });

  test("says the server is offline when it cannot be reached", async ({
    context,
    configuredExtension,
  }) => {
    // Point the extension at a dead port; the popup must explain itself rather than hang.
    await configuredExtension.page.evaluate(
      () =>
        new Promise((resolve) =>
          chrome.storage.local.set({ serverUrl: "http://127.0.0.1:9", lastQuery: "" }, resolve)
        )
    );

    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${configuredExtension.extensionId}/popup.html`);

    await expect(popup.locator(".empty-title")).toHaveText(
      "The convsearch server isn't running.",
      { timeout: 30_000 }
    );
    await expect(popup.locator("#capture-summary")).toHaveText("Server offline");

    await popup.close();
  });
});
