"use strict";

/**
 * Layer 1 — pure extraction.
 *
 * No extension, no server: a plain page is navigated to a fulfilled chatgpt.com URL,
 * extension/capture.js is injected, and globalThis.convsearchExtract(document, url) is
 * called directly. A failure here localises to extension/capture.js and nothing else.
 */

const fs = require("node:fs");
const path = require("node:path");
const { test, expect } = require("@playwright/test");

const { EXTENSION_DIR } = require("./paths");
const { fixtureHtml, serveChatGpt, newConversationId } = require("./helpers");

const CAPTURE_JS = path.join(EXTENSION_DIR, "capture.js");

/** Loads a fixture at a real chatgpt.com URL with capture.js available. */
async function openFixture(page, { fixture, conversationId, title = "Retrieval Architecture" }) {
  await serveChatGpt(page, {
    conversationId,
    html: fixtureHtml(fixture, { __CONV_ID__: conversationId, __TITLE__: title }),
  });
  await page.goto(`https://chatgpt.com/c/${conversationId}`);
  await page.addScriptTag({ content: fs.readFileSync(CAPTURE_JS, "utf8") });
  await expect
    .poll(() => page.evaluate(() => typeof globalThis.convsearchExtract))
    .toBe("function");
}

const extract = (page) =>
  page.evaluate(() => globalThis.convsearchExtract(document, location.href));

test.describe("extraction (extension/capture.js)", () => {
  test("returns a well-formed conversation from a chatgpt.com-shaped page", async ({ page }) => {
    const conversationId = newConversationId();
    await openFixture(page, { fixture: "conversation.html", conversationId });

    const conversation = await extract(page);

    expect(conversation).not.toBeNull();
    expect(conversation.source_conversation_id).toBe(conversationId);
    expect(conversation.title).toBe("Retrieval Architecture");
    expect(conversation.created_at).toBeNull();
    expect(typeof conversation.updated_at).toBe("string");
    expect(Number.isNaN(Date.parse(conversation.updated_at))).toBe(false);

    // The empty turn is dropped, so 5 rendered turns become 4 messages...
    expect(conversation.messages).toHaveLength(4);
    // ...and `order` stays contiguous with no gap where the empty turn was.
    expect(conversation.messages.map((m) => m.order)).toEqual([0, 1, 2, 3]);
    expect(conversation.messages.map((m) => m.role)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(conversation.messages.map((m) => m.source_message_id)).toEqual([
      "msg-user-0001",
      "msg-asst-0002",
      "msg-user-0003",
      "msg-asst-0004",
    ]);
    expect(conversation.messages.every((m) => m.created_at === null)).toBe(true);

    // Text comes from .markdown, and is real prose rather than the whole turn's chrome.
    expect(conversation.messages[1].text).toContain("reciprocal rank fusion");
    expect(conversation.messages[0].text).toContain("retrieval architecture");
    for (const message of conversation.messages) {
      expect(message.text.trim().length).toBeGreaterThan(0);
    }
  });

  test("falls back to document.title when the sidebar link is gone", async ({ page }) => {
    // The sidebar is the part of ChatGPT's markup most likely to be restyled, so the
    // document.title fallback is the one that matters in production.
    const conversationId = newConversationId();
    await openFixture(page, { fixture: "conversation.html", conversationId });
    await page.evaluate(() => {
      document.querySelectorAll("nav, a[aria-current]").forEach((node) => node.remove());
    });

    const conversation = await extract(page);
    expect(conversation.title).toBe("Retrieval Architecture");
  });

  test("returns null (and does not throw) on an unfamiliar page shape", async ({ page }) => {
    const conversationId = newConversationId();
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));

    await openFixture(page, { fixture: "malformed.html", conversationId });

    const conversation = await extract(page);
    expect(conversation).toBeNull();
    expect(errors).toEqual([]);
  });

  test("returns null on a non-conversation URL", async ({ page }) => {
    const conversationId = newConversationId();
    await openFixture(page, { fixture: "conversation.html", conversationId });

    // Same DOM, but the URL is not /c/<uuid> — there is no id to store it under.
    const conversation = await page.evaluate(() =>
      globalThis.convsearchExtract(document, "https://chatgpt.com/")
    );
    expect(conversation).toBeNull();
  });

  test("survives being handed rubbish instead of a document", async ({ page }) => {
    await page.goto("about:blank");
    await page.addScriptTag({ content: fs.readFileSync(CAPTURE_JS, "utf8") });
    const results = await page.evaluate(() => {
      const extract = globalThis.convsearchExtract;
      return [extract(null, "https://chatgpt.com/c/abc12345"), extract({}, null), extract()];
    });
    expect(results).toEqual([null, null, null]);
  });
});
