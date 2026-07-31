"use strict";

/**
 * Shared test plumbing.
 *
 * The central trick: we never touch the real chatgpt.com. Playwright fulfils every
 * request to `https://chatgpt.com/**` with a local fixture, so the content script's
 * manifest match on `https://chatgpt.com/*` fires against the REAL origin with zero
 * network traffic and no login.
 */

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test: base, chromium, expect } = require("@playwright/test");

const { EXTENSION_DIR, FIXTURES_DIR, SERVER_URL } = require("./paths");

/**
 * MV3 extensions historically do not load in old headless Chromium. Headed is the
 * reliable mode and this machine has a Windows desktop, so headed is the default;
 * set E2E_HEADLESS=1 to try the new headless shell.
 */
const HEADLESS = process.env.E2E_HEADLESS === "1";

function fixtureHtml(name, replacements = {}) {
  let html = fs.readFileSync(path.join(FIXTURES_DIR, name), "utf8");
  for (const [token, value] of Object.entries(replacements)) {
    html = html.split(token).join(value);
  }
  return html;
}

/**
 * Route every chatgpt.com request in `target` (a BrowserContext or Page).
 * The conversation path gets the fixture; everything else (favicon, SPA polling)
 * gets an empty 204 so nothing escapes to the internet.
 */
async function serveChatGpt(target, { conversationId, html }) {
  await target.route("https://chatgpt.com/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === `/c/${conversationId}`) {
      await route.fulfill({
        status: 200,
        contentType: "text/html; charset=utf-8",
        body: html,
      });
      return;
    }
    await route.fulfill({ status: 204, body: "" });
  });
}

/** A fresh, unique conversation id per test so tests never collide in the workspace. */
function newConversationId() {
  const hex = (n) =>
    Array.from({ length: n }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  return `${hex(8)}-${hex(4)}-4${hex(3)}-a${hex(3)}-${hex(12)}`;
}

/* -------------------------------------------------------------------------- */
/* server access (this process is Windows node, so it CAN reach the loopback)  */
/* -------------------------------------------------------------------------- */

async function getJson(pathAndQuery) {
  const response = await fetch(`${SERVER_URL}${pathAndQuery}`);
  const body = await response.json();
  return { status: response.status, body };
}

async function health() {
  const { body } = await getJson("/health");
  return body;
}

async function search(query, params = {}) {
  const search = new URLSearchParams({ q: query, ...params });
  const { body } = await getJson(`/search?${search}`);
  return body;
}

async function reindex() {
  const response = await fetch(`${SERVER_URL}/reindex`, { method: "POST" });
  return response.json();
}

/* -------------------------------------------------------------------------- */
/* the extension fixture                                                      */
/* -------------------------------------------------------------------------- */

const test = base.extend({
  /** A persistent context is mandatory: --load-extension only works there. */
  context: async ({}, use) => {
    const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "convsearch-e2e-"));
    const context = await chromium.launchPersistentContext(userDataDir, {
      headless: HEADLESS,
      args: [
        `--disable-extensions-except=${EXTENSION_DIR}`,
        `--load-extension=${EXTENSION_DIR}`,
        "--no-first-run",
        "--no-default-browser-check",
      ],
      // Required for context.route() to see the background worker's fetches.
      serviceWorkers: "allow",
    });
    try {
      await use(context);
    } finally {
      await context.close();
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }
  },

  /** The generated chrome-extension:// host, read off the background service worker. */
  extensionId: async ({ context }, use) => {
    let [worker] = context.serviceWorkers();
    if (!worker) worker = await context.waitForEvent("serviceworker", { timeout: 30000 });
    await use(new URL(worker.url()).host);
  },

  /**
   * Points the extension at the disposable test server (it defaults to :8756, which
   * would be a developer's real workspace) and makes sure capture is on.
   * Returns the options page, already open, for tests that want it.
   */
  configuredExtension: async ({ context, extensionId }, use) => {
    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/options.html`);
    await page.evaluate(
      (serverUrl) =>
        new Promise((resolve) =>
          chrome.storage.local.set({ serverUrl, captureEnabled: true }, resolve)
        ),
      SERVER_URL
    );
    await use({ page, extensionId });
    if (!page.isClosed()) await page.close();
  },
});

module.exports = {
  test,
  expect,
  fixtureHtml,
  serveChatGpt,
  newConversationId,
  health,
  search,
  reindex,
  getJson,
  HEADLESS,
};
