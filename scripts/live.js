/**
 * Open the real chatgpt.com in Chromium with the convsearch extension loaded.
 *
 * This is not a test. There are no fixtures, no interception and no mock pages: it drives
 * the real site so live capture runs against real conversations.
 *
 * The browser profile is persisted in .chrome-profile/, so you log in to ChatGPT once and
 * the session survives later runs. Nothing about that profile is sent anywhere.
 *
 *   cmd.exe /c "npm run live"        from WSL
 *   npm run live                     from a Windows terminal
 */

"use strict";

const path = require("path");
const fs = require("fs");
const { chromium } = require("@playwright/test");

const REPO_ROOT = path.resolve(__dirname, "..");
const EXTENSION_DIR = path.join(REPO_ROOT, "extension");
const PROFILE_DIR = path.join(REPO_ROOT, ".chrome-profile");
const SERVER_URL = process.env.CONVSEARCH_SERVER || "http://127.0.0.1:8756";
const START_URL = process.env.CONVSEARCH_START_URL || "https://chatgpt.com/";

async function checkServer() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(`${SERVER_URL}/health`, { signal: controller.signal });
    return await response.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function main() {
  if (!fs.existsSync(path.join(EXTENSION_DIR, "manifest.json"))) {
    console.error(`No extension found at ${EXTENSION_DIR}`);
    process.exit(1);
  }

  const health = await checkServer();
  if (!health) {
    console.error(`\n  The convsearch server is not answering at ${SERVER_URL}.`);
    console.error("  Capture will queue in the extension and drain once it is up, but");
    console.error("  nothing becomes searchable until you start it:\n");
    console.error("      .venv\\Scripts\\convsearch.exe serve --workspace ./workspace\n");
  } else {
    console.log(
      `\n  Server up — ${health.conversations} conversations ` +
        `(${health.captured_conversations} captured live), ` +
        `index ${health.indexed ? (health.stale_index ? "stale" : "current") : "not built"}.`
    );
  }

  let context;
  try {
    context = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      viewport: null,
      args: [
        `--disable-extensions-except=${EXTENSION_DIR}`,
        `--load-extension=${EXTENSION_DIR}`,
      ],
    });
  } catch (error) {
    // Chromium refuses to reuse a profile another instance still holds. The underlying
    // message is buried under a few hundred command-line flags, so surface the fix.
    if (String(error.message).includes("existing browser session")) {
      console.error(`
  A browser is already using ${path.basename(PROFILE_DIR)}.

  Close any Chromium window this script opened earlier, then run it again. If no
  window is visible, a stray background process is still holding the lock:

      powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name='chrome.exe'\\" |
        Where-Object { $_.CommandLine -like '*.chrome-profile*' } | Stop-Process -Force"
`);
      process.exit(1);
    }
    throw error;
  }

  // Surface content-script errors here rather than leaving them buried in devtools.
  context.on("weberror", (error) => {
    console.error(`  [page error] ${error.error().message}`);
  });

  let worker = context.serviceWorkers()[0];
  if (!worker) worker = await context.waitForEvent("serviceworker", { timeout: 10000 });
  const extensionId = new URL(worker.url()).host;

  // Point the extension at this server, in case it was configured differently before.
  const setup = await context.newPage();
  await setup.goto(`chrome-extension://${extensionId}/options.html`);
  await setup.evaluate(
    (url) =>
      new Promise((resolve) =>
        chrome.storage.local.set({ serverUrl: url, captureEnabled: true }, resolve)
      ),
    SERVER_URL
  );
  await setup.close();

  const page = await context.newPage();
  await page.goto(START_URL);

  console.log(`
  Chromium is open on ${START_URL} with the extension loaded.

  1. Log in to ChatGPT if you are not already. The profile in .chrome-profile/
     persists, so you only do this once.
  2. Open the conversations you want indexed. Each is captured about 1.5s after
     the page settles, then indexed automatically a few seconds later — there is
     no button to press.
  3. Click the convsearch toolbar icon and search. The popup shows "indexing…"
     while a pass is running, and only offers "Rebuild index" if that failed.

  Extension popup:  chrome-extension://${extensionId}/popup.html
  Close the browser window to end this session.
`);

  await context.waitForEvent("close", { timeout: 0 });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
