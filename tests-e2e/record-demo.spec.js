"use strict";

/**
 * DEMO-ASSET RECORDER — not a test of behaviour, a producer of marketing assets.
 *
 * Run explicitly (it is skipped by the normal suite via the RECORD_DEMO gate):
 *
 *   RECORD_DEMO=1 npx playwright test tests-e2e/record-demo.spec.js
 *
 * Playwright's `webServer` (see playwright.config.js -> server-launcher.js) boots the REAL
 * Python server against a DISPOSABLE .tmp-workspace and inits it, exactly like the rest of the
 * suite. This file then seeds that workspace over the loopback /capture endpoint, waits for the
 * content to become searchable, and drives the NEW side-panel UI to produce:
 *
 *   docs/screenshots/panel.png        dark, a search with the "Ranked because" card + score chart
 *   docs/screenshots/panel-light.png  the same, in the light theme (shows the theme toggle)
 *   docs/screenshots/store-1280x800.png  Chrome Web Store tile, the panel framed on a canvas
 *   docs/screenshots/demo.webm        ~25s screen recording of the new flow
 *   site/vendor/demo.webm             copy for the landing page (task's stable name)
 *   site/media/demo.webm              copy the landing HTML actually references
 *   site/media/demo-poster.png        poster frame for the landing <video>
 *   docs/screenshots/landing*.png     the local landing site at desktop width
 *
 * ROBUSTNESS CONTRACT: the PNGs matter most, so each is taken as early as possible and every
 * later interaction is wrapped so a mid-flow failure still flushes the video and still leaves the
 * screenshots on disk. The three producers are independent test() blocks so a failure in one does
 * not deny the others — in particular the landing shots need no extension or model at all.
 *
 * The "Ranked because" reason is only serialised when explain=1 (server/serializers.py), so the
 * flow ticks the Explain checkbox before searching — that is what surfaces both the reason
 * sentence and the score mini-chart.
 */

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { test, expect, chromium } = require("@playwright/test");

const { REPO_ROOT, EXTENSION_DIR, SERVER_URL } = require("./paths");

const GATED = process.env.RECORD_DEMO === "1";

const DOCS = path.join(REPO_ROOT, "docs", "screenshots");
const SITE_VENDOR = path.join(REPO_ROOT, "site", "vendor");
const SITE_MEDIA = path.join(REPO_ROOT, "site", "media");
const VIDEO_TMP = path.join(REPO_ROOT, "tests-e2e", ".tmp-demo-video");

fs.mkdirSync(DOCS, { recursive: true });

/* -------------------------------------------------------------------------- */
/* demo corpus — realistic ChatGPT history that makes retrieval look good.    */
/* This file runs standalone against a fresh workspace, so the vocabulary is   */
/* free to be readable; nothing else is in the index to collide with.          */
/* -------------------------------------------------------------------------- */

function convo(title, ask, reply) {
  const hex = (n) =>
    Array.from({ length: n }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  const id = `${hex(8)}-${hex(4)}-4${hex(3)}-a${hex(3)}-${hex(12)}`;
  return {
    id,
    title,
    payload: {
      source_conversation_id: id,
      title,
      created_at: "2026-07-14T09:00:00Z",
      updated_at: "2026-07-27T16:30:00Z",
      messages: [
        { source_message_id: `${id}-0`, role: "user", text: ask, order: 0, created_at: "2026-07-14T09:00:00Z" },
        { source_message_id: `${id}-1`, role: "assistant", text: reply, order: 1, created_at: "2026-07-14T09:00:20Z" },
      ],
    },
  };
}

const RUST = convo(
  "Sharing state across async tasks",
  "What is the idiomatic way to share mutable state between asynchronous Rust tasks without data races?",
  "Wrap the value in Arc<Mutex<T>> so ownership is shared across tasks and every access is " +
    "serialised through the lock. Clone the Arc into each spawned task. Watch for deadlocks: they " +
    "appear the moment two tasks acquire their locks in the opposite order, so always lock in a " +
    "consistent, documented sequence."
);
const PG = convo(
  "Postgres connection pooling under load",
  "Our API stalls under load and I think it is exhausting Postgres connections — how should I pool them?",
  "Put pgbouncer in transaction pooling mode in front of Postgres and size the pool to the number " +
    "of CPU cores, not the number of app workers. Then enable pg_stat_statements and sort by total " +
    "execution time to find the statements actually holding connections open."
);
const RAG = convo(
  "Choosing a vector database for RAG",
  "Which vector database should I use for a retrieval-augmented generation feature over our docs?",
  "For a local-first setup a flat FAISS IndexFlatIP over sentence embeddings is exact and hard to " +
    "beat until you pass a few hundred thousand vectors; only then reach for HNSW or a hosted store. " +
    "Keep the embeddings and the source text in the same store so retrieval stays reproducible."
);
const REACT = convo(
  "useEffect cleanup and race conditions",
  "Why does my React component fetch twice and sometimes render stale data from an old request?",
  "Return a cleanup function from useEffect that flips an `ignore` flag, and check it before you " +
    "call setState — that discards the response from a request that was superseded while in flight. " +
    "In development, StrictMode intentionally double-invokes effects to surface exactly this bug."
);
const LIMIT = convo(
  "Designing a fair rate limiter",
  "How do I build a rate limiter that smooths bursts instead of hard-cutting users at the boundary?",
  "Use a token-bucket: refill tokens at a steady rate and let the bucket depth absorb short bursts. " +
    "It is smoother than a fixed window, which double-counts traffic straddling the reset instant, " +
    "and cheaper to reason about than a sliding-log limiter."
);

const CORPUS = [RUST, PG, RAG, REACT, LIMIT];
const DEMO_QUERY = "share mutable state between async tasks";

/* -------------------------------------------------------------------------- */
/* small utilities                                                            */
/* -------------------------------------------------------------------------- */

const beat = (page, ms = 900) => page.waitForTimeout(ms);

async function getJson(pathAndQuery) {
  const res = await fetch(`${SERVER_URL}${pathAndQuery}`);
  return { status: res.status, body: await res.json() };
}

async function searchable(query, convId) {
  const { body } = await getJson(`/search?q=${encodeURIComponent(query)}&limit=10`);
  return (body.results || []).some((r) => r.source_conversation_id === convId);
}

/** Launch a persistent context with the unpacked extension loaded (mirrors helpers.js). */
async function launchExtensionContext({ recordVideoDir, size }) {
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "convsearch-demo-"));
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: false,
    viewport: size,
    args: [
      `--disable-extensions-except=${EXTENSION_DIR}`,
      `--load-extension=${EXTENSION_DIR}`,
      "--no-first-run",
      "--no-default-browser-check",
    ],
    serviceWorkers: "allow",
    ...(recordVideoDir ? { recordVideo: { dir: recordVideoDir, size } } : {}),
  });
  let [worker] = context.serviceWorkers();
  if (!worker) worker = await context.waitForEvent("serviceworker", { timeout: 30000 });
  const extensionId = new URL(worker.url()).host;
  return { context, extensionId, userDataDir };
}

/** Point the extension at the disposable test server and enable capture. */
async function configure(context, extensionId) {
  const page = await context.newPage();
  await page.goto(`chrome-extension://${extensionId}/options.html`);
  await page.evaluate(
    (serverUrl) =>
      new Promise((resolve) =>
        chrome.storage.local.set({ serverUrl, captureEnabled: true, "convsearch:theme": "dark" }, resolve)
      ),
    SERVER_URL
  );
  await page.close();
}

/** Open the panel on a query with Explain on, and wait for a real result to render. */
async function drivePanelToResults(panel, extensionId, { query = DEMO_QUERY } = {}) {
  await panel.goto(`chrome-extension://${extensionId}/sidepanel.html`);
  await expect(panel.locator("#view-home")).toBeVisible();
  await beat(panel, 700);

  // Move to Ask & Search via the icon rail.
  await panel.locator("#tab-search").click();
  await expect(panel.locator("#view-search")).toBeVisible();
  await beat(panel, 500);

  // Explain on -> "Ranked because" reason + score mini-chart both render.
  await panel.locator("#explain").check();

  // Type deliberately so the video reads well.
  await panel.locator("#search-input").click();
  await panel.locator("#search-input").pressSequentially(query, { delay: 55 });

  // Wait for a genuine result card.
  await expect(panel.locator("#search-results .result").first()).toBeVisible({ timeout: 30000 });
  // And for the reason sentence that the explain path adds.
  await expect(panel.locator("#search-results .why-label").first()).toBeVisible({ timeout: 15000 });
  await beat(panel, 900);
}

/* -------------------------------------------------------------------------- */
/* seed + gate                                                                */
/* -------------------------------------------------------------------------- */

test.describe.serial("demo assets", () => {
  test.skip(!GATED, "asset recorder — set RECORD_DEMO=1 to run");

  test.beforeAll(async () => {
    test.setTimeout(240000);
    const res = await fetch(`${SERVER_URL}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversations: CORPUS.map((c) => c.payload) }),
    });
    expect(res.status, "capture endpoint rejected the demo corpus").toBe(200);
    const body = await res.json();
    expect(body.conversations_written, `capture wrote unexpected count: ${JSON.stringify(body)}`).toBe(
      CORPUS.length
    );

    for (const c of CORPUS) {
      await expect
        .poll(() => searchable(c.id === RUST.id ? DEMO_QUERY : c.title, c.id), {
          timeout: 180000,
          intervals: [1500],
          message: `"${c.title}" never became searchable — cannot record a demo without a corpus`,
        })
        .toBe(true);
    }
  });

  /* ---- producer 1: panel PNGs (dark + light) and the demo video ---------- */
  test("panel screenshots + demo video (360x720)", async () => {
    test.setTimeout(300000);
    fs.rmSync(VIDEO_TMP, { recursive: true, force: true });
    fs.mkdirSync(VIDEO_TMP, { recursive: true });

    // 440x780: a real Chrome side panel can be dragged this wide, and at the default ~360 the
    // result-head flex squeezes the title to one character per line. 440 gives the title/meta
    // row room to read (matching the store-tile shot's quality).
    const size = { width: 440, height: 780 };
    const { context, extensionId, userDataDir } = await launchExtensionContext({
      recordVideoDir: VIDEO_TMP,
      size,
    });
    await configure(context, extensionId);

    const panel = await context.newPage();
    let video = panel.video();

    try {
      await drivePanelToResults(panel, extensionId);

      // ── PNGs FIRST (most important). Dark hero, results + ranked-because. ──
      await panel.screenshot({ path: path.join(DOCS, "panel.png") });
      await panel.screenshot({ path: path.join(SITE_MEDIA, "demo-poster.png") });

      // Everything below is video choreography; a failure here must not lose the PNGs
      // already written or the webm still recording.
      try {
        // Reveal the score breakdown explicitly (it is auto-open with Explain, but clicking
        // it reads as a deliberate "expand the detail" beat on camera).
        const toggle = panel.locator("#search-results .why-toggle").first();
        if (await toggle.count()) {
          await toggle.click();
          await beat(panel, 700);
          await toggle.click(); // collapse, then...
          await beat(panel, 400);
          await toggle.click(); // ...re-open so the chart is showing at rest
          await beat(panel, 700);
        }

        // Inspect the top result — conversation > segment > passage detail.
        const inspect = panel.locator("#search-results .result").first().getByRole("button", { name: /Inspect/ });
        if (await inspect.count()) {
          await inspect.click();
          await beat(panel, 1400);
        }

        // Light theme — and grab panel-light.png while the results are on screen.
        await panel.locator("#theme-toggle").click();
        await beat(panel, 900);
        await panel.screenshot({ path: path.join(DOCS, "panel-light.png") });
        await beat(panel, 500);
        // Back to dark for the rest of the reel.
        await panel.locator("#theme-toggle").click();
        await beat(panel, 700);

        // Tour a couple of views through the icon rail.
        for (const view of ["timeline", "captures", "status", "privacy"]) {
          await panel.locator(`#tab-${view}`).click();
          await beat(panel, 900);
        }

        // Back to search and fire an Ask. Without a local model this lands on the honest
        // "answer needs a local model" note; with one it shows a grounded, cited answer.
        // Either way it demonstrates the grounded-answer surface, so we don't fail on it.
        await panel.locator("#tab-search").click();
        await beat(panel, 500);
        await panel.locator("#ask-submit").click();
        await beat(panel, 2600);
      } catch (err) {
        console.error(`[demo] video choreography aborted (PNGs already saved): ${err.message}`);
      }
    } finally {
      await context.close(); // flushes the webm
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }

    // Move the flushed video into place.
    if (video) {
      try {
        const src = await video.path();
        const outDocs = path.join(DOCS, "demo.webm");
        fs.copyFileSync(src, outDocs);
        fs.mkdirSync(SITE_VENDOR, { recursive: true });
        fs.copyFileSync(src, path.join(SITE_VENDOR, "demo.webm"));
        fs.mkdirSync(SITE_MEDIA, { recursive: true });
        fs.copyFileSync(src, path.join(SITE_MEDIA, "demo.webm"));
        console.log(`[demo] webm saved (${fs.statSync(outDocs).size} bytes)`);
      } catch (err) {
        console.error(`[demo] video flush/copy FAILED: ${err.message}`);
      }
    } else {
      console.error("[demo] no video handle — webm not produced");
    }
    fs.rmSync(VIDEO_TMP, { recursive: true, force: true });
  });

  /* ---- producer 2: the Chrome Web Store tile (1280x800) ------------------ */
  test("store tile 1280x800", async () => {
    test.setTimeout(180000);
    const size = { width: 1280, height: 800 };
    const { context, extensionId, userDataDir } = await launchExtensionContext({ size });
    await configure(context, extensionId);
    const panel = await context.newPage();
    try {
      await drivePanelToResults(panel, extensionId);
      // Frame the narrow panel as a centred card on a soft canvas — presentation only,
      // no product data is altered.
      await panel.addStyleTag({
        content: `
          html, body { height: 100%; }
          body {
            display: flex; align-items: center; justify-content: center;
            background: radial-gradient(120% 120% at 50% 0%, #1b2130 0%, #0c0f16 60%, #070a10 100%);
          }
          .app-shell {
            flex: none !important;
            width: 440px; height: 720px;
            border-radius: 18px; overflow: hidden;
            box-shadow: 0 40px 120px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.06);
          }
        `,
      });
      await beat(panel, 600);
      await panel.screenshot({ path: path.join(DOCS, "store-1280x800.png") });
    } finally {
      await context.close();
      fs.rmSync(userDataDir, { recursive: true, force: true });
    }
  });

  /* ---- producer 3: the landing site (needs no extension, no model) ------- */
  test("landing screenshots", async () => {
    test.setTimeout(180000);
    const siteDir = path.join(REPO_ROOT, "site");
    const port = 8799;
    const server = spawn(
      path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
      ["-m", "http.server", String(port), "--bind", "127.0.0.1"],
      { cwd: siteDir, stdio: "ignore" }
    );

    const browser = await chromium.launch({ headless: true });
    try {
      // Wait for the static server to answer.
      for (let i = 0; i < 40; i += 1) {
        try {
          const r = await fetch(`http://127.0.0.1:${port}/index.html`);
          if (r.ok) break;
        } catch {
          /* not up yet */
        }
        await new Promise((r) => setTimeout(r, 250));
      }

      const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
      const page = await ctx.newPage();
      await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: "networkidle" });
      await page.waitForTimeout(1200);

      // Default (as-shipped) theme.
      await page.screenshot({ path: path.join(DOCS, "landing.png") });
      await page.screenshot({ path: path.join(DOCS, "landing-full.png"), fullPage: true });

      // Toggle to the other theme and capture both named variants deterministically.
      const themeNow = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
      await page.locator("#theme-toggle").click();
      await page.waitForTimeout(700);
      const themeAfter = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));

      const shotFor = async (theme) => {
        const current = await page.evaluate(() => document.documentElement.getAttribute("data-theme"));
        if (current !== theme) {
          await page.locator("#theme-toggle").click();
          await page.waitForTimeout(700);
        }
        await page.screenshot({ path: path.join(DOCS, `landing-${theme}.png`) });
      };
      await shotFor("dark");
      await shotFor("light");

      console.log(`[demo] landing theme default=${themeNow} afterToggle=${themeAfter}`);
      await ctx.close();
    } finally {
      await browser.close();
      server.kill();
    }
  });
});
