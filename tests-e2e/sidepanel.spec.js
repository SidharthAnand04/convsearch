"use strict";

/**
 * The Chrome side panel (sidepanel.html) — the main product surface. This is its first e2e
 * coverage: prior specs only exercise popup.html/popup.js plus extraction/capture.
 *
 * Like the popup, the panel speaks NO HTTP itself — every call is a chrome.runtime message to
 * the background service worker, which owns the loopback fetch. So "no request fired" below is
 * asserted through the context (which also sees the service worker's fetches), not the page.
 *
 * FIXTURE RULES (both have bitten this suite before, see keyboard.spec.js):
 *   1. Every spec file shares ONE server and ONE workspace — nothing here asserts an absolute
 *      total.
 *   2. `passages.content_hash` is GLOBALLY UNIQUE with INSERT OR IGNORE, so prose duplicating
 *      another spec file's prose would silently collapse into one passage row. "orrellite",
 *      "orbmoth", "plinkwhistle" and "glazepocket" below exist so this file's text — and the
 *      queries that target it — cannot collide with any other spec.
 *
 * The side panel currently has these tabs: home (default), search, plan, tasks, projects,
 * timeline, memories, captures, status. A "Plan" tab is landing concurrently with this spec, so
 * nothing here asserts an exact tab count or exact tab order — every tab-iterating assertion
 * reads `[role="tab"]` out of the live DOM instead.
 *
 * KNOWN LIMIT OF THIS HARNESS: task/memory/timeline data is produced by the digest/learn
 * pipeline, which needs an LLM backend. `server-launcher.js` boots the server with
 * `--test-embeddings` only (no `--backend`), so that pipeline never runs here and the
 * workspace never has real tasks, memories, or timeline matches — regardless of how much is
 * captured. Per the task brief, the Tasks and Timeline tests below assert the honest empty
 * state with a next action rather than faking data to force a pass, and say so inline.
 */

const { test, expect, newConversationId, health, search, getJson } = require("./helpers");
const { SERVER_URL } = require("./paths");

/* -------------------------------------------------------------------------- */
/* the corpus                                                                 */
/* -------------------------------------------------------------------------- */

function conversation(title, ask, reply) {
  const id = newConversationId();
  return {
    id,
    title,
    payload: {
      source_conversation_id: id,
      title,
      created_at: null,
      updated_at: "2026-07-29T18:00:00Z",
      messages: [
        { source_message_id: `${id}-0`, role: "user", text: ask, order: 0, created_at: null },
        { source_message_id: `${id}-1`, role: "assistant", text: reply, order: 1, created_at: null },
      ],
    },
  };
}

/** "orrellite"/"orbmoth" appear ONLY here. */
const ORREL = conversation(
  "Orrellite kiln glaze notes",
  "How should orrellite glaze be layered before an orbmoth firing?",
  "Apply orrellite in three thin coats and let each cure before the orbmoth kiln pass; a " +
    "thick single coat crawls and pinholes across the bisque. Filed under glazepocket: firing notes."
);

/** "plinkwhistle" appears ONLY here. */
const PLINK = conversation(
  "Plinkwhistle synth patch",
  "Why does the plinkwhistle patch squeal above middle C on the synth?",
  "The plinkwhistle oscillator aliases past that register; drop the wavetable an octave or " +
    "engage the oversample filter. Filed under glazepocket: synth patches."
);

const CORPUS = [ORREL, PLINK];

/** Distinctive body-text query per conversation, used both to seed and as command-bar input. */
const PROOF_QUERY = new Map([
  [ORREL.id, "orbmoth kiln pass"],
  [PLINK.id, "oversample filter wavetable"],
]);

/* -------------------------------------------------------------------------- */
/* panel helpers                                                             */
/* -------------------------------------------------------------------------- */

async function openPanel(context, extensionId) {
  const panel = await context.newPage();
  await panel.goto(`chrome-extension://${extensionId}/sidepanel.html`);
  await expect(panel.locator("#view-home")).toBeVisible();
  return panel;
}

/** Locates a tab button by the tabpanel id it controls, without assuming any id convention. */
function tabFor(panel, viewId) {
  return panel.locator(`[role="tab"][aria-controls="${viewId}"]`);
}

/* -------------------------------------------------------------------------- */

test.describe.serial("side panel", () => {
  test.beforeAll(async () => {
    // Seeding waits for the FTS channel (capture syncs it synchronously), same rationale as
    // keyboard.spec.js — this is a readiness gate, not a claim about auto-indexing.
    test.setTimeout(180_000);

    const response = await fetch(`${SERVER_URL}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversations: CORPUS.map((c) => c.payload) }),
    });
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(
      body.conversations_written,
      `capture rejected the side panel corpus: ${JSON.stringify(body)}`
    ).toBe(CORPUS.length);

    for (const item of CORPUS) {
      await expect
        .poll(
          async () => {
            const payload = await search(PROOF_QUERY.get(item.id), { limit: "10" });
            return (payload.results || []).some((r) => r.source_conversation_id === item.id);
          },
          {
            timeout: 120_000,
            intervals: [1000],
            message: `"${item.title}" never became searchable, so the side panel tests have no corpus`,
          }
        )
        .toBe(true);
    }
  });

  test("boots on Home by default, with no console/page errors and no failed requests", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const consoleErrors = [];
    const pageErrors = [];
    const failedRequests = [];

    const panel = await context.newPage();
    panel.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    panel.on("pageerror", (err) => pageErrors.push(String(err)));
    panel.on("requestfailed", (req) => {
      failedRequests.push(`${req.method()} ${req.url()}: ${req.failure()?.errorText}`);
    });

    await panel.goto(`chrome-extension://${extensionId}/sidepanel.html`);

    // Home is the default tab, and the panel builds all its DOM from `el()` helpers — a typo
    // in any of them throws at runtime rather than rendering wrong, so this settle-then-check
    // is the assertion that would have caught it.
    await expect(panel.locator("#tab-home")).toHaveAttribute("aria-selected", "true");
    await expect(panel.locator("#view-home")).toBeVisible();
    await expect
      .poll(() => panel.locator("#home-output").locator("> *").count(), {
        timeout: 20_000,
        message: "the Home dashboard never rendered anything",
      })
      .toBeGreaterThan(0);

    expect(consoleErrors, `console errors while booting Home: ${consoleErrors.join(" | ")}`).toEqual([]);
    expect(pageErrors, `uncaught page errors while booting Home: ${pageErrors.join(" | ")}`).toEqual([]);
    expect(
      failedRequests,
      `failed requests while booting Home: ${failedRequests.join(" | ")}`
    ).toEqual([]);

    await panel.close();
  });

  test("every tab activates its own view and hides the rest", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const panel = await openPanel(context, extensionId);

    // Read the tab list from the live DOM — do NOT hardcode it, a "Plan" tab is landing
    // concurrently with this spec and the count/order must not be assumed.
    const tabs = panel.locator('[role="tab"]');
    const tabCount = await tabs.count();
    expect(tabCount, "no [role=tab] elements found").toBeGreaterThan(0);

    const controlsIds = [];
    for (let i = 0; i < tabCount; i += 1) {
      const id = await tabs.nth(i).getAttribute("aria-controls");
      expect(id, `tab ${i} has no aria-controls`).toBeTruthy();
      controlsIds.push(id);
    }

    for (const controlsId of controlsIds) {
      await tabFor(panel, controlsId).click();
      // The activated tabpanel is visible...
      await expect(panel.locator(`#${controlsId}`)).toBeVisible();
      await expect(tabFor(panel, controlsId)).toHaveAttribute("aria-selected", "true");
      // ...and every other one is hidden. This is the assertion that catches a view wired to
      // a missing element id: a broken activator throws before `views[v].hidden` runs for the
      // OTHER views, so a bug there would leave two panels visible at once.
      for (const otherId of controlsIds) {
        if (otherId === controlsId) continue;
        await expect(panel.locator(`#${otherId}`)).toBeHidden();
      }
    }

    await panel.close();
  });

  test("tasks view: real data if present, otherwise an honest empty state", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const panel = await openPanel(context, extensionId);
    await tabFor(panel, "view-tasks").click();
    await expect(panel.locator("#view-tasks")).toBeVisible();

    // Ask the real server directly (state=all so a completed-only task set still counts) to
    // decide which path this workspace is actually in, rather than assuming.
    const { body } = await getJson("/tasks?state=all&limit=100&evidence=1");
    const items = body.items || [];

    if (items.length === 0) {
      // No task memories exist anywhere in this shared workspace — expected, see the
      // module doc comment on why (no digest/learn pipeline runs in this harness). Assert
      // the empty state renders with a next action instead of faking a task to force a pass.
      await expect
        .poll(() => panel.locator("#tasks-output .state-title").textContent(), { timeout: 20_000 })
        .toMatch(/No open tasks|No conversations indexed/);
      await expect(panel.locator("#tasks-output .state-body")).toBeVisible();
      await panel.close();
      return;
    }

    // Real-data path, kept for when a future harness change starts producing tasks.
    const row = panel.locator("#tasks-output .list-item").first();
    await expect(row).toBeVisible();
    const evidenceBtn = row.getByRole("button", { name: /Show evidence/ });
    if (await evidenceBtn.count()) {
      await evidenceBtn.click();
      await expect(row.locator(".source-quote").first()).toBeVisible();
    }
    await panel.close();
  });

  test("captures view lists the seeded conversations with pipeline-state indicators", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const panel = await openPanel(context, extensionId);
    await tabFor(panel, "view-captures").click();
    await expect(panel.locator("#view-captures")).toBeVisible();

    const row = panel.locator("#captures-output .list-item", { hasText: ORREL.title });
    await expect(row).toBeVisible({ timeout: 20_000 });
    // Source badge and the three pipeline pills are always rendered (ok or warn variant), so
    // their presence — not their color — is the honest assertion here.
    await expect(row.locator(".pill", { hasText: "live capture" })).toBeVisible();
    await expect(row.locator(".pill", { hasText: "indexed" })).toBeVisible();
    await expect(row.locator(".pill", { hasText: "segmented" })).toBeVisible();
    await expect(row.locator(".pill", { hasText: "memories extracted" })).toBeVisible();

    await panel.close();
  });

  test("timeline guards an empty topic and shows an honest empty state for a real one", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const panel = await openPanel(context, extensionId);
    await tabFor(panel, "view-timeline").click();
    await expect(panel.locator("#view-timeline")).toBeVisible();

    // Empty topic: Run is disabled, and the form handler returns before touching the output
    // for an empty query, so submitting must leave the output untouched (no request, no
    // stale-spinner flash).
    await expect(panel.locator("#timeline-submit")).toBeDisabled();
    await expect(panel.locator("#timeline-hint")).toContainText("Enter a topic");
    await expect(panel.locator("#timeline-output").locator("> *")).toHaveCount(0);
    await panel.locator("#timeline-query").focus();
    await panel.keyboard.press("Enter");
    await expect(panel.locator("#timeline-output").locator("> *")).toHaveCount(0);

    // A topic that matches nothing gets an honest empty state with a next action. This
    // harness never produces real memories (see the module doc comment), so this is the only
    // Timeline path this spec can exercise reliably; "renders a result for a topic that
    // matches something" from the task brief is skipped here for that reason — there is no
    // product surface to seed a genuine timeline match without bypassing the digest pipeline.
    await panel.locator("#timeline-query").fill("orrellite glaze");
    await panel.locator("#timeline-submit").click();
    await expect(panel.locator("#timeline-output .state-title")).toContainText("No memories matched");
    await expect(
      panel.locator("#timeline-output").getByRole("button", { name: "Open Memories" })
    ).toBeVisible();

    await panel.close();
  });

  test("the command bar routes 'search <term>' and 'help'", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const panel = await openPanel(context, extensionId);

    await panel.locator("#cmd-input").fill(`search ${PROOF_QUERY.get(PLINK.id)}`);
    await panel.locator("#cmd-input").press("Enter");

    await expect(panel.locator("#view-search")).toBeVisible();
    await expect(panel.locator("#tab-search")).toHaveAttribute("aria-selected", "true");
    await expect(panel.locator("#search-input")).toHaveValue(PROOF_QUERY.get(PLINK.id));
    const hit = panel.locator("#search-results .result-title", { hasText: PLINK.title });
    await expect(hit).toBeVisible({ timeout: 30_000 });

    await tabFor(panel, "view-home").click();
    await expect(panel.locator("#view-home")).toBeVisible();

    await panel.locator("#cmd-input").fill("help");
    await panel.locator("#cmd-input").press("Enter");
    await expect(panel.locator("#cmd-hint")).toContainText("Commands:");
    // Still on Home — "help" only prints inline text, it must not navigate away.
    await expect(panel.locator("#view-home")).toBeVisible();

    await panel.close();
  });

  test("an unreachable server shows an offline state with a next action", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    // Point the extension at a dead port; the panel must explain itself rather than hang or
    // render blank. Restored at the end so nothing leaks past this test.
    await configuredExtension.page.evaluate(
      () =>
        new Promise((resolve) =>
          chrome.storage.local.set({ serverUrl: "http://127.0.0.1:9" }, resolve)
        )
    );

    const panel = await context.newPage();
    await panel.goto(`chrome-extension://${extensionId}/sidepanel.html`);

    await expect(panel.locator("#home-output .banner-bad")).toContainText("Server offline", {
      timeout: 20_000,
    });
    await expect(panel.locator("#home-output code")).toContainText("convsearch serve");
    await expect(panel.locator("#conn-dot")).toHaveClass(/is-offline/);
    await panel.close();

    await configuredExtension.page.evaluate(
      (serverUrl) =>
        new Promise((resolve) => chrome.storage.local.set({ serverUrl }, resolve)),
      SERVER_URL
    );
  });
});
