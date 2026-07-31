"use strict";

/**
 * The keyboard-driven popup — the headline interaction.
 *
 * What is covered here:
 *   - the popup opens with the caret already in the query box, so you can just type
 *   - ArrowDown/ArrowUp and Ctrl+J/Ctrl+K move a VISIBLE selection
 *   - Enter opens the selected result in a real tab at its chatgpt.com deep link
 *   - Escape clears the query, and a second Escape asks the popup to close
 *   - as-you-type search returns results with nobody pressing Search
 *   - overlapping as-you-type requests render the FINAL query's results, never an
 *     earlier slower one
 *
 * FIXTURE RULES (both of these have bitten this suite before):
 *   1. Every spec file shares ONE server and ONE workspace, so nothing here asserts an
 *      absolute conversation/message total.
 *   2. `passages.content_hash` is GLOBALLY UNIQUE with INSERT OR IGNORE, so prose that
 *      duplicates another spec file's prose would silently collapse into one passage row.
 *      The nonsense tokens below ("flitterwake", "grumbleshanks", …) exist so this file's
 *      text, and the queries that target it, cannot collide with any other spec.
 */

const { test, expect, newConversationId, search } = require("./helpers");
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
        {
          source_message_id: `${id}-0`,
          role: "user",
          text: ask,
          order: 0,
          created_at: null,
        },
        {
          source_message_id: `${id}-1`,
          role: "assistant",
          text: reply,
          order: 1,
          created_at: null,
        },
      ],
    },
  };
}

/** "flitterwake" appears ONLY here. */
const RINK = conversation(
  "Flitterwake rink resurfacing cadence",
  "How often should the flitterwake rink be resurfaced during a doubleheader evening?",
  "Resurface between periods. On a flitterwake sheet the blade shave depth decides how " +
    "true the surface stays, far more than the flood water temperature does. " +
    "Filed under quibblesnout: ice surface craft."
);

/** "grumbleshanks" appears ONLY here. */
const DRONE = conversation(
  "Grumbleshanks drone tuning",
  "Why does the grumbleshanks drone string go sour after about an hour of playing?",
  "Cotton wadding on the grumbleshanks wheel packs down and hoards rosin dust. Re-cotton " +
    "the drone string, then re-rosin very lightly and let it settle. " +
    "Filed under quibblesnout: reed and wheel maintenance."
);

const LANTERN = conversation(
  "Floating lantern release paperwork",
  "What paperwork does a floating lantern release need on a municipal boating lake?",
  "A burn permit, a notice to boat traffic, and a written retrieval plan for spent frames. " +
    "Wardens generally insist the retrieval crew be named on the application itself. " +
    "Filed under quibblesnout: permits and paperwork."
);

const PUCK = conversation(
  "Espresso puck craters",
  "Why does my espresso puck show a deep crater in the middle once extraction finishes?",
  "That crater is channelling caused by uneven grounds distribution. Rake the bed with a " +
    "needle distributor and tamp dead level before you lock the basket in. " +
    "Filed under quibblesnout: extraction diagnostics."
);

const CORPUS = [RINK, DRONE, LANTERN, PUCK];

/** Distinctive body-text query per conversation, used to prove it reached the index. */
const PROOF_QUERY = new Map([
  [RINK.id, "flitterwake blade shave depth"],
  [DRONE.id, "grumbleshanks wheel cotton wadding rosin"],
  [LANTERN.id, "retrieval plan for spent lantern frames"],
  [PUCK.id, "needle distributor tamp level channelling"],
]);

/** Exactly what the popup asks the server for, so node-side and popup-side lists compare. */
const POPUP_PARAMS = { limit: "10", passages: "2" };

/**
 * The as-you-type race pair. PREFIX_QUERY is a proper prefix of FINAL_QUERY, so typing
 * FINAL_QUERY naturally issues PREFIX_QUERY first; "flitterwake" is unique to RINK and
 * "grumbleshanks" to DRONE, so the two queries have genuinely different answers.
 */
const PREFIX_QUERY = "flitterwake";
const FINAL_QUERY = "flitterwake grumbleshanks";

/**
 * Matches all four fixtures and nothing else lexically.
 *
 * The selection tests need several rows to move between, and they must get them from THIS
 * file's own corpus: running `keyboard.spec.js` on its own gave a two-row list and the
 * movement test failed for want of a third row. "quibblesnout" is the one token every
 * fixture above shares — each reply still has entirely distinct prose, so no two passages
 * collide on `content_hash`.
 */
const MOVE_QUERY = "quibblesnout";

/* -------------------------------------------------------------------------- */
/* seeding                                                                    */
/* -------------------------------------------------------------------------- */

test.beforeAll(async () => {
  // Seeding waits for a genuine auto-index pass on a freshly started server, which is
  // slower than any single test; give the hook its own budget.
  test.setTimeout(300_000);
  const started = Date.now();
  const elapsed = () => `${((Date.now() - started) / 1000).toFixed(1)}s`;

  const response = await fetch(`${SERVER_URL}/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversations: CORPUS.map((c) => c.payload) }),
  });
  expect(response.status).toBe(200);
  const body = await response.json();
  expect(body.conversations_written, `capture rejected the keyboard corpus: ${JSON.stringify(body)}`)
    .toBe(CORPUS.length);

  // Poll the real search endpoint until every fixture comes back for text that exists only
  // in its body. This is a readiness gate, NOT a claim about auto-indexing: capture syncs
  // FTS synchronously, so the lexical channel can answer before any vector pass runs.
  // Auto-indexing has its own coverage in autoindex.spec.js and scenario.spec.js.
  for (const item of CORPUS) {
    await expect
      .poll(
        async () => {
          const payload = await search(PROOF_QUERY.get(item.id), POPUP_PARAMS);
          return (payload.results || []).some((r) => r.source_conversation_id === item.id);
        },
        {
          timeout: 120_000,
          intervals: [1000],
          message: `"${item.title}" never became searchable, so the keyboard tests have no corpus`,
        }
      )
      .toBe(true);
  }
  console.log(`[keyboard.spec] corpus of ${CORPUS.length} seeded and searchable in ${elapsed()}`);
});

/* -------------------------------------------------------------------------- */
/* popup helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Opens the popup page with `window.close()` instrumented.
 *
 * The popup legitimately closes itself after opening a result and on a second Escape. A
 * Playwright page that closed mid-assertion cannot be inspected, so record the call and
 * do NOT forward it; `closeRequests()` is then the assertion for "the popup closed".
 */
async function openPopup(context, extensionId) {
  const popup = await context.newPage();
  await popup.addInitScript(() => {
    globalThis.__closeCalls = 0;
    window.close = () => {
      globalThis.__closeCalls += 1;
    };
  });
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  return popup;
}

const closeRequests = (popup) => popup.evaluate(() => globalThis.__closeCalls);

const activeId = (popup) =>
  popup.evaluate(() => (document.activeElement ? document.activeElement.id : null));

/** Index of the row carrying `.selected`, or -1. Mirrors what the user actually sees. */
const selectedIndex = (popup) =>
  popup.evaluate(() =>
    Array.from(document.querySelectorAll("#result-list .result")).findIndex((el) =>
      el.classList.contains("selected")
    )
  );

const renderedTitles = (popup) =>
  popup.evaluate(() =>
    Array.from(document.querySelectorAll("#result-list .result .result-title")).map(
      (el) => el.textContent
    )
  );

/** Titles the server returns for `query`, in order, as the popup would render them. */
async function serverTitles(query) {
  const payload = await search(query, POPUP_PARAMS);
  return (payload.results || []).map((r) => r.title || "(untitled)");
}

/** Waits for at least `n` rendered results. */
async function waitForResults(popup, n = 1) {
  await expect
    .poll(() => popup.locator("#result-list .result").count(), {
      timeout: 30_000,
      intervals: [250],
      message: `fewer than ${n} results ever rendered`,
    })
    .toBeGreaterThanOrEqual(n);
}

/**
 * Puts `query` in the box and waits until the list on screen is the answer to THAT query.
 *
 * This matters for every selection assertion. As-you-type fires a search per debounce
 * window, and each render pre-selects row 0, so pressing arrow keys while an intermediate
 * query's response is still on its way produces a selection that jumps back to 0 and a row
 * count that changes under the test. Using `fill` issues a single input event, and polling
 * the rendered titles against what the server returns for the final query proves the list
 * has stopped moving before any key is pressed. Returns the settled titles.
 */
async function settleQuery(popup, query) {
  const expected = await serverTitles(query);
  expect(expected.length, `"${query}" returned nothing from the server`).toBeGreaterThan(0);
  await popup.locator("#query").fill(query);
  await expect
    .poll(() => renderedTitles(popup), {
      timeout: 30_000,
      intervals: [250],
      message: `the popup never settled on the results for "${query}"`,
    })
    .toEqual(expected);
  return expected;
}

/* -------------------------------------------------------------------------- */

test.describe.serial("keyboard-driven popup", () => {
  test("opens focused on the query field, so typing goes straight into it", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await openPopup(context, extensionId);

    // Focus is asserted, not assumed: `autofocus` is unreliable in an extension popup and
    // popup.js focuses again from its storage callback, so poll rather than read once.
    await expect
      .poll(() => activeId(popup), {
        timeout: 15_000,
        message: "the popup did not put focus in the query box",
      })
      .toBe("query");

    // The behavioural claim, not just the DOM one: keystrokes land in the box with no
    // click and no locator focusing anything first.
    await popup.keyboard.type("flitterwake");
    await expect(popup.locator("#query")).toHaveValue("flitterwake");

    await popup.close();
  });

  test("as-you-type search returns results with no submit press", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await openPopup(context, extensionId);
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    // Nobody clicks #submit and nobody presses Enter anywhere in this test.
    const query = PROOF_QUERY.get(RINK.id);
    const expected = await serverTitles(query);
    await popup.locator("#query").pressSequentially(query, { delay: 25 });

    await waitForResults(popup, 1);
    // Typing character by character issues a search per debounce window; wait until the
    // list is the answer to the whole query rather than to some prefix of it.
    await expect
      .poll(() => renderedTitles(popup), { timeout: 30_000, intervals: [250] })
      .toEqual(expected);
    await expect(popup.locator("#status")).toContainText("result");
    await expect(
      popup.locator("#result-list .result-title", { hasText: RINK.title })
    ).toHaveAttribute("href", `https://chatgpt.com/c/${RINK.id}`);

    await popup.close();
  });

  test("arrow keys and Ctrl+J / Ctrl+K move a visible selection", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await openPopup(context, extensionId);
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    const titles = await settleQuery(popup, MOVE_QUERY);
    expect(
      titles.length,
      `"${MOVE_QUERY}" should match all ${CORPUS.length} fixtures; there is nothing to ` +
        "move through otherwise"
    ).toBeGreaterThanOrEqual(3);
    const count = titles.length;

    // The top hit is pre-selected so Enter is one keystroke from opening it.
    await expect.poll(() => selectedIndex(popup)).toBe(0);
    const rows = popup.locator("#result-list .result");
    await expect(rows.nth(0)).toHaveAttribute("aria-selected", "true");

    // "Visible" means the user can see which row is live: the selected class must be on
    // exactly one row, and the query box must point at it for screen readers.
    await expect(popup.locator("#result-list .result.selected")).toHaveCount(1);
    await expect(popup.locator("#query")).toHaveAttribute("aria-activedescendant", "result-0");

    await popup.keyboard.press("ArrowDown");
    await expect.poll(() => selectedIndex(popup)).toBe(1);
    await expect(rows.nth(1)).toHaveAttribute("aria-selected", "true");
    await expect(rows.nth(0)).toHaveAttribute("aria-selected", "false");
    await expect(popup.locator("#query")).toHaveAttribute("aria-activedescendant", "result-1");

    await popup.keyboard.press("Control+j");
    await expect.poll(() => selectedIndex(popup)).toBe(2);

    await popup.keyboard.press("Control+k");
    await expect.poll(() => selectedIndex(popup)).toBe(1);

    await popup.keyboard.press("ArrowUp");
    await expect.poll(() => selectedIndex(popup)).toBe(0);

    // Wrapping: Up from the first row lands on the last.
    await popup.keyboard.press("ArrowUp");
    await expect.poll(() => selectedIndex(popup)).toBe(count - 1);
    await expect(popup.locator("#result-list .result.selected")).toHaveCount(1);

    // Movement must not steal focus from the query box — typing has to keep working.
    expect(await activeId(popup)).toBe("query");

    await popup.close();
  });

  test("Enter opens the selected result at its chatgpt.com deep link", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    // Keep the opened tab off the internet. The URL is still the real chatgpt.com URL,
    // which is what the assertion is about.
    await context.route("https://chatgpt.com/**", (route) =>
      route.fulfill({ status: 200, contentType: "text/html", body: "<title>stub</title>" })
    );

    const popup = await openPopup(context, extensionId);
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    const titles = await settleQuery(popup, MOVE_QUERY);

    // Walk the selection onto the DRONE row with arrow keys only, then open it. The claim
    // being tested is that Enter opens whichever row the keyboard selection is on — so the
    // target is picked by identity, not by position.
    const target = titles.indexOf(DRONE.title);
    expect(
      target,
      `"${DRONE.title}" was not in the rendered results; got ${JSON.stringify(titles)}`
    ).toBeGreaterThanOrEqual(0);

    await popup.keyboard.press("Home");
    await expect.poll(() => selectedIndex(popup)).toBe(0);
    for (let i = 0; i < target; i += 1) {
      await popup.keyboard.press("ArrowDown");
    }
    await expect.poll(() => selectedIndex(popup)).toBe(target);

    const expectedUrl = `https://chatgpt.com/c/${DRONE.id}`;
    // The selected row's href comes from the server's `url` field, and popup.js only sets
    // it when the value starts with https://chatgpt.com/ — so this also proves the server
    // built the deep link.
    await expect(popup.locator("#result-list .result.selected .result-title")).toHaveAttribute(
      "href",
      expectedUrl
    );

    const [opened] = await Promise.all([
      context.waitForEvent("page", { timeout: 20_000 }),
      popup.keyboard.press("Enter"),
    ]);
    await expect
      .poll(() => opened.url(), {
        timeout: 20_000,
        message: "Enter opened a tab, but not at the selected conversation's URL",
      })
      .toBe(expectedUrl);

    // Opening a result in the foreground closes the popup.
    await expect.poll(() => closeRequests(popup), { timeout: 10_000 }).toBeGreaterThan(0);

    await opened.close();
    await popup.close();
  });

  test("Escape clears the query, and a second Escape closes the popup", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await openPopup(context, extensionId);
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    await settleQuery(popup, FINAL_QUERY);

    await popup.keyboard.press("Escape");
    await expect(popup.locator("#query")).toHaveValue("");
    // Clearing must also drop the selection, or Enter would open a row that is no longer
    // on screen.
    await expect.poll(() => selectedIndex(popup)).toBe(-1);
    await expect(popup.locator("#query")).not.toHaveAttribute("aria-activedescendant", /.*/);
    // The first Escape only clears; it must not close.
    expect(await closeRequests(popup)).toBe(0);
    expect(await activeId(popup)).toBe("query");

    await popup.keyboard.press("Escape");
    await expect.poll(() => closeRequests(popup), { timeout: 10_000 }).toBe(1);

    await popup.close();
  });


  /* ---------------------------------------------------------------------- */
  /* the stale-response race                                               */
  /* ---------------------------------------------------------------------- */

  /**
   * Answers `fastQuery` immediately and holds EVERY other search for `delayMs`.
   *
   * That inverts the natural order: whatever the user typed on the way to the final query
   * is answered only after the final query's results are already on screen. Every response
   * is the REAL server's — the route handler forwards the request and only sits on the
   * reply — so this exercises the popup's staleness handling, not a stubbed payload.
   *
   * Returns live logs: `issued` in request order, `settled` in delivery order.
   */
  async function holdEarlierQueries(popup, fastQuery, delayMs) {
    const issued = [];
    const settled = [];
    await popup.route(/\/search\?/, async (route) => {
      const query = new URL(route.request().url()).searchParams.get("q");
      issued.push(query);

      let response = null;
      let body = null;
      try {
        response = await route.fetch();
        body = await response.text();
      } catch {
        // The page gave up on this request while we held it.
      }

      if (query !== fastQuery) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
      try {
        if (response) await route.fulfill({ response, body });
        else await route.abort();
      } catch {
        // Aborted from the page side; there is nothing left to deliver.
      }
      settled.push(query);
    });
    return { issued, settled };
  }

  /** True once every request the handler saw has been dealt with. */
  const allSettled = (log) => log.settled.length > 0 && log.settled.length === log.issued.length;

  test("overlapping as-you-type requests render the final query's results", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    // Fixture sanity FIRST: if the two queries returned the same list, everything below
    // would pass without testing anything.
    const prefixTitles = await serverTitles(PREFIX_QUERY);
    const finalTitles = await serverTitles(FINAL_QUERY);
    expect(
      finalTitles,
      "the prefix and final queries return identical result lists, so this test cannot " +
        "distinguish a stale response from a fresh one — the fixture corpus is wrong"
    ).not.toEqual(prefixTitles);
    expect(finalTitles.length).toBeGreaterThan(0);

    const popup = await openPopup(context, extensionId);
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    const log = await holdEarlierQueries(popup, FINAL_QUERY, 4000);
    const box = popup.locator("#query");

    // Type the first half, then wait — event-driven, not a sleep — until that partial query
    // has actually reached the server and is being held. Only then is the rest of the
    // typing guaranteed to overlap an in-flight request.
    await box.pressSequentially(PREFIX_QUERY, { delay: 20 });
    await expect
      .poll(() => log.issued.length, {
        timeout: 20_000,
        intervals: [100],
        message: "no partial query ever reached the server, so no overlap was created",
      })
      .toBeGreaterThan(0);
    expect(log.settled, "the partial query should still be in flight").toEqual([]);

    await box.pressSequentially(" grumbleshanks", { delay: 20 });
    await expect
      .poll(() => log.issued.includes(FINAL_QUERY), {
        timeout: 20_000,
        intervals: [100],
        message: "the final query never reached the server",
      })
      .toBe(true);

    // The final query is answered first even though it was asked last: the overlap is real
    // and the delivery order is genuinely inverted.
    await expect
      .poll(() => log.settled[0], { timeout: 20_000, intervals: [100] })
      .toBe(FINAL_QUERY);

    // Now let every held earlier response land.
    await expect
      .poll(() => allSettled(log), { timeout: 40_000, intervals: [250] })
      .toBe(true);
    expect(
      log.issued.length,
      "only one request was ever issued, so nothing overlapped"
    ).toBeGreaterThan(1);

    // Deliberate negative: give the late responses a window in which they could still
    // repaint the list before asserting that they did not.
    await popup.waitForTimeout(1000);

    expect(await box.inputValue()).toBe(FINAL_QUERY);
    expect(
      await renderedTitles(popup),
      "a slower earlier as-you-type request overwrote the results of the newer query"
    ).toEqual(finalTitles);

    await popup.unroute(/\/search\?/);
    await popup.close();
  });

  test("a stale response cannot overwrite a newer one even when abort does not fire", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    // popup.js documents two independent defences against the stale-response race: an
    // AbortController per request, and a sequence number checked before touching the DOM.
    // Abort alone would let the previous test pass while the sequence check was broken, so
    // neuter abort here — the stale response then really arrives and really gets handled,
    // and only the sequence check can keep it off the screen.
    const prefixTitles = await serverTitles(PREFIX_QUERY);
    const finalTitles = await serverTitles(FINAL_QUERY);
    expect(finalTitles).not.toEqual(prefixTitles);

    const popup = await openPopup(context, extensionId);
    await popup.evaluate(() => {
      AbortController.prototype.abort = function abort() {};
    });
    await expect.poll(() => activeId(popup), { timeout: 15_000 }).toBe("query");

    const log = await holdEarlierQueries(popup, FINAL_QUERY, 4000);
    const box = popup.locator("#query");

    // `fill` issues exactly one input event, hence exactly one request per step, so the
    // delivery order below can be asserted precisely instead of tolerantly.
    await box.fill(PREFIX_QUERY);
    await expect
      .poll(() => log.issued.length, { timeout: 20_000, intervals: [100] })
      .toBe(1);
    expect(log.issued[0]).toBe(PREFIX_QUERY);

    await box.fill(FINAL_QUERY);

    // Both must genuinely be DELIVERED to the page, newest first and the stale one last.
    await expect
      .poll(() => log.settled.join(" | "), { timeout: 40_000, intervals: [250] })
      .toBe(`${FINAL_QUERY} | ${PREFIX_QUERY}`);

    // Deliberate negative: the stale payload has been handed to the page; give its handler
    // room to do damage before asserting it did none.
    await popup.waitForTimeout(1000);

    expect(
      await renderedTitles(popup),
      "the sequence-number guard failed: a stale response repainted the list after a " +
        "newer query had already rendered"
    ).toEqual(finalTitles);
    await expect(popup.locator("#status")).toContainText(`${finalTitles.length} result`);

    await popup.unroute(/\/search\?/);
    await popup.close();
  });
});
