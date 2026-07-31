"use strict";

/**
 * The full realistic arc, end to end through the real extension and real server:
 *
 *   1. browse THREE conversations on chatgpt.com  → all captured
 *   2. let the index build                        → all three searchable
 *   3. exercise the advanced query syntax         → phrases, exclusion, identifiers
 *   4. browse TWO MORE conversations              → auto-indexed incrementally
 *   5. re-check everything                        → new ones found, old ones intact,
 *                                                   advanced syntax still discriminates
 *
 * Step 5 is the point. Appending to a FAISS index is where this design can silently break:
 * if the vector map and the index drift apart, the earlier conversations either vanish or
 * start returning the wrong passages, and only a query that spans both batches shows it.
 *
 * Every conversation's prose is deliberately distinct. `passages.content_hash` is GLOBALLY
 * UNIQUE with INSERT OR IGNORE, so shared boilerplate would collapse into a single passage
 * row and make counts wrong for reasons unrelated to the code under test.
 */

const { test, expect, fixtureHtml, serveChatGpt, newConversationId, health, search } =
  require("./helpers");
const { SERVER_URL } = require("./paths");

/* -------------------------------------------------------------------------- */
/* the corpus                                                                 */
/* -------------------------------------------------------------------------- */

function topic(title, ask, reply) {
  return { id: newConversationId(), title, ask, reply };
}

/**
 * First batch of three.
 *
 * These subjects must not overlap the corpus in any OTHER spec file. The whole suite shares
 * one workspace, so vocabulary shared with autoindex.spec.js would make two different
 * conversations match the same phrase and quietly invalidate the precision assertions below.
 */
const TURF = topic(
  "Icelandic turf house construction",
  "How were traditional Icelandic turf houses kept warm through the winter?",
  "Thick sod walls were stacked in a herringbone pattern over driftwood turf lintels, " +
    "trapping still air so interior temperatures stayed survivable without much fuel."
);
const KILN = topic(
  "Pottery kiln cooling schedule",
  "How slowly should I cool a stoneware kiln to avoid cracking the glaze?",
  "Ease through quartz inversion near five hundred and seventy degrees celsius, because " +
    "a sudden contraction there is what crazes the glaze surface and splits thick walls."
);
const RUST = topic(
  "Shared state across tasks",
  "What is the idiomatic way to share mutable state between asynchronous Rust tasks?",
  "Wrap the value in Arc<Mutex<T>> so ownership is shared and access serialised. A " +
    "deadlock appears when two tasks acquire their locks in opposite order."
);

/** Second batch of two, browsed after the first index build. */
const POSTGRES = topic(
  "Finding slow statements",
  "Which extension should I enable to find the slowest statements on my database?",
  "Enable pg_stat_statements and sort by total execution time. Note that a deadlock " +
    "under Postgres aborts whichever transaction started more recently."
);
const BICYCLE = topic(
  "Gearing for steep climbs",
  "What gearing should I fit for sustained twenty percent gradients on a loaded tourer?",
  "Fit a sub-compact crankset and a wide cassette so the lowest ratio drops under one " +
    "to one; a long-cage derailleur is required to take up the extra chain wrap."
);

const FIRST_BATCH = [TURF, KILN, RUST];
const SECOND_BATCH = [POSTGRES, BICYCLE];

function htmlFor(t) {
  return fixtureHtml("topic.html", {
    __CONV_ID__: t.id,
    __TITLE__: t.title,
    __ASK__: t.ask,
    __REPLY__: t.reply,
  });
}

/* -------------------------------------------------------------------------- */
/* helpers                                                                    */
/* -------------------------------------------------------------------------- */

async function browse(context, t) {
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
  await serveChatGpt(context, { conversationId: t.id, html: htmlFor(t) });

  const page = await context.newPage();
  await page.goto(`https://chatgpt.com/c/${t.id}`);
  await expect
    .poll(() => captured.length, { timeout: 45_000, message: `no capture for "${t.title}"` })
    .toBeGreaterThan(0);

  await context.unroute(`${SERVER_URL}/capture`);
  await page.close();
  return captured[0];
}

/** Ids returned for a query. */
async function idsFor(query, params = {}) {
  const payload = await search(query, { limit: "10", ...params });
  return (payload.results || []).map((r) => r.source_conversation_id);
}

/**
 * Which retrieval channels matched a given conversation.
 *
 * This matters for the advanced syntax. Search is HYBRID: quoting and `-exclusion` are
 * compiled into FTS5 and therefore only constrain the LEXICAL channel. The semantic channel
 * scores by embedding similarity and knows nothing about negation, so an excluded
 * conversation can still surface through rank fusion. The honest assertion is therefore
 * "the lexical channel stopped matching", not "the result disappeared".
 */
async function channelsFor(query, conversationId, params = {}) {
  const payload = await search(query, { limit: "10", passages: "5", ...params });
  const hit = (payload.results || []).find((r) => r.source_conversation_id === conversationId);
  if (!hit) return null;
  return new Set(hit.passages.flatMap((p) => p.channels || []));
}

async function waitUntilSearchable(t, timeout = 120_000) {
  await expect
    .poll(async () => (await idsFor(t.ask.split(" ").slice(0, 6).join(" "))).includes(t.id), {
      timeout,
      intervals: [1000],
      message: `"${t.title}" never became searchable — auto-indexing did not run`,
    })
    .toBe(true);
}

/** Wait for the server to report no indexing pass running or queued. */
async function waitUntilIdle(timeout = 120_000) {
  await expect
    .poll(async () => (await health()).indexing === false, { timeout, intervals: [1000] })
    .toBe(true);
}

/* -------------------------------------------------------------------------- */

/**
 * Counts are RELATIVE to a baseline taken when this file starts.
 *
 * Every spec file in the suite shares one server and one workspace, so by the time this
 * file runs, other specs have already stored conversations. Asserting absolute totals would
 * pass when running this file alone and fail in the full suite — which is exactly what it
 * did before this baseline existed.
 */
let baseline = null;

test.describe.serial("full scenario", () => {
  test("step 1-2: three browsed conversations all become searchable", async ({
    context,
    configuredExtension,
  }) => {
    baseline = await health();

    for (const t of FIRST_BATCH) {
      const result = await browse(context, t);
      expect(result.conversations_written, `capture rejected "${t.title}"`).toBe(1);
    }

    await waitUntilIdle();
    for (const t of FIRST_BATCH) {
      await waitUntilSearchable(t);
    }

    const after = await health();
    expect(after.conversations - baseline.conversations).toBe(3);
    expect(after.captured_conversations - baseline.captured_conversations).toBe(3);
    expect(after.stale_index).toBe(false);
  });

  test("step 3: advanced query syntax discriminates across the corpus", async ({
    configuredExtension,
  }) => {
    // A quoted phrase matches lexically ONLY where those words are adjacent. Asserting the
    // result list equals [OTTERS] would be wrong: the semantic channel also surfaces loosely
    // related conversations stored by other spec files in this shared workspace. The precise
    // claim is about which conversation the phrase matched *lexically*.
    expect(await channelsFor('"turf lintels"', TURF.id)).toContain("lexical");
    const phraseHits = await search('"turf lintels"', { limit: "10", passages: "5" });
    const lexicalPhraseMatches = (phraseHits.results || [])
      .filter((r) => r.passages.some((p) => (p.channels || []).includes("lexical")))
      .map((r) => r.source_conversation_id);
    expect(lexicalPhraseMatches).toEqual([TURF.id]);

    // A technical identifier must survive tokenisation rather than being split apart,
    // and must match lexically — not merely land nearby in embedding space.
    expect(await idsFor("Arc<Mutex<T>>")).toContain(RUST.id);
    expect(await channelsFor("Arc<Mutex<T>>", RUST.id)).toContain("lexical");

    // Baseline: "deadlock" matches the Rust conversation lexically.
    expect(await idsFor("deadlock")).toContain(RUST.id);
    expect(await channelsFor("deadlock", RUST.id)).toContain("lexical");

    // Negation suppresses the LEXICAL match. The conversation may still be reachable via
    // the semantic channel, which is inherent to hybrid retrieval, so assert on the
    // channel rather than on absence from the result list.
    const excluded = await channelsFor("deadlock -opposite", RUST.id);
    if (excluded !== null) {
      expect(
        excluded,
        "-opposite should have removed the lexical match for the Rust conversation"
      ).not.toContain("lexical");
    }

    // A term in no conversation yields no lexical match anywhere.
    for (const id of FIRST_BATCH.map((t) => t.id)) {
      const channels = await channelsFor("derailleur", id);
      if (channels !== null) expect(channels).not.toContain("lexical");
    }
  });

  test("step 4-5: two more conversations auto-index without disturbing the first three", async ({
    context,
    configuredExtension,
  }) => {
    for (const t of SECOND_BATCH) {
      const result = await browse(context, t);
      expect(result.conversations_written, `capture rejected "${t.title}"`).toBe(1);
    }

    await waitUntilIdle();
    for (const t of SECOND_BATCH) {
      await waitUntilSearchable(t);
    }

    const after = await health();
    expect(after.conversations - baseline.conversations).toBe(5);
    expect(after.captured_conversations - baseline.captured_conversations).toBe(5);
    expect(after.stale_index).toBe(false);

    // The original three must still be retrievable. This is the assertion that catches an
    // append that misaligned or clobbered the earlier vectors.
    for (const t of FIRST_BATCH) {
      expect(
        await idsFor(t.reply.split(" ").slice(0, 8).join(" ")),
        `"${t.title}" was lost when the second batch was appended`
      ).toContain(t.id);
    }

    // Advanced syntax must now discriminate across BOTH batches. "deadlock" appears in the
    // old Rust conversation and the new Postgres one; excluding Postgres must leave Rust.
    const bothDeadlocks = await idsFor("deadlock");
    expect(bothDeadlocks).toContain(RUST.id);
    expect(bothDeadlocks).toContain(POSTGRES.id);

    // Excluding a term unique to the Postgres conversation must strip ITS lexical match
    // while leaving the Rust one intact — proving negation still discriminates after an
    // incremental append, across conversations indexed in different passes.
    expect(await channelsFor("deadlock", POSTGRES.id)).toContain("lexical");
    const withoutPostgres = await channelsFor("deadlock -pg_stat_statements", POSTGRES.id);
    if (withoutPostgres !== null) {
      expect(withoutPostgres).not.toContain("lexical");
    }
    expect(await channelsFor("deadlock -pg_stat_statements", RUST.id)).toContain("lexical");

    // Identifiers from the newly appended batch resolve too.
    expect(await idsFor("pg_stat_statements")).toContain(POSTGRES.id);
    expect(await idsFor("derailleur")).toContain(BICYCLE.id);

    // And the deep links still point at the real site.
    const payload = await search("pg_stat_statements", { limit: "5" });
    const hit = (payload.results || []).find((r) => r.source_conversation_id === POSTGRES.id);
    expect(hit.url).toBe(`https://chatgpt.com/c/${POSTGRES.id}`);
  });

  test("step 6: the popup searches the whole corpus and links out", async ({
    context,
    extensionId,
    configuredExtension,
  }) => {
    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${extensionId}/popup.html`);

    // Nothing to nag about: everything captured has been indexed automatically.
    await expect
      .poll(() => popup.locator("#capture-alert").isVisible(), {
        timeout: 20_000,
        message: "the stale-index alert was visible even though the index is current",
      })
      .toBe(false);
    const total = baseline.conversations + 5;
    await expect(popup.locator("#capture-summary")).toContainText(`${total} conversations`);

    await popup.locator("#query").fill("pg_stat_statements");
    await popup.locator("#submit").click();

    const first = popup.locator(".result").first();
    await expect(first).toBeVisible({ timeout: 30_000 });
    await expect(first.locator(".result-title")).toHaveAttribute(
      "href",
      `https://chatgpt.com/c/${POSTGRES.id}`
    );
  });
});
