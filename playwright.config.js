"use strict";

const { defineConfig } = require("@playwright/test");
const { PORT, SERVER_URL } = require("./tests-e2e/paths");

module.exports = defineConfig({
  testDir: "./tests-e2e",
  // The suite shares one server and one SQLite workspace; parallelism would make the
  // health/idempotency counts non-deterministic.
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  // Every test launches its own persistent Chromium with the unpacked extension loaded.
  // Measured on this machine while other work was in flight, that launch plus fixture
  // teardown alone runs 1.5–3 minutes, and the previous 90s budget was being exhausted in
  // `Tearing down "context"` rather than by anything a test asserted. This is environment
  // headroom only — no assertion depends on it.
  timeout: 240_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  globalTeardown: require.resolve("./tests-e2e/global-teardown.js"),

  use: {
    trace: "retain-on-failure",
    video: "off",
  },

  // Boots the REAL Python server against a disposable workspace (see server-launcher.js).
  webServer: {
    command: "node tests-e2e/server-launcher.js",
    url: `${SERVER_URL}/health`,
    // Never adopt a server someone else started — it could be on the real workspace.
    reuseExistingServer: false,
    // Measured cold start on this machine: `convsearch init` ~30s plus ~230s for the
    // serve process to import its dependencies and bind the port — 260s total, which blew
    // straight through the previous 180s limit and failed the whole suite before a single
    // test ran. The import cost lives in product code, not here; give it real headroom.
    timeout: 600_000,
    stdout: "pipe",
    stderr: "pipe",
    env: { CONVSEARCH_TEST_PORT: String(PORT) },
  },
});
