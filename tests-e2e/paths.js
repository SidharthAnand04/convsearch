"use strict";

/** Shared paths and the test port. Kept in one place so the launcher, the Playwright
 *  config and the specs can never disagree about which server/workspace is in play. */

const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..");

/** Deliberately NOT the default 8756: the suite must never talk to a server a developer
 *  happens to be running against their real workspace. */
const PORT = Number(process.env.CONVSEARCH_TEST_PORT || 8791);

module.exports = {
  REPO_ROOT,
  PORT,
  SERVER_URL: `http://127.0.0.1:${PORT}`,
  EXTENSION_DIR: path.join(REPO_ROOT, "extension"),
  FIXTURES_DIR: path.join(__dirname, "fixtures"),
  TEST_WORKSPACE: path.join(REPO_ROOT, "tests-e2e", ".tmp-workspace"),
  CONVSEARCH_EXE: path.join(REPO_ROOT, ".venv", "Scripts", "convsearch.exe"),
};
