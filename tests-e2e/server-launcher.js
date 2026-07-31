"use strict";

/**
 * Boots the REAL Python server for the Playwright suite against a DISPOSABLE workspace.
 *
 * This is a wrapper rather than a bare `webServer.command` because Playwright starts the
 * webServer plugin *before* globalSetup runs, so seeding the workspace from globalSetup
 * would race the server. Doing it here makes the ordering unconditional.
 *
 * Never point this at ./workspace — that is the user's real data.
 */

const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const { REPO_ROOT, TEST_WORKSPACE, CONVSEARCH_EXE, PORT } = require("./paths");

if (!fs.existsSync(CONVSEARCH_EXE)) {
  console.error(`[e2e] missing Windows venv executable: ${CONVSEARCH_EXE}`);
  process.exit(1);
}

// Guard against ever being repointed at real data.
const resolved = path.resolve(TEST_WORKSPACE);
if (!path.basename(resolved).startsWith(".tmp-")) {
  console.error(`[e2e] refusing to use a workspace that is not disposable: ${resolved}`);
  process.exit(1);
}

fs.rmSync(resolved, { recursive: true, force: true });
fs.mkdirSync(resolved, { recursive: true });

const init = spawnSync(CONVSEARCH_EXE, ["init", resolved, "--force"], {
  cwd: REPO_ROOT,
  stdio: "inherit",
});
if (init.status !== 0) {
  console.error(`[e2e] convsearch init failed (exit ${init.status})`);
  process.exit(init.status || 1);
}

const server = spawn(
  CONVSEARCH_EXE,
  ["serve", "--workspace", resolved, "--port", String(PORT), "--test-embeddings"],
  { cwd: REPO_ROOT, stdio: "inherit" }
);

server.on("exit", (code) => process.exit(code === null ? 1 : code));
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.kill(signal));
}
