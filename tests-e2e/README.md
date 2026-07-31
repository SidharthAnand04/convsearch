# End-to-end suite

Playwright driving a real Chromium, with the real unpacked extension from `../extension/`
loaded, against a **real Python server**. There is no mock layer anywhere: the server is the
actual `convsearch serve` process writing to an actual SQLite workspace. These tests are not
run in CI (`.github/workflows/ci.yml` covers ruff, mypy, pytest and the synthetic eval only);
they are a local pre-release gate.

## Known portability limit — read this first

`paths.js:21` hardcodes `.venv/Scripts/convsearch.exe`, and `server-launcher.js:19-22` does a
hard `process.exit(1)` when that file is absent. **The suite only runs on Windows as written.**
On macOS or Linux the executable lives at `.venv/bin/convsearch`, and there is no environment
variable or config switch that redirects it today — fixing this needs a code change in
`paths.js`.

## Running it

```bash
npm install
npm run setup          # playwright install chromium   (package.json:12)
npm test               # headed by default, see below
npm run test:headed    # explicit --headed
npm run test:ui        # Playwright UI mode
```

Commands run from the repo root; `playwright.config.js` lives there and points `testDir` here.

Tests run **headed by default**. MV3 extensions historically do not load in old headless
Chromium, so headed is the mode that reliably works; set `E2E_HEADLESS=1` to opt into the new
headless shell (`helpers.js:24`).

## Isolation guarantees

The suite must never touch a developer's real data, and several independent guards enforce it.

- **Port 8791**, deliberately not the default 8756 (`paths.js:10-12`), overridable with
  `CONVSEARCH_TEST_PORT`.
- **Workspace `tests-e2e/.tmp-workspace`**, wiped before the run and again after. Both the
  launcher (`server-launcher.js:25-29`) and the teardown (`global-teardown.js:11`) refuse to
  touch a path whose basename does not start with `.tmp-`, so a mis-edited path fails loudly
  instead of deleting `./workspace`.
- **`reuseExistingServer: false`** in `playwright.config.js`, so the run never adopts a server
  a developer already has running — that server could be on real data.
- The server starts with **`--test-embeddings`** (`server-launcher.js:45`), giving deterministic
  local vectors instead of downloading a model.

`server-launcher.js` exists as a wrapper rather than a bare `webServer.command` because
Playwright starts the webServer plugin *before* `globalSetup`, so seeding the workspace from
globalSetup would race the server. Doing init inside the launcher makes the ordering
unconditional.

## The chatgpt.com trick

The suite never reaches the real chatgpt.com, yet the content script still runs against the
real origin. Playwright intercepts every `https://chatgpt.com/**` request in the context and
fulfils it locally (`helpers.js:3-10, :39-40`): the conversation path `/c/<id>` gets a fixture
from `fixtures/`, and everything else — favicon, SPA polling — gets an empty 204 so nothing
escapes to the internet.

Because the URL and origin are genuinely `https://chatgpt.com`, the manifest's content-script
match fires exactly as it does in production. No login, no network, no real account, and no
special-cased test origin in the manifest.

## Why it is slow

`workers: 1` and `fullyParallel: false`, because every test shares one server and one SQLite
workspace — parallel runs would make health and idempotency counts non-deterministic.

Two large timeouts, both justified inline in `playwright.config.js` with measured numbers:

- **Test timeout 240s.** Each test launches its own persistent Chromium with the unpacked
  extension; that launch plus context teardown alone measured 1.5-3 minutes. This is
  environment headroom — no assertion depends on it.
- **webServer timeout 600s.** Measured cold start was ~30s for `convsearch init` plus ~230s for
  the serve process to import its dependencies and bind the port. The import cost is in product
  code, not here.

## Files

- `paths.js` — single source of truth for the port and every path. The launcher, the config and
  the specs all read from it so they cannot disagree.
- `server-launcher.js` — wipes the workspace, runs `convsearch init --force`, spawns
  `convsearch serve`.
- `global-teardown.js` — removes the disposable workspace.
- `helpers.js` — the Playwright fixture: extension-loaded persistent context, chatgpt.com
  routing, unique conversation ids per test.
- `fixtures/` — `conversation.html`, `topic.html`, `malformed.html`.
- Seven specs: `autoindex`, `capture`, `extraction`, `keyboard`, `popup`, `scenario`,
  `sidepanel`.

## Not part of this suite

`scripts/live.js` (`npm run live`) opens the **real** chatgpt.com in Chromium with the
extension loaded, using a persisted profile in `.chrome-profile/` so you log in once. There is
no interception and there are no fixtures. It is a manual tool for exercising live capture, not
a test — nothing asserts, and it writes to whatever server is on `CONVSEARCH_SERVER`
(default `http://127.0.0.1:8756`, i.e. your real workspace).
