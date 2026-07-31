# convsearch Chrome extension

The browser half of convsearch. It captures the conversation you are looking at on
chatgpt.com, hands it to the local Python server, and gives you a side panel to search and
reason over everything captured so far.

There is **no build step**. This folder is the extension: load it unpacked from
`chrome://extensions` with Developer mode on. Manifest V3, version `0.3.0`
(`manifest.json:2-4`). No bundler, no npm dependencies, no transpilation — what you edit is
what Chrome runs, so a reload of the extension is the whole edit-test loop.

## File map

| File | Role |
| --- | --- |
| `background.js` | Service worker. The **only** file that speaks HTTP. Owns the offline capture queue, the retry schedule, and native-host auto-start. |
| `capture.js` | Pure DOM extraction for chatgpt.com. Deliberately free of `chrome.*` APIs so the test suite can load it into a plain page and call `convsearchExtract(document, url)` directly. |
| `content.js` | Decides *when* to extract (SPA navigation, mutation settle, streaming) and forwards the result to the worker. Never modifies the host page. |
| `sidepanel.{html,js,css}` | The eleven-view panel. `sidepanel.js` is ~3465 lines of vanilla JS. |
| `popup.{html,js,css}` | Quick-search fallback behind the toolbar action (Alt+Shift+K). |
| `options.{html,js}` | Server URL and capture toggle. |
| `theme.css`, `icons/` | Shared dark/light tokens; store icons at 16/32/48/128. |

Both content scripts are injected together at `document_idle` on `https://chatgpt.com/*` and
`https://chat.openai.com/*` (`manifest.json:24-30`); `capture.js` loads first and exposes its
entry point on `globalThis`, because content scripts are not ES modules.

## Three load-bearing rules

Almost every mistake in this codebase is a violation of one of these.

### 1. Only `background.js` fetches

The panel, the popup, and the content scripts never call `fetch`. They send a
`chrome.runtime` message; the worker dispatches it through the `HANDLERS` table
(`background.js:954-984`) and replies with a uniform envelope — `{ ok: true, data }` or
`{ ok: false, error, status }`. Adding a server call therefore means two edits: a `handle*`
function that wraps `apiGet`/`apiPost`, and an entry in `HANDLERS`. Nothing else.

This exists so there is a single choke point for the next rule.

### 2. Loopback allow-list

`isLoopback()` (`background.js:184-194`) accepts only `127.0.0.1` or `localhost` over
`http:`/`https:`, and it is checked inside **both** `apiGet` (`:206`) and `apiPost` (`:243`)
before the URL is built. Because the worker is the only component that fetches, a tampered
stored `serverUrl` still cannot reach anything but your own machine. The options page mirrors
the same check on save (`options.js:38-39`) so bad input is rejected early, but the worker's
check is the one that actually enforces it — never remove it in favour of the UI check.

### 3. No module-level durable state

MV3 workers are torn down at roughly 30s idle (`background.js:9-13`). Anything that must
survive that goes to `chrome.storage.local` and is read back on demand: the capture queue, its
retry schedule, the last successful capture time, the server URL and capture toggle
(`STORAGE_DEFAULTS`, `:38-44`). Module-level variables are legitimate only as a within-wake-up
cache. The ensure-server debounce uses `chrome.storage.session` instead (`:291`), since it
should reset when the browser restarts but must survive worker teardown.

## Talking to the server

Default `http://127.0.0.1:8756` (`background.js:15`), overridable on the options page.
Per-call timeouts are chosen by what the call actually does, not by a single global:

| Call | Timeout | Why |
| --- | --- | --- |
| health/status | 2s (`:18`) | A dead port must not hang the popup. |
| query (search/ask/memories) | 45s (`:20`) | Ask can invoke a local LLM. |
| capture POST | 10s (`:22`) | A small local write; should be fast even under load. |
| reindex | 300s (`:24`) | Rebuilding embeddings legitimately takes seconds to minutes. |
| learn run | 120s (`:26`) | May walk many events through the local LLM. |

Bodies are capped at 8 MB client-side (`:29`) to match the server's `MAX_CAPTURE_BYTES`
(`src/convsearch/capture/ingest.py:30`), so an oversized capture fails locally instead of
burning a round trip on a 413.

When the server is down, captures go to a bounded offline queue: at most 200 items (`:31`,
oldest dropped first), retried on the ladder `[5s, 15s, 45s, 2m, 5m, 10m]` (`:34`, last value
repeats), and abandoned after 12 attempts (`:36`, roughly an hour).

## Auto-start via native messaging

The MV3 sandbox cannot launch a process, so when a health check finds the server down the
worker calls `chrome.runtime.connectNative("com.convsearch.host")` (`background.js:284, :345`)
and asks a registered native host to run `convsearch serve`.

A missing or unregistered host is **not an error**. `connectNative` fails, the worker resolves
`{ ok: false, reason: "native_host_unavailable" }`, and the extension degrades silently to its
normal offline behaviour — the panel already shows setup guidance (`:276-283`). Nothing here
may throw or log for a missing host; users who never ran the installer are the common case.

Registration is one-time and documented in `../scripts/native-host/README.md`.

## The eleven views

From the `data-view` attributes on the rail (`sidepanel.html:15-75`): **home**, **search**
(labelled "Ask"), **plan**, **tasks**, **projects**, **timeline**, **memories**, **review**,
**captures**, **privacy**, **status**.

The command bar above them parses input locally — no LLM, no extra endpoint. Recognised verbs
(`sidepanel.js:3232-3250`): `search`, `ask`, `find`, `show`, `project`, `open`, `rebuild`,
`diagnostics`, `status`, `help`, `tasks`, `timeline`, `captures`, `capture`, `review`,
`privacy`. A known verb used wrongly gets an inline command list rather than a silent no-op;
unknown input falls through to search.

## Security posture

Server-supplied text is rendered with `textContent` and **never** `innerHTML`. Every node in
the panel is built through the `el` helper, which only sets text (`sidepanel.js:11, :64`); the
popup follows the same rule (`popup.js:107`). This is an XSS control against untrusted
conversation content, and it is also what keeps the extension inside MV3's prohibition on
executing remotely-supplied code — a review failure, not just a bug.

Host permissions are exactly three (`manifest.json:7-11`): `http://127.0.0.1/*`,
`http://localhost/*`, `https://chatgpt.com/*`.

Worth stating deliberately: `chat.openai.com` appears as a content-script match
(`manifest.json:26`) and as a server-side CORS origin
(`src/convsearch/server/app.py:83-88`), but **not** in `host_permissions`. The content script
runs there and can extract, and the server would accept the origin, but the extension holds no
host permission for it.

## Testing

`../tests-e2e/` drives this exact folder with real Chromium and a real Python server. See
`../tests-e2e/README.md` — note the Windows-only limitation documented there.

## Packaging

```
./scripts/package-extension.sh      # or scripts/package-extension.ps1
```

Produces `dist/convsearch-extension-v0.3.0.zip` with `manifest.json` at the ZIP root, which is
what the Chrome Web Store requires — the script archives the *contents* of this folder, not the
folder itself. Bump `manifest.json` version for every submission.

Two numbering gotchas:

- The extension version (`0.3.0`) is a separate number space from the Python package version
  (`0.1.0` in `pyproject.toml`). They are not meant to track each other.
- Chrome accepts only dot-separated integers. No `-rc1`, no `-beta`, no `+build`.
