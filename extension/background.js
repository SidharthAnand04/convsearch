"use strict";

/**
 * convsearch background service worker.
 *
 * The only component that speaks HTTP to the local Python server. Content scripts and the
 * popup talk to it exclusively through chrome.runtime messages.
 *
 * MV3 service workers are torn down aggressively (~30s idle), so NOTHING here may rely on
 * module-level state surviving. Every piece of durable state (the capture queue, its retry
 * schedule, the last successful capture time) lives in chrome.storage.local and is read
 * back on demand. Module-level variables are treated purely as a within-wake-up cache.
 */

const DEFAULT_SERVER = "http://127.0.0.1:8756";

/** Popup/status calls must never hang on a dead port. */
const HEALTH_TIMEOUT_MS = 2000;
/** Side-panel read queries (search/ask/memories/…). Ask can invoke a local LLM, so allow more. */
const QUERY_TIMEOUT_MS = 45000;
/** A capture POST is a small local write; it should be fast even on a busy server. */
const CAPTURE_TIMEOUT_MS = 10000;
/** Reindex rebuilds embeddings and legitimately takes seconds. Generous ceiling only. */
const REINDEX_TIMEOUT_MS = 300000;
/** A learn run may invoke the local LLM over many events; give it a long ceiling. */
const LEARN_TIMEOUT_MS = 120000;

/** Server rejects bodies over 8 MB with 413; don't waste a round trip. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Bound the queue so a long offline stretch can't fill the storage quota. */
const MAX_QUEUE_ITEMS = 200;

/** Backoff ladder in ms, indexed by consecutive-failure count; last value repeats. */
const BACKOFF_MS = [5000, 15000, 45000, 120000, 300000, 600000];
/** Give up on a single item after this many attempts (~1h of retries). */
const MAX_ATTEMPTS = 12;

const STORAGE_DEFAULTS = {
  serverUrl: DEFAULT_SERVER,
  captureEnabled: true,
  lastCaptureAt: null,
  captureQueue: [],
  queueState: { nextAttemptAt: 0, failures: 0, offline: false, lastError: null },
};

/* -------------------------------------------------------------------------- */
/* storage helpers                                                            */
/* -------------------------------------------------------------------------- */

function storageGet(defaults) {
  return new Promise((resolve) => {
    chrome.storage.local.get(defaults, (stored) => {
      if (chrome.runtime.lastError) resolve(defaults);
      else resolve(stored);
    });
  });
}

function storageSet(values) {
  return new Promise((resolve) => {
    chrome.storage.local.set(values, () => {
      // Reading lastError marks it handled; a failed write is not worth throwing over.
      void chrome.runtime.lastError;
      resolve();
    });
  });
}

async function getServerUrl() {
  const stored = await storageGet({ serverUrl: DEFAULT_SERVER });
  const raw = typeof stored.serverUrl === "string" && stored.serverUrl.trim()
    ? stored.serverUrl.trim()
    : DEFAULT_SERVER;
  return raw.replace(/\/+$/, "");
}

/**
 * Serialises all read-modify-write cycles on the queue. Multiple content scripts (one per
 * ChatGPT tab) can message us concurrently; without this, two captures read the same queue
 * and one overwrites the other.
 */
let lock = Promise.resolve();

function withQueueLock(fn) {
  const run = lock.then(() => fn());
  lock = run.then(
    () => undefined,
    () => undefined
  );
  return run;
}

async function readQueue() {
  const stored = await storageGet({
    captureQueue: [],
    queueState: STORAGE_DEFAULTS.queueState,
  });
  const queue = Array.isArray(stored.captureQueue) ? stored.captureQueue : [];
  const state = stored.queueState && typeof stored.queueState === "object"
    ? stored.queueState
    : { ...STORAGE_DEFAULTS.queueState };
  return { queue, state };
}

async function writeQueue(queue, state) {
  await storageSet({ captureQueue: queue, queueState: state });
  await updateBadge(queue.length, state.offline);
}

/* -------------------------------------------------------------------------- */
/* badge                                                                      */
/* -------------------------------------------------------------------------- */

const BADGE_OFFLINE = "#b3261e";
const BADGE_PENDING = "#8a6100";

/**
 * Quiet when everything is fine (no badge at all). Pending work shows a count; an amber
 * count means "just catching up", red means "the server isn't answering". With nothing
 * queued but a known-dead server we show a bare dot so the state is still visible without
 * being noisy.
 */
async function updateBadge(pending, offline) {
  if (!chrome.action || !chrome.action.setBadgeText) return;
  let text = "";
  let color = BADGE_PENDING;
  if (pending > 0) {
    text = pending > 99 ? "99+" : String(pending);
    color = offline ? BADGE_OFFLINE : BADGE_PENDING;
  } else if (offline) {
    text = "·";
    color = BADGE_OFFLINE;
  }
  try {
    await chrome.action.setBadgeBackgroundColor({ color });
    await chrome.action.setBadgeText({ text });
    await chrome.action.setTitle({
      title: badgeTitle(pending, offline),
    });
  } catch {
    // The action API can be unavailable during shutdown; never let this throw.
  }
}

function badgeTitle(pending, offline) {
  if (pending > 0 && offline) {
    return `convsearch — server offline, ${pending} capture${pending === 1 ? "" : "s"} queued`;
  }
  if (pending > 0) {
    return `convsearch — ${pending} capture${pending === 1 ? "" : "s"} pending`;
  }
  if (offline) return "convsearch — server offline";
  return "convsearch";
}

/* -------------------------------------------------------------------------- */
/* http                                                                       */
/* -------------------------------------------------------------------------- */

/** fetch with a hard deadline. AbortController is the only way to bound a dead port. */
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * The loopback allow-list. The service worker is the ONLY component that fetches, so this is
 * the single choke point: even if a stored serverUrl were somehow tampered with, a query can
 * only ever reach localhost. Mirrors the check the options page enforces on save.
 */
function isLoopback(urlString) {
  try {
    const url = new URL(urlString);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      (url.hostname === "127.0.0.1" || url.hostname === "localhost")
    );
  } catch {
    return false;
  }
}

/**
 * Generic GET proxy used by every side-panel view. Returns the uniform envelope the panel
 * expects: `{ ok: true, data }` on success, `{ ok: false, error, status }` otherwise. All the
 * network/JSON/error handling lives here so the panel never touches fetch.
 *
 * `params` values that are null/undefined/"" are dropped, so callers can pass a flat options
 * object without pre-filtering.
 */
async function apiGet(path, params = {}, timeoutMs = QUERY_TIMEOUT_MS) {
  const serverUrl = await getServerUrl();
  if (!isLoopback(serverUrl)) {
    return { ok: false, error: "server URL must be a loopback address", status: 0 };
  }

  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined) continue;
    const str = String(value);
    if (str === "") continue;
    search.set(key, str);
  }
  const query = search.toString();
  const url = `${serverUrl}${path}${query ? `?${query}` : ""}`;

  let response;
  try {
    response = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
  } catch (error) {
    return { ok: false, error: describeError(error), status: 0 };
  }

  const payload = await readJson(response);
  if (!response.ok) {
    const message =
      (payload && (payload.error || payload.detail)) || `HTTP ${response.status}`;
    return { ok: false, error: String(message), status: response.status };
  }
  return { ok: true, data: payload || {} };
}

/**
 * Generic POST proxy, the write-side twin of `apiGet`. Same loopback allow-list, same uniform
 * `{ ok, data } | { ok:false, error, status }` envelope, same single choke point on fetch. Used
 * for interaction-logging writes (feedback) that the side panel fires and forgets.
 */
async function apiPost(path, body = {}, timeoutMs = QUERY_TIMEOUT_MS) {
  const serverUrl = await getServerUrl();
  if (!isLoopback(serverUrl)) {
    return { ok: false, error: "server URL must be a loopback address", status: 0 };
  }

  const url = `${serverUrl}${path}`;
  let response;
  try {
    response = await fetchWithTimeout(
      url,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      },
      timeoutMs
    );
  } catch (error) {
    return { ok: false, error: describeError(error), status: 0 };
  }

  const payload = await readJson(response);
  if (!response.ok) {
    const message =
      (payload && (payload.error || payload.detail)) || `HTTP ${response.status}`;
    return { ok: false, error: String(message), status: response.status };
  }
  return { ok: true, data: payload || {} };
}

/* -------------------------------------------------------------------------- */
/* native messaging — auto-start the local server                             */
/* -------------------------------------------------------------------------- */

/**
 * The MV3 sandbox cannot launch a process, so when the local server is down we ask a
 * registered Native Messaging host (scripts/native_host.py, reached through a launcher)
 * to start `convsearch serve` for us. The one-time registration is done by
 * scripts/install-native-host.{ps1,sh}; if it was never run, connectNative fails and we
 * degrade silently to the existing offline behaviour (the panel already shows setup
 * guidance). Nothing here ever throws or spams the console for a missing host.
 */
const NATIVE_HOST_NAME = "com.convsearch.host";

/** Don't pester the host: at most one ensure attempt per this window. */
const ENSURE_DEBOUNCE_MS = 30000;
/** The host polls /health for ~25s after a spawn; give the port a little more. */
const NATIVE_HOST_TIMEOUT_MS = 30000;
/** storage.session key so the debounce survives service-worker teardown. */
const ENSURE_TS_KEY = "convsearchLastEnsureAt";

/** Within-wake-up cache of the last attempt; storage.session is the durable copy. */
let lastEnsureAttemptAt = 0;

async function getLastEnsureAttempt() {
  if (chrome.storage && chrome.storage.session) {
    try {
      const stored = await new Promise((resolve) => {
        chrome.storage.session.get({ [ENSURE_TS_KEY]: 0 }, (value) => {
          void chrome.runtime.lastError;
          resolve(value);
        });
      });
      return Number(stored[ENSURE_TS_KEY] || 0) || lastEnsureAttemptAt;
    } catch {
      return lastEnsureAttemptAt;
    }
  }
  return lastEnsureAttemptAt;
}

async function setLastEnsureAttempt(ts) {
  lastEnsureAttemptAt = ts;
  if (chrome.storage && chrome.storage.session) {
    try {
      await new Promise((resolve) => {
        chrome.storage.session.set({ [ENSURE_TS_KEY]: ts }, () => {
          void chrome.runtime.lastError;
          resolve();
        });
      });
    } catch {
      /* session storage unavailable — the in-memory copy still debounces this wake-up. */
    }
  }
}

/**
 * Connect to the native host and ask it to ensure the server is up. Resolves with the
 * host's response object on success, or `{ ok:false, reason:"native_host_unavailable" }`
 * when the host is missing/unregistered or the connection drops. Never rejects.
 */
function ensureServerViaNativeHost() {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      resolve(result);
    };

    let port;
    try {
      port = chrome.runtime.connectNative(NATIVE_HOST_NAME);
    } catch {
      // Reading lastError (if any) keeps it from surfacing as an unchecked error.
      void chrome.runtime.lastError;
      finish({ ok: false, reason: "native_host_unavailable" });
      return;
    }
    if (!port) {
      finish({ ok: false, reason: "native_host_unavailable" });
      return;
    }

    const timer = setTimeout(() => {
      try {
        port.disconnect();
      } catch {
        /* already gone */
      }
      finish({ ok: false, reason: "native_host_timeout" });
    }, NATIVE_HOST_TIMEOUT_MS);

    port.onMessage.addListener((msg) => {
      clearTimeout(timer);
      try {
        port.disconnect();
      } catch {
        /* already gone */
      }
      finish(msg && typeof msg === "object" ? msg : { ok: false, reason: "native_host_bad_response" });
    });

    port.onDisconnect.addListener(() => {
      clearTimeout(timer);
      // lastError is set when the host is not installed/registered; consume it quietly.
      void chrome.runtime.lastError;
      finish({ ok: false, reason: "native_host_unavailable" });
    });

    try {
      port.postMessage({ action: "ensure_server" });
    } catch {
      clearTimeout(timer);
      finish({ ok: false, reason: "native_host_unavailable" });
    }
  });
}

/**
 * Debounced auto-start entry point used by the wake-up hooks and the offline status path.
 * At most one native-host call per ENSURE_DEBOUNCE_MS. Fire-and-forget friendly: callers
 * do not block on the (up-to-25s) server launch; the next health poll observes the result.
 */
async function maybeAutoStartServer() {
  const now = Date.now();
  const last = await getLastEnsureAttempt();
  if (now - last < ENSURE_DEBOUNCE_MS) {
    return { ok: false, reason: "debounced" };
  }
  await setLastEnsureAttempt(now);
  const result = await ensureServerViaNativeHost();
  // A running server means new captures can flush; nudge the queue.
  if (result && result.running) {
    drainQueue({ force: true }).catch(reportError);
  }
  return result;
}

/* -------------------------------------------------------------------------- */
/* queue                                                                      */
/* -------------------------------------------------------------------------- */

function backoffFor(failures) {
  const index = Math.min(Math.max(failures - 1, 0), BACKOFF_MS.length - 1);
  return BACKOFF_MS[index];
}

function conversationId(conversation) {
  return conversation && typeof conversation.source_conversation_id === "string"
    ? conversation.source_conversation_id
    : null;
}

/**
 * Adds a conversation to the durable queue. A newer capture of the same conversation
 * supersedes the queued one — the server upserts, so only the latest snapshot matters and
 * this keeps the queue from growing while a user chats in one long thread offline.
 */
async function enqueue(conversation) {
  return withQueueLock(async () => {
    const { queue, state } = await readQueue();
    const id = conversationId(conversation);
    const item = {
      id: id || `anon-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      conversation,
      attempts: 0,
      queuedAt: new Date().toISOString(),
    };
    const existing = id ? queue.findIndex((entry) => entry.id === id) : -1;
    if (existing >= 0) {
      item.attempts = queue[existing].attempts;
      item.queuedAt = queue[existing].queuedAt;
      queue[existing] = item;
    } else {
      queue.push(item);
    }
    // Oldest first out if we somehow blow past the cap.
    while (queue.length > MAX_QUEUE_ITEMS) queue.shift();
    await writeQueue(queue, state);
    return queue.length;
  });
}

async function postCapture(serverUrl, conversation) {
  const body = JSON.stringify({ conversations: [conversation] });
  if (body.length > MAX_BODY_BYTES) {
    return { ok: false, retryable: false, error: "conversation exceeds the 8 MB capture limit" };
  }
  let response;
  try {
    response = await fetchWithTimeout(
      `${serverUrl}/capture`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      },
      CAPTURE_TIMEOUT_MS
    );
  } catch (error) {
    // Connection refused / aborted / DNS — the server simply isn't up. Normal case.
    return { ok: false, retryable: true, offline: true, error: describeError(error) };
  }

  const payload = await readJson(response);
  if (response.ok) {
    return { ok: true, payload: payload || {} };
  }
  // 4xx (other than throttling) means this body will never be accepted. Drop it.
  const retryable = response.status === 408 || response.status === 429 || response.status >= 500;
  const message = (payload && (payload.error || payload.detail)) || `HTTP ${response.status}`;
  return { ok: false, retryable, error: String(message) };
}

function describeError(error) {
  if (error && error.name === "AbortError") return "request timed out";
  if (error && error.message) return error.message;
  return "network error";
}

/**
 * Drains the queue head-first. Stops at the first retryable failure so ordering is kept
 * and we don't hammer a dead port once per queued item.
 *
 * Returns the number of items still pending.
 */
async function drainQueue({ force = false } = {}) {
  return withQueueLock(async () => {
    const serverUrl = await getServerUrl();
    let { queue, state } = await readQueue();

    if (!queue.length) {
      if (state.offline || state.failures) {
        state = { nextAttemptAt: 0, failures: 0, offline: false, lastError: null };
        await writeQueue(queue, state);
      } else {
        await updateBadge(0, false);
      }
      return 0;
    }

    if (!force && state.nextAttemptAt && Date.now() < state.nextAttemptAt) {
      await updateBadge(queue.length, state.offline);
      scheduleDrain(state.nextAttemptAt - Date.now());
      return queue.length;
    }

    while (queue.length) {
      const item = queue[0];
      const result = await postCapture(serverUrl, item.conversation);

      if (result.ok) {
        queue.shift();
        state = { nextAttemptAt: 0, failures: 0, offline: false, lastError: null };
        await writeQueue(queue, state);
        await storageSet({ lastCaptureAt: new Date().toISOString() });
        continue;
      }

      item.attempts = (item.attempts || 0) + 1;

      if (!result.retryable || item.attempts >= MAX_ATTEMPTS) {
        // Permanently bad payload (or exhausted): drop just this item and keep going so
        // one poisoned conversation cannot block every later capture.
        queue.shift();
        state = { ...state, lastError: result.error };
        await writeQueue(queue, state);
        continue;
      }

      const failures = (state.failures || 0) + 1;
      const delay = backoffFor(failures);
      state = {
        nextAttemptAt: Date.now() + delay,
        failures,
        offline: Boolean(result.offline),
        lastError: result.error,
      };
      await writeQueue(queue, state);
      scheduleDrain(delay);
      return queue.length;
    }

    await writeQueue(queue, state);
    return 0;
  });
}

/**
 * Best-effort in-process timer. The worker may well be killed before it fires — that is
 * fine and expected: the queue and its nextAttemptAt live in storage, and every later
 * wake-up (message from a content script or the popup, browser start, extension install,
 * or the keep-alive alarm if the permission is ever granted) re-enters drainQueue and
 * picks up exactly where this left off. The timer is an optimisation, never the guarantee.
 */
let drainTimer = null;

function scheduleDrain(delayMs) {
  const delay = Math.max(0, Math.min(delayMs, BACKOFF_MS[BACKOFF_MS.length - 1]));
  if (drainTimer) clearTimeout(drainTimer);
  drainTimer = setTimeout(() => {
    drainTimer = null;
    drainQueue().catch(reportError);
  }, delay);
  if (chrome.alarms && chrome.alarms.create) {
    // Only works if the "alarms" permission is present; harmless (and silent) otherwise.
    try {
      chrome.alarms.create("convsearch:drain", { delayInMinutes: Math.max(delay / 60000, 0.5) });
    } catch {
      /* no alarms permission — the wake-up-driven drain covers us. */
    }
  }
}

function reportError(error) {
  // A service worker that throws is invisible to the user; log loudly instead.
  console.error("[convsearch] background error:", error);
}

/* -------------------------------------------------------------------------- */
/* message handlers                                                           */
/* -------------------------------------------------------------------------- */

async function handleCapture(message) {
  const conversation = message && message.conversation;
  if (!conversation || typeof conversation !== "object") {
    return { ok: false, error: "missing conversation" };
  }
  if (!conversationId(conversation)) {
    return { ok: false, error: "conversation has no source_conversation_id" };
  }
  if (!Array.isArray(conversation.messages) || conversation.messages.length === 0) {
    return { ok: false, error: "conversation has no messages" };
  }

  const stored = await storageGet({ captureEnabled: true });
  if (stored.captureEnabled === false) {
    return { ok: true, written: 0, skipped: true };
  }

  const serverUrl = await getServerUrl();
  const direct = await postCapture(serverUrl, conversation);
  if (direct.ok) {
    await storageSet({ lastCaptureAt: new Date().toISOString() });
    await withQueueLock(async () => {
      const { queue, state } = await readQueue();
      const cleared = { nextAttemptAt: 0, failures: 0, offline: false, lastError: null };
      // Server is back — nudge anything that piled up while it was down.
      await writeQueue(queue, state.offline || state.failures ? cleared : state);
    });
    if (drainTimer === null) drainQueue({ force: true }).catch(reportError);
    return {
      ok: true,
      written: Number(direct.payload.conversations_written || 0),
      messages_written: Number(direct.payload.messages_written || 0),
      skipped_unchanged: Number(direct.payload.skipped_unchanged || 0),
      stale_index: Boolean(direct.payload.stale_index),
    };
  }

  if (!direct.retryable) {
    return { ok: false, error: direct.error };
  }

  // The normal case: the local server isn't running. Keep the capture, retry later.
  const pending = await enqueue(conversation);
  await withQueueLock(async () => {
    const { queue, state } = await readQueue();
    const failures = (state.failures || 0) + 1;
    await writeQueue(queue, {
      nextAttemptAt: Date.now() + backoffFor(failures),
      failures,
      offline: Boolean(direct.offline),
      lastError: direct.error,
    });
  });
  scheduleDrain(backoffFor(1));
  return { ok: false, error: direct.error, queued: true, pending };
}

async function handleStatus() {
  const serverUrl = await getServerUrl();
  const stored = await storageGet({ lastCaptureAt: null, captureEnabled: true });
  const { queue, state } = await readQueue();
  const pending = queue.length;

  let online = false;
  let health = null;
  let error = null;
  try {
    const response = await fetchWithTimeout(`${serverUrl}/health`, { method: "GET" }, HEALTH_TIMEOUT_MS);
    health = await readJson(response);
    online = response.ok && Boolean(health);
    if (!online) error = `HTTP ${response.status}`;
  } catch (caught) {
    error = describeError(caught);
  }

  await updateBadge(pending, !online || Boolean(state.offline));

  if (!online) {
    // Server is down — ask the native messaging host to start it (debounced,
    // fire-and-forget). If the host was never installed this is a cheap no-op;
    // otherwise the next status poll observes the server once it comes up.
    maybeAutoStartServer().catch(() => {});
  }

  if (online && pending > 0) {
    // Server came back while the popup was open — start catching up immediately.
    drainQueue({ force: true }).catch(reportError);
  }

  return {
    serverUrl,
    online,
    health,
    lastCaptureAt: stored.lastCaptureAt || null,
    pending,
    captureEnabled: stored.captureEnabled !== false,
    error,
  };
}

// RECOVERY-ONLY. The product is auto-index: the server's AutoIndexer indexes captured
// conversations on its own, so nothing here triggers reindex on normal events. This POST
// /reindex path exists solely as the manual fallback the panel/popup call when the index is
// stale or the automatic pass has failed. Keep the "convsearch:reindex" message contract intact.
async function handleReindex() {
  const serverUrl = await getServerUrl();
  let response;
  try {
    // Deliberately NOT the 2s timeout: a real reindex takes seconds.
    response = await fetchWithTimeout(
      `${serverUrl}/reindex`,
      { method: "POST", headers: { "Content-Type": "application/json" } },
      REINDEX_TIMEOUT_MS
    );
  } catch (error) {
    return { ok: false, error: describeError(error) };
  }
  const payload = await readJson(response);
  if (!response.ok) {
    const message = (payload && (payload.error || payload.detail)) || `HTTP ${response.status}`;
    return { ok: false, error: String(message) };
  }
  return {
    ok: true,
    indexed_passages: Number((payload && payload.indexed_passages) || 0),
    stale_index: Boolean(payload && payload.stale_index),
  };
}

/* -------------------------------------------------------------------------- */
/* side-panel read handlers — thin GET proxies over the local server           */
/* -------------------------------------------------------------------------- */

function handleSearch(message) {
  const p = (message && message.params) || {};
  return apiGet("/search", {
    q: p.q,
    level: p.level,
    explain: p.explain,
    limit: p.limit,
    passages: p.passages,
    profile: p.profile,
    branches: p.branches,
  });
}

function handleAsk(message) {
  const p = (message && message.params) || {};
  return apiGet("/ask", {
    q: p.q,
    limit: p.limit,
    passages: p.passages,
    backend: p.backend,
  });
}

function handleMemories(message) {
  const p = (message && message.params) || {};
  return apiGet("/memories", {
    q: p.q,
    kind: p.kind,
    status: p.status,
    project: p.project,
    limit: p.limit,
  });
}

function handleMemory(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing memory id", status: 0 });
  return apiGet(`/memories/${encodeURIComponent(String(id))}`);
}

function handleProjects() {
  return apiGet("/projects");
}

function handleProject(message) {
  const name = message && message.name;
  if (!name) return Promise.resolve({ ok: false, error: "missing project name", status: 0 });
  return apiGet(`/projects/${encodeURIComponent(String(name))}`);
}

function handleConversation(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing conversation id", status: 0 });
  return apiGet(`/conversation/${encodeURIComponent(String(id))}`);
}

function handleProjectExport(message) {
  const name = message && message.name;
  if (!name) return Promise.resolve({ ok: false, error: "missing project name", status: 0 });
  return apiGet(`/projects/${encodeURIComponent(String(name))}/export`);
}

function handleTasks(message) {
  const p = (message && message.params) || {};
  return apiGet("/tasks", {
    state: p.state,
    project: p.project,
    limit: p.limit,
    since: p.since,
    evidence: p.evidence,
  });
}

function handleTimeline(message) {
  const p = (message && message.params) || {};
  return apiGet("/timeline", {
    q: p.q,
    project: p.project,
    limit: p.limit,
    evidence: p.evidence,
  });
}

function handleCaptures(message) {
  const p = (message && message.params) || {};
  return apiGet("/captures", {
    source: p.source,
    limit: p.limit,
    problems: p.problems,
  });
}

function handleMemoriesReview(message) {
  const p = (message && message.params) || {};
  return apiGet("/memories/review", {
    limit: p.limit,
    kind: p.kind,
    project: p.project,
    include_reviewed: p.include_reviewed,
  });
}

function handlePrivacy() {
  return apiGet("/privacy", {}, HEALTH_TIMEOUT_MS);
}

/**
 * Local-model setup assistant. Thin GET proxy over the fixed /diagnostics contract: the panel
 * renders `ready`/`backend`/`summary` as a banner, `llm_checks`/`doctor_checks` as check rows,
 * and — only when `ready` is false — `remediation` as a copy-pasteable command block.
 */
function handleDiagnostics() {
  return apiGet("/diagnostics", {}, HEALTH_TIMEOUT_MS);
}

/**
 * The three review mutators (confirm/invalidate/pin) all share this shape: a memory id in the
 * path, an optional `reason` string in the body, and — for pin only — a required boolean
 * `pinned`. Each POST returns the updated item so the panel can re-render from the response
 * instead of refetching the whole queue.
 */
function reviewMutationBody(p, extra) {
  const body = extra ? { ...extra } : {};
  if (p.reason !== null && p.reason !== undefined && p.reason !== "") body.reason = p.reason;
  return body;
}

/**
 * Task complete/reopen mirror the review mutators: id in the path, optional `reason` in the
 * body, updated task item returned so the panel re-renders that one row from the response.
 */
function handleTaskComplete(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing task id", status: 0 });
  const p = (message && message.params) || {};
  return apiPost(`/tasks/${encodeURIComponent(String(id))}/complete`, reviewMutationBody(p));
}

function handleTaskReopen(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing task id", status: 0 });
  const p = (message && message.params) || {};
  return apiPost(`/tasks/${encodeURIComponent(String(id))}/reopen`, reviewMutationBody(p));
}

function handleMemoryConfirm(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing memory id", status: 0 });
  const p = (message && message.params) || {};
  return apiPost(`/memories/${encodeURIComponent(String(id))}/confirm`, reviewMutationBody(p));
}

function handleMemoryInvalidate(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing memory id", status: 0 });
  const p = (message && message.params) || {};
  return apiPost(`/memories/${encodeURIComponent(String(id))}/invalidate`, reviewMutationBody(p));
}

function handleMemoryPin(message) {
  const id = message && message.id;
  if (!id) return Promise.resolve({ ok: false, error: "missing memory id", status: 0 });
  const p = (message && message.params) || {};
  return apiPost(`/memories/${encodeURIComponent(String(id))}/pin`, reviewMutationBody(p, { pinned: Boolean(p.pinned) }));
}

/* -------------------------------------------------------------------------- */
/* learning — interaction logging, suggestions, stats                          */
/* -------------------------------------------------------------------------- */

const FEEDBACK_EVENTS = new Set(["search", "open", "inspect", "ask"]);

/**
 * Records one interaction event. The panel fires these and forgets, so this only needs to be a
 * thin POST proxy over the fixed /feedback contract. Unknown/absent fields are dropped so the
 * caller can pass a flat object; event_type is validated to keep obviously-bad writes off the
 * wire. A short timeout keeps a dead server from tying up the channel on a background write.
 */
function handleFeedback(message) {
  const p = (message && message.params) || {};
  if (!FEEDBACK_EVENTS.has(p.event_type)) {
    return Promise.resolve({ ok: false, error: "invalid event_type", status: 0 });
  }
  const body = { event_type: p.event_type };
  for (const key of ["query", "conversation_id", "passage_id", "segment_id", "position"]) {
    if (p[key] !== null && p[key] !== undefined && p[key] !== "") body[key] = p[key];
  }
  return apiPost("/feedback", body, CAPTURE_TIMEOUT_MS);
}

function handleSuggestions(message) {
  const p = (message && message.params) || {};
  return apiGet("/suggestions", { limit: p.limit }, HEALTH_TIMEOUT_MS);
}

function handleLearnStats() {
  return apiGet("/learn/stats", {}, HEALTH_TIMEOUT_MS);
}

/**
 * Grounded planner. Thin GET proxy over the fixed /plan contract: the panel passes the raw
 * question and renders the returned answer/intent/steps/calls/findings itself.
 */
function handlePlan(message) {
  const p = (message && message.params) || {};
  return apiGet("/plan", { q: p.q });
}

/**
 * User-initiated self-improvement pass. May run the local LLM over the interaction log, so it
 * gets the long LEARN_TIMEOUT_MS rather than the default query ceiling. `use_llm` defaults to
 * true; the panel exposes it as an opt-out checkbox.
 */
function handleLearnRun(message) {
  const p = (message && message.params) || {};
  const useLlm = p.use_llm === undefined ? true : Boolean(p.use_llm);
  return apiPost("/learn", { use_llm: useLlm }, LEARN_TIMEOUT_MS);
}

function handleLearnPrefs(message) {
  const p = (message && message.params) || {};
  return apiGet("/learn/preferences", { limit: p.limit }, HEALTH_TIMEOUT_MS);
}

const HANDLERS = {
  "convsearch:capture": handleCapture,
  "convsearch:status": handleStatus,
  "convsearch:ensureServer": maybeAutoStartServer,
  "convsearch:reindex": handleReindex,
  "convsearch:search": handleSearch,
  "convsearch:ask": handleAsk,
  "convsearch:memories": handleMemories,
  "convsearch:memory": handleMemory,
  "convsearch:projects": handleProjects,
  "convsearch:project": handleProject,
  "convsearch:conversation": handleConversation,
  "convsearch:projectExport": handleProjectExport,
  "convsearch:diagnostics": handleDiagnostics,
  "convsearch:tasks": handleTasks,
  "convsearch:timeline": handleTimeline,
  "convsearch:captures": handleCaptures,
  "convsearch:memoriesReview": handleMemoriesReview,
  "convsearch:memoryConfirm": handleMemoryConfirm,
  "convsearch:memoryInvalidate": handleMemoryInvalidate,
  "convsearch:memoryPin": handleMemoryPin,
  "convsearch:taskComplete": handleTaskComplete,
  "convsearch:taskReopen": handleTaskReopen,
  "convsearch:privacy": handlePrivacy,
  "convsearch:feedback": handleFeedback,
  "convsearch:suggestions": handleSuggestions,
  "convsearch:learnStats": handleLearnStats,
  "convsearch:plan": handlePlan,
  "convsearch:learnRun": handleLearnRun,
  "convsearch:learnPrefs": handleLearnPrefs,
};

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const handler = message && HANDLERS[message.type];
  if (!handler) return false;

  // `return true` keeps the message channel open; without it the async reply is dropped.
  Promise.resolve()
    .then(() => handler(message))
    .then((result) => sendResponse(result))
    .catch((error) => {
      reportError(error);
      sendResponse({ ok: false, error: describeError(error) });
    });
  return true;
});

/* -------------------------------------------------------------------------- */
/* wake-up hooks — every one of these re-enters the persisted queue           */
/* -------------------------------------------------------------------------- */

function resumeFromStorage() {
  readQueue()
    .then(({ queue, state }) => {
      updateBadge(queue.length, state.offline).catch(reportError);
      if (queue.length) drainQueue().catch(reportError);
    })
    .catch(reportError);
}

chrome.runtime.onStartup.addListener(resumeFromStorage);
chrome.runtime.onInstalled.addListener(resumeFromStorage);
// On wake, try to bring the local server up on its own via the native host.
chrome.runtime.onStartup.addListener(() => {
  maybeAutoStartServer().catch(() => {});
});
chrome.runtime.onInstalled.addListener(() => {
  maybeAutoStartServer().catch(() => {});
});

/* -------------------------------------------------------------------------- */
/* context menu → side panel                                                  */
/* -------------------------------------------------------------------------- */

const CONTEXT_MENU_ID = "convsearch:search-selection";

/**
 * Selecting text on any page and choosing "Search convsearch for …" stashes the selection and
 * opens the side panel. sidePanel.open() must run inside the click's user gesture, so we open
 * first and write the pending query in parallel; the panel reads it on load. Menus are
 * (re)created on install/update — creating an existing id throws, so it's swept first.
 */
function createContextMenu() {
  if (!chrome.contextMenus || !chrome.contextMenus.create) return;
  try {
    chrome.contextMenus.removeAll(() => {
      void chrome.runtime.lastError;
      try {
        chrome.contextMenus.create({
          id: CONTEXT_MENU_ID,
          title: 'Search convsearch for "%s"',
          contexts: ["selection"],
        });
      } catch (error) {
        reportError(error);
      }
    });
  } catch (error) {
    reportError(error);
  }
}

chrome.runtime.onInstalled.addListener(createContextMenu);
chrome.runtime.onStartup.addListener(createContextMenu);

if (chrome.contextMenus && chrome.contextMenus.onClicked) {
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (!info || info.menuItemId !== CONTEXT_MENU_ID) return;
    const text = (info.selectionText || "").trim();
    if (!text) return;
    // Cap it: a menu selection can be a whole paragraph; a search query need not be.
    const pendingQuery = { text: text.slice(0, 400), at: Date.now() };
    storageSet({ pendingQuery }).catch(reportError);
    const windowId = tab && tab.windowId;
    if (chrome.sidePanel && chrome.sidePanel.open && windowId !== undefined) {
      chrome.sidePanel.open({ windowId }).catch(reportError);
    }
  });
}

// Let the toolbar icon keep its quick-search popup; the panel is reached via the context menu
// and the popup's "Open side panel" button. This call is harmless if the API is absent.
if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: false })
    .catch(() => {
      /* older Chrome without the API — the context menu path still works. */
    });
}

if (chrome.alarms && chrome.alarms.onAlarm) {
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm && alarm.name === "convsearch:drain") drainQueue().catch(reportError);
  });
}

// Cold start of this worker for any reason (including a message that woke it).
resumeFromStorage();
