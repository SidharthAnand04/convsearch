"use strict";

// Mirror the side panel's chosen theme so the popup matches it. theme.css defaults to dark;
// the panel persists its choice under "convsearch:theme", and setting data-theme on <html>
// lets the light overrides in theme.css take effect. Read as early as possible to avoid a flash.
try {
  chrome.storage.local.get({ "convsearch:theme": null }, (stored) => {
    const theme = stored && stored["convsearch:theme"];
    if (theme === "light" || theme === "dark") {
      document.documentElement.dataset.theme = theme;
    }
  });
} catch {
  /* no chrome.storage (e.g. plain-page tests) — theme.css dark default stands. */
}

const DEFAULT_SERVER = "http://127.0.0.1:8756";
const SERVE_HINT = "convsearch serve --workspace ./workspace";

/** As-you-type delay. Long enough to skip mid-word noise, short enough to feel live. */
const DEBOUNCE_MS = 150;
/** A one-character query is nearly always a wasted round trip; explicit submit ignores this. */
const MIN_AUTO_QUERY = 2;
const SHORTCUTS_URL = "chrome://extensions/shortcuts";

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const branchesInput = document.getElementById("branches");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const listEl = document.getElementById("result-list");

const helpEl = document.getElementById("help");
const helpToggle = document.getElementById("help-toggle");
const editShortcutsButton = document.getElementById("edit-shortcuts");

const captureEl = document.getElementById("capture");
const captureDot = document.getElementById("capture-dot");
const captureSummary = document.getElementById("capture-summary");
const captureDetail = document.getElementById("capture-detail");
const captureAlert = document.getElementById("capture-alert");
const captureAlertText = document.getElementById("capture-alert-text");
const reindexButton = document.getElementById("reindex");
const refreshButton = document.getElementById("refresh");
const optionsButton = document.getElementById("open-options");
const panelButton = document.getElementById("open-panel");

let serverUrl = DEFAULT_SERVER;
let captureEnabled = true;
/** Latest {online, health, lastCaptureAt, pending} from the background worker. */
let state = null;
let reindexing = false;

/* ---------- small helpers ---------- */

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function timeAgo(iso) {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${plural(minutes, "minute")} ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${plural(hours, "hour")} ago`;
  return `${plural(Math.round(hours / 24), "day")} ago`;
}

/** The background service worker owns all HTTP; the popup only talks to it. */
function ask(message) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (value) => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          done({ ok: false, error: chrome.runtime.lastError.message });
          return;
        }
        done(response || { ok: false, error: "The background worker sent no reply." });
      });
    } catch (error) {
      done({ ok: false, error: String((error && error.message) || error) });
    }
  });
}

/* ---------- rendering (never innerHTML: conversation text is untrusted markup) ---------- */

function renderPassage(passage) {
  const wrap = document.createElement("div");
  wrap.className = passage.is_primary_path === false ? "passage alternate" : "passage";

  const role = document.createElement("span");
  role.className = "role";
  role.textContent = passage.role || "message";
  wrap.append(role);

  const text = passage.text || "";
  wrap.append(text.length > 300 ? `${text.slice(0, 300)}…` : text);
  return wrap;
}

function conversationUrl(result) {
  return typeof result.url === "string" && result.url.startsWith("https://chatgpt.com/")
    ? result.url
    : null;
}

function renderResult(result, index) {
  const article = document.createElement("article");
  article.className = "result";
  // aria-activedescendant on the query box points here, so every option needs a stable id.
  article.id = `result-${index}`;
  article.setAttribute("role", "option");
  article.setAttribute("aria-selected", "false");

  const url = conversationUrl(result);
  if (url) article.dataset.url = url;

  const head = document.createElement("div");
  head.className = "result-head";

  const marker = document.createElement("span");
  marker.className = "result-marker";
  marker.setAttribute("aria-hidden", "true");
  marker.textContent = "›";
  head.append(marker);

  const title = document.createElement("a");
  title.className = "result-title";
  title.textContent = result.title || "(untitled)";
  if (url) {
    title.href = url;
    title.target = "_blank";
    title.rel = "noreferrer";
  }
  // Focus stays in the query box; the list is driven by aria-activedescendant, so the
  // anchors must not be separate tab stops.
  title.tabIndex = -1;
  head.append(title);

  const meta = document.createElement("span");
  meta.className = "result-meta";
  const date = (result.updated_at || result.created_at || "").slice(0, 10);
  const score = typeof result.score === "number" ? result.score.toFixed(3) : "";
  meta.textContent = [date, score].filter(Boolean).join(" · ");
  head.append(meta);

  article.append(head);
  for (const passage of result.passages || []) {
    article.append(renderPassage(passage));
  }
  return article;
}

/**
 * Empty and error states. Each one says what happened and what to do next.
 * `spec` = { title, body, code, actionLabel, onAction }.
 */
function renderEmpty(spec) {
  const wrap = document.createElement("div");
  wrap.className = "empty";

  const title = document.createElement("strong");
  title.className = "empty-title";
  title.textContent = spec.title;
  wrap.append(title);

  if (spec.body) wrap.append(spec.body);

  if (spec.code) {
    const code = document.createElement("code");
    code.textContent = spec.code;
    wrap.append(code);
  }

  if (spec.actionLabel && spec.onAction) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = spec.actionLabel;
    button.addEventListener("click", spec.onAction);
    wrap.append(button);
  }
  return wrap;
}

/**
 * The result list is a persistent listbox; empty states are siblings of it inside #results.
 * Keeping them apart means a listbox never contains non-option children, and lets a search
 * swap results without destroying the element aria-activedescendant refers to.
 */
function clearEmptyStates() {
  for (const node of Array.from(resultsEl.querySelectorAll(".empty"))) node.remove();
}

function clearResults() {
  clear(listEl);
  setSelection(-1);
  queryInput.setAttribute("aria-expanded", "false");
}

function showEmpty(spec) {
  clearResults();
  clearEmptyStates();
  setStale(false);
  resultsEl.append(renderEmpty(spec));
}

/* ---------- selection (keyboard driven) ---------- */

let selectedIndex = -1;

function resultElements() {
  return Array.from(listEl.children);
}

function setSelection(index, { scroll = true } = {}) {
  const items = resultElements();
  if (!items.length) {
    selectedIndex = -1;
    queryInput.removeAttribute("aria-activedescendant");
    return;
  }
  const next = Math.max(-1, Math.min(index, items.length - 1));
  selectedIndex = next;

  items.forEach((item, i) => {
    const on = i === next;
    item.classList.toggle("selected", on);
    item.setAttribute("aria-selected", on ? "true" : "false");
  });

  if (next < 0) {
    queryInput.removeAttribute("aria-activedescendant");
    return;
  }
  const active = items[next];
  queryInput.setAttribute("aria-activedescendant", active.id);
  if (scroll) active.scrollIntoView({ block: "nearest" });
}

function moveSelection(delta) {
  const items = resultElements();
  if (!items.length) return false;
  // From "nothing selected", Down lands on the first row and Up on the last.
  const from = selectedIndex < 0 ? (delta > 0 ? -1 : 0) : selectedIndex;
  let next = from + delta;
  if (next < 0) next = items.length - 1;
  if (next >= items.length) next = 0;
  setSelection(next);
  return true;
}

/**
 * `background: true` keeps the popup open so several conversations can be queued in a row —
 * the whole point of Shift+Enter here.
 */
function openSelected({ background = false } = {}) {
  const items = resultElements();
  const item = items[selectedIndex];
  const url = item && item.dataset.url;
  if (!url || !url.startsWith("https://chatgpt.com/")) return false;

  if (chrome.tabs && chrome.tabs.create) {
    chrome.tabs.create({ url, active: !background });
  } else {
    window.open(url, "_blank", "noreferrer");
  }
  if (background) {
    setStatus("opened in a background tab");
    return true;
  }
  window.close();
  return true;
}

/* ---------- copy for each state ---------- */

const COPY = {
  offline: {
    title: "The convsearch server isn't running.",
    body:
      "Nothing can be captured or searched until it is. Start it in a terminal from " +
      "the convsearch folder, then reopen this popup.",
    code: SERVE_HINT,
  },
  nothingCaptured: {
    title: "Nothing captured yet.",
    body:
      "There's nothing to set up — just open chatgpt.com and read your conversations. " +
      "convsearch saves each one you visit to your own machine and indexes it a few " +
      "seconds later, so it becomes searchable on its own.",
  },
  indexing: {
    title: "Indexing…",
    body:
      "The conversations you just opened are being added to the search index. This " +
      "usually takes a few seconds; no action needed.",
  },
  captureOff: {
    title: "Capture is turned off.",
    body:
      "convsearch isn't saving the conversations you open. Turn capture back on in " +
      "Settings, then browse chatgpt.com to build up your searchable history.",
  },
  notIndexed: {
    title: "Captured, but not searchable yet.",
    body:
      "Your conversations are stored locally, but they haven't been indexed. This " +
      "normally happens by itself within a few seconds of opening a conversation, so " +
      "if you're seeing this the automatic pass may have failed — rebuild to fix it.",
  },
  noResults: {
    title: "No conversations matched that.",
    body:
      "Search matches meaning, not exact words, so try describing what the " +
      "conversation was about rather than quoting it.",
  },
};

/* ---------- capture status panel ---------- */

function renderCaptureState() {
  captureEl.hidden = false;
  clear(captureDetail);
  captureAlert.hidden = true;
  reindexButton.disabled = reindexing;

  if (!state || !state.online) {
    captureDot.className = "dot offline";
    captureSummary.textContent = "Server offline";
    captureDetail.textContent = `Not reachable at ${serverUrl}`;
    return;
  }

  const health = state.health || {};
  const captured = health.captured_conversations || 0;
  const total = health.conversations || 0;
  const stale = Boolean(health.stale_index);
  const indexed = Boolean(health.indexed);
  // The server indexes captured conversations on its own; `indexing` is true while a pass
  // is running or queued. Auto-indexing can be switched off with `serve --no-auto-index`.
  const indexing = Boolean(health.indexing);
  const autoIndex = health.auto_index !== false;

  captureDot.className = indexing ? "dot warn" : stale || !indexed ? "dot warn" : "dot online";
  captureSummary.textContent = `${plural(total, "conversation")} · ${captured} captured live`;

  const bits = [];
  const ago = timeAgo(state.lastCaptureAt);
  bits.push(ago ? `Last capture ${ago}` : "No capture yet");
  if (health.messages) bits.push(plural(health.messages, "message"));
  if (state.pending) bits.push(`${state.pending} queued`);
  if (!captureEnabled) bits.push("capture off");
  captureDetail.textContent = bits.join(" · ");

  if (reindexing) {
    captureAlert.hidden = false;
    captureAlertText.textContent = "Rebuilding the index — this takes a few seconds…";
    reindexButton.disabled = true;
    reindexButton.textContent = "Rebuilding…";
    return;
  }

  reindexButton.textContent = "Rebuild index";

  // While the server is indexing on its own there is nothing for the user to do, so say
  // what is happening instead of showing a warning with a button.
  if (indexing) {
    captureSummary.textContent += " · indexing…";
    return;
  }

  // A warning is only honest once automatic indexing has finished or is disabled;
  // otherwise it would fire during perfectly normal operation.
  if (!indexed && captured) {
    captureAlert.hidden = false;
    captureAlertText.textContent = autoIndex
      ? "Nothing is searchable yet. Indexing should start on its own — rebuild if it does not."
      : "Nothing is searchable yet — build the index first.";
  } else if (stale) {
    captureAlert.hidden = false;
    captureAlertText.textContent = autoIndex
      ? "Some conversations aren't indexed yet. Automatic indexing may have failed."
      : "Newly captured conversations aren't searchable yet. Rebuild to include them.";
  }
}

/** Pick the empty state that matches the current server/capture situation. */
function renderIdleState() {
  // The first status round trip can land AFTER the user has already typed — opening the
  // popup and typing straight away is the normal case, and this used to wipe the results
  // that had just arrived. An idle state is only correct when nothing is being searched.
  if (queryInput.value.trim() && (listEl.children.length || inFlight || debounceTimer !== null)) {
    return;
  }

  if (!state || !state.online) {
    setStatus("server offline", true);
    showEmpty(COPY.offline);
    return;
  }

  const health = state.health || {};
  const captured = health.captured_conversations || 0;
  const total = health.conversations || 0;

  if (!total && !captured) {
    setStatus("nothing captured", true);
    showEmpty(captureEnabled ? COPY.nothingCaptured : COPY.captureOff);
    return;
  }

  // An indexing pass in flight is the normal state right after opening a conversation.
  // Report it rather than offering a button, and let the poll below flip us to ready.
  if (health.indexing) {
    setStatus("indexing…");
    showEmpty(COPY.indexing);
    return;
  }

  if (!health.indexed) {
    setStatus("not indexed", true);
    showEmpty({
      ...COPY.notIndexed,
      actionLabel: "Build index now",
      onAction: runReindex,
    });
    return;
  }

  setStatus(`${plural(total, "conversation")} searchable`);
  clearResults();
  clearEmptyStates();
}

/**
 * While the server is indexing, keep checking so the popup flips to "searchable" without
 * the user having to close and reopen it. Cleared as soon as indexing stops, and on
 * unload, so a forgotten interval cannot keep waking the service worker.
 */
let indexingPoll = null;

function stopIndexingPoll() {
  if (indexingPoll !== null) {
    clearInterval(indexingPoll);
    indexingPoll = null;
  }
}

function syncIndexingPoll() {
  const indexing = Boolean(state && state.health && state.health.indexing);
  if (indexing && indexingPoll === null) {
    indexingPoll = setInterval(() => {
      refreshStatus({ rerenderResults: !listEl.querySelector(".result") });
    }, 2000);
  } else if (!indexing) {
    stopIndexingPoll();
  }
}

window.addEventListener("unload", stopIndexingPoll);

async function refreshStatus({ rerenderResults = true } = {}) {
  const response = await ask({ type: "convsearch:status" });
  if (response && response.serverUrl) serverUrl = response.serverUrl.replace(/\/+$/, "");
  state = response && typeof response === "object" ? response : { online: false };
  if (response && response.ok === false && !response.health) state.online = false;
  renderCaptureState();
  if (rerenderResults) renderIdleState();
  syncIndexingPoll();
  return state;
}

async function runReindex() {
  if (reindexing) return;
  reindexing = true;
  renderCaptureState();
  setStatus("rebuilding index…");

  const response = await ask({ type: "convsearch:reindex" });
  reindexing = false;

  if (!response || response.ok === false) {
    await refreshStatus({ rerenderResults: false });
    setStatus("rebuild failed", true);
    showEmpty({
      title: "Rebuilding the index failed.",
      body:
        (response && response.error) ||
        "The server couldn't finish the rebuild. Check that it's still running, then try again.",
      actionLabel: "Try again",
      onAction: runReindex,
    });
    return;
  }

  const count = response.indexed_passages;
  await refreshStatus({ rerenderResults: false });
  setStatus(
    typeof count === "number" ? `indexed ${plural(count, "passage")}` : "index rebuilt"
  );

  const query = queryInput.value.trim();
  if (query) {
    runSearch(query, { force: true });
  } else {
    showEmpty({
      title: "Index rebuilt.",
      body: "Everything captured so far is searchable now — try a query above.",
    });
  }
}

/* ---------- search ---------- */

/**
 * Every request gets a sequence number and its own AbortController. A response is only
 * allowed to touch the DOM if its sequence is still the newest one, so a slow earlier
 * request can never overwrite the results of a newer one — with or without abort landing
 * first. Aborting as well saves the server the wasted work.
 */
let searchSeq = 0;
let inFlight = null;
let debounceTimer = null;
/** What the newest issued request asked for, so identical repeats are skipped. */
let lastRequested = null;

/** Dim the old results instead of blanking them: something is on screen at all times. */
function setStale(stale) {
  listEl.classList.toggle("stale", Boolean(stale) && listEl.children.length > 0);
}

function cancelPending() {
  if (debounceTimer !== null) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
}

function abortInFlight() {
  if (inFlight) {
    inFlight.abort();
    inFlight = null;
  }
}

function requestKey(query) {
  return `${branchesInput.checked ? "b" : "-"} ${query}`;
}

async function runSearch(query, { force = false } = {}) {
  const key = requestKey(query);
  if (!force && key === lastRequested) return;
  lastRequested = key;

  cancelPending();
  abortInFlight();

  const seq = ++searchSeq;
  const controller = new AbortController();
  inFlight = controller;

  const params = new URLSearchParams({ q: query, limit: "10", passages: "2" });
  if (branchesInput.checked) params.set("branches", "1");

  // Keep whatever is already rendered and dim it. Never blank the list mid-typing.
  setStale(true);
  setStatus("searching…");
  submitButton.disabled = true;

  let response;
  try {
    response = await fetch(`${serverUrl}/search?${params}`, { signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted || seq !== searchSeq) return;
    inFlight = null;
    submitButton.disabled = false;
    setStatus("server offline", true);
    await refreshStatus({ rerenderResults: false });
    if (seq === searchSeq) showEmpty(COPY.offline);
    return;
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }

  // The only gate that matters: a stale response returns here without touching anything.
  if (seq !== searchSeq) return;
  inFlight = null;
  submitButton.disabled = false;

  if (!response.ok) {
    setStatus(`error ${response.status}`, true);
    showEmpty({
      title: `The server returned an error (${response.status}).`,
      body:
        payload.error ||
        "The search request didn't succeed. Check the terminal running convsearch serve for details.",
    });
    return;
  }

  const results = payload.results || [];
  const count = payload.count || results.length || 0;
  setStatus(`${plural(count, "result")}`);

  if (!count) {
    const stale = Boolean(state && state.health && state.health.stale_index);
    showEmpty(
      stale
        ? {
            title: "No conversations matched that.",
            body:
              "Some recently captured conversations aren't in the index yet, so they " +
              "can't match. Rebuild the index and search again.",
            actionLabel: "Rebuild index",
            onAction: runReindex,
          }
        : COPY.noResults
    );
    return;
  }

  clearEmptyStates();
  clear(listEl);
  results.forEach((result, index) => listEl.append(renderResult(result, index)));
  setStale(false);
  queryInput.setAttribute("aria-expanded", "true");
  // Pre-select the top hit so Enter is a single keystroke away from opening it.
  setSelection(0, { scroll: false });

  // Keep the panel honest about staleness while results are on screen.
  refreshStatus({ rerenderResults: false });
}

/** Called from the input event. Debounced, and short queries are left alone. */
function scheduleSearch() {
  cancelPending();
  const query = queryInput.value.trim();

  if (!query) {
    abortInFlight();
    searchSeq += 1; // invalidate anything still in flight
    lastRequested = null;
    renderIdleState();
    return;
  }
  if (query.length < MIN_AUTO_QUERY) {
    // Don't blank or show an empty state for a single character — just wait.
    setStale(true);
    return;
  }
  setStale(true);
  debounceTimer = setTimeout(() => {
    debounceTimer = null;
    runSearch(query);
  }, DEBOUNCE_MS);
}

function submitSearch() {
  cancelPending();
  const query = queryInput.value.trim();
  chrome.storage.local.set({ lastQuery: queryInput.value });
  if (query) runSearch(query, { force: true });
}

/* ---------- help panel ---------- */

function setHelpVisible(visible) {
  helpEl.hidden = !visible;
  helpToggle.setAttribute("aria-expanded", visible ? "true" : "false");
}

function toggleHelp() {
  setHelpVisible(helpEl.hidden);
}

function openShortcutsPage() {
  if (chrome.tabs && chrome.tabs.create) {
    chrome.tabs.create({ url: SHORTCUTS_URL });
    window.close();
  }
}

/* ---------- keyboard ---------- */

function isTextField(node) {
  if (!node) return false;
  const tag = node.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || node.isContentEditable === true;
}

function focusQuery({ select = false } = {}) {
  queryInput.focus();
  if (select) queryInput.select();
}

/**
 * One document-level handler in the capture phase, so it wins over the browser's own
 * behaviour for `type="search"` (which clears the field on Escape) and over any focused
 * button's default Enter/Space handling for keys we claim.
 */
document.addEventListener(
  "keydown",
  (event) => {
    if (event.isComposing) return;

    const key = event.key;
    const mod = event.ctrlKey || event.metaKey;
    const inText = isTextField(event.target);

    // ---- help ----
    if (mod && key === "/") {
      event.preventDefault();
      toggleHelp();
      return;
    }
    if (key === "?" && !inText) {
      event.preventDefault();
      toggleHelp();
      return;
    }
    if (key === "F1") {
      event.preventDefault();
      toggleHelp();
      return;
    }

    // ---- movement ----
    if (key === "ArrowDown" || (mod && !event.shiftKey && (key === "j" || key === "J"))) {
      if (moveSelection(1)) {
        event.preventDefault();
        focusQuery();
      }
      return;
    }
    if (key === "ArrowUp" || (mod && !event.shiftKey && (key === "k" || key === "K"))) {
      if (moveSelection(-1)) {
        event.preventDefault();
        focusQuery();
      }
      return;
    }
    // Inside a text box Home/End are caret keys and must stay that way, so jumping to the
    // first/last result needs the modifier there. Outside one, the bare key is enough.
    if (key === "Home" || key === "End") {
      const claim = mod || !inText;
      if (claim && resultElements().length) {
        event.preventDefault();
        setSelection(key === "Home" ? 0 : resultElements().length - 1);
        focusQuery();
      }
      return;
    }

    // ---- open / search ----
    if (key === "Enter") {
      // Let a focused non-submit button do its own thing.
      if (event.target instanceof HTMLButtonElement && event.target.type !== "submit") return;

      const background = mod || event.shiftKey;
      if (selectedIndex >= 0 && openSelected({ background })) {
        event.preventDefault();
        return;
      }
      event.preventDefault();
      submitSearch();
      return;
    }

    // ---- escape ----
    if (key === "Escape") {
      event.preventDefault();
      if (!helpEl.hidden) {
        setHelpVisible(false);
        focusQuery();
        return;
      }
      if (queryInput.value !== "") {
        queryInput.value = "";
        chrome.storage.local.set({ lastQuery: "" });
        cancelPending();
        abortInFlight();
        searchSeq += 1;
        lastRequested = null;
        focusQuery();
        renderIdleState();
        return;
      }
      window.close();
      return;
    }

    // ---- refocus ----
    if (key === "/" && !mod && !inText) {
      event.preventDefault();
      focusQuery({ select: true });
      return;
    }
  },
  true
);

/* ---------- wiring ---------- */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitSearch();
});

queryInput.addEventListener("input", scheduleSearch);

queryInput.addEventListener("change", () => {
  chrome.storage.local.set({ lastQuery: queryInput.value });
});

branchesInput.addEventListener("change", () => {
  const query = queryInput.value.trim();
  if (query) runSearch(query, { force: true });
});

// Mouse and keyboard drive the same selection, so clicking a row then pressing Enter or
// arrowing on from it behaves the way it looks like it should.
listEl.addEventListener("mousedown", (event) => {
  const item = event.target.closest(".result");
  if (!item) return;
  const index = resultElements().indexOf(item);
  if (index >= 0) setSelection(index, { scroll: false });
});

helpToggle.addEventListener("click", toggleHelp);
editShortcutsButton.addEventListener("click", openShortcutsPage);

reindexButton.addEventListener("click", runReindex);

refreshButton.addEventListener("click", () => {
  setStatus("checking…");
  refreshStatus();
});

optionsButton.addEventListener("click", () => {
  if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
});

// The side panel is the full app; the popup stays as the quick-search fallback. Opening the
// panel must happen inside this click's user gesture, so it's called directly here.
if (panelButton) {
  panelButton.addEventListener("click", () => {
    if (!chrome.sidePanel || !chrome.sidePanel.open) {
      setStatus("side panel needs a newer Chrome", true);
      return;
    }
    chrome.windows.getCurrent((win) => {
      if (chrome.runtime.lastError || !win) return;
      chrome.sidePanel.open({ windowId: win.id }).then(
        () => window.close(),
        () => setStatus("couldn't open the side panel", true)
      );
    });
  });
}

chrome.storage.local.get(
  { serverUrl: DEFAULT_SERVER, lastQuery: "", captureEnabled: true },
  (stored) => {
    serverUrl = String(stored.serverUrl || DEFAULT_SERVER).replace(/\/+$/, "");
    captureEnabled = stored.captureEnabled !== false;
    // Only restore if the user (or a test) has not already typed into the box.
    if (queryInput.value === "") queryInput.value = stored.lastQuery || "";
    // The whole previous query is selected, so the first keystroke replaces it rather than
    // appending to it — the popup is meant to be opened and typed into immediately.
    focusQuery({ select: true });
    setStatus("checking…");
    refreshStatus();
  }
);

// `autofocus` is unreliable in an extension popup, so ask explicitly as well; the storage
// callback above focuses again once it lands.
focusQuery();
