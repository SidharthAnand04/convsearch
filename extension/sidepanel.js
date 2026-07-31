"use strict";

/*
 * convsearch side panel.
 *
 * Like the popup, this page speaks NO HTTP itself. Every call goes through a chrome.runtime
 * message to the background service worker, which owns the loopback allow-list and all fetch/
 * error handling and replies with a uniform { ok, data } | { ok:false, error, status } envelope.
 *
 * Server text (titles, quotes, answers, statements) is untrusted markup and is NEVER assigned
 * as innerHTML. Every node is built with the `el` helper, which only ever sets text through
 * textContent / createTextNode.
 */

const DEBOUNCE_MS = 220;
const MIN_AUTO_QUERY = 2;
const QUOTE_LIMIT = 260;
const PENDING_MAX_AGE_MS = 60000;

/* ------------------------------------------------------------------ */
/* message bridge                                                     */
/* ------------------------------------------------------------------ */

/** Send a message to the SW; always resolves to an envelope, never throws. */
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
          done({ ok: false, error: chrome.runtime.lastError.message, status: 0 });
          return;
        }
        done(response || { ok: false, error: "The background worker sent no reply.", status: 0 });
      });
    } catch (error) {
      done({ ok: false, error: String((error && error.message) || error), status: 0 });
    }
  });
}

/**
 * Fire-and-forget interaction logging. The learning loop is a nice-to-have that must NEVER
 * block, delay, or surface an error in the main flow: we don't await it, and any failure
 * (server down, bad reply, thrown message) is swallowed. `event_type` is one of
 * 'search' | 'open' | 'inspect' | 'ask'.
 */
function logFeedback(params) {
  try {
    const p = Promise.resolve(ask({ type: "convsearch:feedback", params }));
    p.then(() => {}, () => {});
  } catch {
    /* never let telemetry break the UI */
  }
}

/* ------------------------------------------------------------------ */
/* DOM helpers (text only — never innerHTML)                          */
/* ------------------------------------------------------------------ */

function append(node, children) {
  if (children === null || children === undefined) return;
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
}

/**
 * el("div", { class, text, onclick, "aria-label", ... }, children)
 * Every attribute is set via setAttribute; text only ever via textContent. There is no path
 * here that injects HTML.
 */
function el(tag, opts, children) {
  const node = document.createElement(tag);
  if (opts) {
    for (const [key, value] of Object.entries(opts)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2).toLowerCase(), value);
      } else {
        node.setAttribute(key, value === true ? "" : String(value));
      }
    }
  }
  append(node, children);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* formatting                                                         */
/* ------------------------------------------------------------------ */

function plural(n, word) {
  return `${n} ${word}${Number(n) === 1 ? "" : "s"}`;
}

function fmtDate(iso) {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return String(iso).slice(0, 10);
  return new Date(t).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function timeAgo(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${plural(m, "minute")} ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${plural(h, "hour")} ago`;
  return `${plural(Math.round(h / 24), "day")} ago`;
}

function fmtScore(score) {
  return typeof score === "number" ? score.toFixed(3) : "";
}

function cap(text) {
  const s = String(text || "");
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

const CHANNEL_LABELS = {
  lexical: "keyword",
  semantic: "meaning",
  title: "title",
  reranker: "reranked",
  exact: "exact",
};
const channelLabel = (c) => CHANNEL_LABELS[c] || String(c);

function chatgptUrl(url) {
  return typeof url === "string" && url.startsWith("https://chatgpt.com/") ? url : null;
}

function openTab(url) {
  const safe = chatgptUrl(url);
  if (!safe) return false;
  if (chrome.tabs && chrome.tabs.create) chrome.tabs.create({ url: safe });
  else window.open(safe, "_blank", "noreferrer");
  return true;
}

/* ------------------------------------------------------------------ */
/* shared UI primitives                                               */
/* ------------------------------------------------------------------ */

function pill(text, variant) {
  return el("span", { class: variant ? `pill pill-${variant}` : "pill", text });
}

function chip(text) {
  return el("span", { class: "chip", text });
}

function spinner() {
  return el("div", { class: "spinner", role: "status", "aria-label": "Loading" });
}

function skeletonList(rows = 3) {
  return el(
    "div",
    { class: "output", "aria-hidden": "true" },
    Array.from({ length: rows }, () =>
      el("div", { class: "skeleton-card" }, [
        el("div", { class: "sk-line w-40" }),
        el("div", { class: "sk-line w-90" }),
        el("div", { class: "sk-line w-70" }),
      ])
    )
  );
}

/* Inline SVG built through the SVG namespace — createElement("svg") would make an inert HTML
 * element, so state/scenario icons go through createElementNS. Shapes are drawn from static,
 * hand-authored specs (never server data). */
const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

// Distinct glyph per state situation. Each entry is a list of [tag, attrs] shapes.
const STATE_ICONS = {
  empty: [["circle", { cx: 11, cy: 11, r: 6.5 }], ["path", { d: "m20 20-4.6-4.6" }]],
  offline: [
    ["path", { d: "M3 3l18 18" }],
    ["path", { d: "M8.6 6.9A6 6 0 0 1 19 11.5" }],
    ["path", { d: "M5.3 9.4A6 6 0 0 0 8 18h8" }],
  ],
  error: [["path", { d: "M12 3 2.6 20h18.8z" }], ["path", { d: "M12 9.5v4.2" }], ["path", { d: "M12 17.2h.01" }]],
  "empty-index": [
    ["ellipse", { cx: 12, cy: 6, rx: 7, ry: 2.6 }],
    ["path", { d: "M5 6v12c0 1.44 3.13 2.6 7 2.6s7-1.16 7-2.6V6" }],
    ["path", { d: "M5 12c0 1.44 3.13 2.6 7 2.6s7-1.16 7-2.6" }],
  ],
};

function stateIcon(kind) {
  const svg = svgEl("svg", {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": 1.6,
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  for (const [tag, attrs] of STATE_ICONS[kind] || STATE_ICONS.empty) svg.append(svgEl(tag, attrs));
  const span = el("span", { class: "state-icon", "aria-hidden": "true" });
  span.append(svg);
  return span;
}

/**
 * State block: loading / empty / offline / error / empty-index, each stating what happened and
 * the next step. `variant` (empty | offline | error | empty-index) picks the accent + default
 * icon; primary actions render as the styled .state-action CTA.
 * spec = { variant?, icon?, spinner?, title, body?, code?, actions?: [{ label, primary?, onClick, href? }] }
 */
function stateBlock(spec) {
  const variant = spec.variant;
  const wrap = el("div", { class: variant ? `state state--${variant}` : "state", role: spec.role || "status" });
  if (spec.spinner) wrap.append(spinner());
  else if (spec.icon || variant) wrap.append(stateIcon(spec.icon || variant));
  if (spec.title) wrap.append(el("strong", { class: "state-title", text: spec.title }));
  if (spec.body) wrap.append(el("p", { class: "state-body", text: spec.body }));
  if (spec.code) wrap.append(el("code", { text: spec.code }));
  if (spec.actions && spec.actions.length) {
    const row = el("div", { class: "result-actions" });
    for (const a of spec.actions) {
      if (a.href) {
        row.append(el("a", { class: a.primary ? "state-action" : "btn", href: a.href, target: "_blank", rel: "noreferrer", text: a.label }));
      } else {
        row.append(el("button", { type: "button", class: a.primary ? "state-action" : "btn", text: a.label, onclick: a.onClick }));
      }
    }
    wrap.append(row);
  }
  return wrap;
}

function loadingInto(container, rows) {
  clear(container);
  container.append(skeletonList(rows));
}

function errorState(err, onRetry) {
  const offline = !err || err.status === 0;
  return stateBlock({
    variant: offline ? "offline" : "error",
    title: offline ? "The local server isn't running." : "The server returned an error.",
    body: offline
      ? "convsearch talks to a local server on your machine, and it isn't responding. Start it with the launcher — or run the command below from the convsearch folder — then retry. Nothing loads until it's up."
      : (err && err.error) || "The request didn't succeed. Check the terminal running the server, then retry.",
    code: offline ? "convsearch serve --workspace ./workspace" : undefined,
    actions: onRetry ? [{ label: "Retry", primary: true, onClick: onRetry }] : [],
  });
}

/* Truncated quote with an inline expand affordance. */
function quoteBlock(text, className) {
  const full = String(text || "");
  const wrap = el("span", { class: className || "passage-quote" });
  if (full.length <= QUOTE_LIMIT) {
    wrap.textContent = full;
    return wrap;
  }
  let expanded = false;
  const body = document.createTextNode(full.slice(0, QUOTE_LIMIT) + "…");
  const btn = el("button", {
    type: "button",
    class: "expand-btn",
    text: "expand",
    "aria-expanded": "false",
  });
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    expanded = !expanded;
    body.textContent = expanded ? full : full.slice(0, QUOTE_LIMIT) + "…";
    btn.textContent = expanded ? "collapse" : "expand";
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
  });
  wrap.append(body, btn);
  return wrap;
}

/* ------------------------------------------------------------------ */
/* router / tabs                                                      */
/* ------------------------------------------------------------------ */

const VIEWS = ["home", "search", "plan", "tasks", "projects", "timeline", "memories", "review", "captures", "privacy", "status"];
// Display names for the .view-title in the main-column header. "search" reads as "Ask" to match
// the rail label and the unified ask/search surface.
const VIEW_TITLES = {
  home: "Home",
  search: "Ask",
  plan: "Plan",
  tasks: "Tasks",
  projects: "Projects",
  timeline: "Timeline",
  memories: "Memories",
  review: "Review",
  captures: "Captures",
  privacy: "Privacy",
  status: "Status",
};
const tabs = {};
const views = {};
let currentView = "home";
const loadedOnce = {};

for (const name of VIEWS) {
  tabs[name] = $(`tab-${name}`);
  views[name] = $(`view-${name}`);
}

const VIEW_ACTIVATORS = {
  home: () => loadHome(),
  search: () => focusSearch(),
  plan: () => $("plan-input").focus(),
  tasks: () => {
    if (!loadedOnce.tasks) {
      loadedOnce.tasks = true;
      loadTasks();
    }
  },
  memories: () => {
    if (!loadedOnce.memories) {
      loadedOnce.memories = true;
      loadMemories();
    }
  },
  projects: () => {
    if (!loadedOnce.projects) {
      loadedOnce.projects = true;
      loadProjects();
    }
  },
  timeline: () => $("timeline-query").focus(),
  review: () => {
    if (!loadedOnce.review) {
      loadedOnce.review = true;
      loadReview();
    }
  },
  captures: () => {
    if (!loadedOnce.captures) {
      loadedOnce.captures = true;
      loadCaptures();
    }
  },
  privacy: () => loadPrivacy(),
  status: () => loadStatus(),
};

function setView(name) {
  if (!VIEWS.includes(name)) return;
  currentView = name;
  for (const v of VIEWS) {
    const active = v === name;
    views[v].hidden = !active;
    tabs[v].setAttribute("aria-selected", active ? "true" : "false");
    tabs[v].tabIndex = active ? 0 : -1;
  }
  const title = $("view-title");
  if (title) title.textContent = VIEW_TITLES[name] || cap(name);
  const activate = VIEW_ACTIVATORS[name];
  if (activate) activate();
}

for (const name of VIEWS) {
  tabs[name].addEventListener("click", () => setView(name));
}

// Roving-tabindex arrow navigation across the VERTICAL icon rail: Up/Down move, Home/End jump.
document.querySelector(".rail").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp" && e.key !== "Home" && e.key !== "End") return;
  e.preventDefault();
  const idx = VIEWS.indexOf(currentView);
  let next = idx;
  if (e.key === "ArrowDown") next = (idx + 1) % VIEWS.length;
  else if (e.key === "ArrowUp") next = (idx - 1 + VIEWS.length) % VIEWS.length;
  else if (e.key === "Home") next = 0;
  else if (e.key === "End") next = VIEWS.length - 1;
  setView(VIEWS[next]);
  tabs[VIEWS[next]].focus();
});

/* ------------------------------------------------------------------ */
/* theme toggle (header)                                              */
/* ------------------------------------------------------------------ */

/*
 * Dark is the default. The chosen theme persists in chrome.storage.local under
 * "convsearch:theme" ("dark" | "light"); theme.css styles both via :root / :root[data-theme].
 * aria-pressed on #theme-toggle reflects the "light" state (the non-default the button engages).
 */
const THEME_KEY = "convsearch:theme";

function applyTheme(theme) {
  const t = theme === "light" ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  const btn = $("theme-toggle");
  if (btn) btn.setAttribute("aria-pressed", t === "light" ? "true" : "false");
}

function initTheme() {
  chrome.storage.local.get({ [THEME_KEY]: "dark" }, (stored) => {
    applyTheme((stored && stored[THEME_KEY]) || "dark");
  });
  const btn = $("theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      applyTheme(next);
      chrome.storage.local.set({ [THEME_KEY]: next });
    });
  }
}

/* ------------------------------------------------------------------ */
/* connection dot (header)                                            */
/* ------------------------------------------------------------------ */

function setConnDot(kind, title) {
  const dot = $("conn-dot");
  dot.className = "conn-dot" + (kind ? ` is-${kind}` : "");
  dot.setAttribute("title", title || "Server status");
}

async function refreshConn() {
  const res = await ask({ type: "convsearch:status" });
  if (res && res.online) {
    const h = res.health || {};
    setConnDot(h.indexing || h.stale_index ? "busy" : "online", `Connected · ${res.serverUrl}`);
  } else {
    setConnDot("offline", "Server offline");
  }
  return res;
}

/* ================================================================== */
/* SEARCH                                                             */
/* ================================================================== */

const searchInput = $("search-input");
const searchResultsEl = $("search-results");
const searchStatusEl = $("search-status");

let searchSeq = 0;
let searchDebounce = null;
let searchLastKey = null;
let searchSelected = -1;
let searchLastLoggedQuery = null;

function searchStatus(text, isError = false) {
  searchStatusEl.textContent = text;
  searchStatusEl.classList.toggle("is-error", isError);
}

/* ------------------------------------------------------------------ */
/* SEARCH · suggestions                                               */
/* ------------------------------------------------------------------ */

/*
 * When the query box is empty and focused, offer the user's recent and popular queries as
 * clickable, keyboard-navigable chips. Data comes from the SW (`convsearch:suggestions`); like
 * everything here, query strings are untrusted and rendered with textContent only. Failures are
 * silent — suggestions are an enhancement, never a blocker.
 */
const suggestBox = el("div", { class: "suggestions", role: "listbox", "aria-label": "Query suggestions", hidden: true });
searchResultsEl.parentNode.insertBefore(suggestBox, searchResultsEl);

let suggestData = null;
let suggestFetching = false;
let searchFocused = false;

function suggestHasItems() {
  return Boolean(suggestData && ((suggestData.recent && suggestData.recent.length) || (suggestData.popular && suggestData.popular.length)));
}

function suggestShouldShow() {
  return searchFocused && searchInput.value.trim() === "" && suggestHasItems();
}

function hideSuggestions() {
  suggestBox.hidden = true;
}

function pickSuggestion(text) {
  searchInput.value = text;
  hideSuggestions();
  focusSearch();
  runSearch({ force: true });
}

function renderSuggestions() {
  clear(suggestBox);
  if (!suggestShouldShow()) {
    suggestBox.hidden = true;
    return;
  }
  const addGroup = (label, items) => {
    const list = (items || []).filter((it) => (Array.isArray(it) ? it[0] : it));
    if (!list.length) return;
    const group = el("div", { class: "suggest-group" });
    group.append(el("div", { class: "suggest-label", text: label }));
    const chips = el("div", { class: "suggest-chips" });
    for (const it of list) {
      const q = String(Array.isArray(it) ? it[0] : it);
      const count = Array.isArray(it) && typeof it[1] === "number" ? it[1] : null;
      const btn = el("button", {
        type: "button",
        class: "suggest-chip",
        role: "option",
        tabindex: "-1",
        title: q,
      });
      btn.append(el("span", { class: "suggest-chip-text", text: q }));
      if (count != null) btn.append(el("span", { class: "suggest-chip-count", text: String(count) }));
      btn.addEventListener("click", () => pickSuggestion(q));
      chips.append(btn);
    }
    group.append(chips);
    suggestBox.append(group);
  };
  addGroup("Recent", suggestData.recent);
  addGroup("Popular", suggestData.popular);
  suggestBox.hidden = false;
  const first = suggestBox.querySelector(".suggest-chip");
  if (first) first.tabIndex = 0;
}

async function loadSuggestions({ force = false } = {}) {
  if (suggestData && !force) {
    renderSuggestions();
    return;
  }
  if (suggestFetching) return;
  suggestFetching = true;
  const res = await ask({ type: "convsearch:suggestions", params: { limit: 8 } });
  suggestFetching = false;
  if (res && res.ok) {
    const d = res.data || {};
    suggestData = {
      recent: Array.isArray(d.recent) ? d.recent : [],
      popular: Array.isArray(d.popular) ? d.popular : [],
    };
  } else if (!suggestData) {
    suggestData = { recent: [], popular: [] };
  }
  renderSuggestions();
}

function focusFirstSuggestion() {
  if (suggestBox.hidden) return false;
  const chip = suggestBox.querySelector(".suggest-chip");
  if (!chip) return false;
  for (const c of suggestBox.querySelectorAll(".suggest-chip")) c.tabIndex = -1;
  chip.tabIndex = 0;
  chip.focus();
  return true;
}

// Roving-tabindex arrow navigation across the suggestion chips.
suggestBox.addEventListener("keydown", (e) => {
  const chips = Array.from(suggestBox.querySelectorAll(".suggest-chip"));
  if (!chips.length) return;
  if (e.key === "Escape") {
    hideSuggestions();
    focusSearch();
    return;
  }
  const idx = chips.indexOf(document.activeElement);
  let next = idx;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") next = idx < 0 ? 0 : (idx + 1) % chips.length;
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = idx <= 0 ? chips.length - 1 : idx - 1;
  else if (e.key === "Home") next = 0;
  else if (e.key === "End") next = chips.length - 1;
  else return;
  e.preventDefault();
  chips.forEach((c, i) => (c.tabIndex = i === next ? 0 : -1));
  chips[next].focus();
});

searchInput.addEventListener("focus", () => {
  searchFocused = true;
  if (searchInput.value.trim() === "") loadSuggestions();
});
searchInput.addEventListener("blur", (e) => {
  if (suggestBox.contains(e.relatedTarget)) return; // focus moved into the chips
  searchFocused = false;
  hideSuggestions();
});
suggestBox.addEventListener("focusout", (e) => {
  if (suggestBox.contains(e.relatedTarget) || e.relatedTarget === searchInput) return;
  searchFocused = false;
  hideSuggestions();
});

function readSearchControls() {
  const levelInput = document.querySelector('input[name="level"]:checked');
  return {
    q: searchInput.value,
    level: levelInput ? levelInput.value : "conversation",
    profile: $("profile").value,
    branches: $("branches").checked,
    explain: $("explain").checked,
  };
}

let staleBadge = null;

function setSearchStale(stale) {
  const hasResults = searchResultsEl.querySelector(".result") !== null;
  searchResultsEl.classList.toggle("is-stale", Boolean(stale) && hasResults);
  // The opacity fade alone is easy to miss once the user has scrolled past the status line, so
  // pin an explicit "Updating…" chip above the (still-visible, dimmed) old results.
  if (stale && hasResults) {
    if (!staleBadge) {
      staleBadge = el("div", { class: "chip", text: "Updating…" });
      searchResultsEl.insertBefore(staleBadge, searchResultsEl.firstChild);
    }
  } else if (staleBadge) {
    staleBadge.remove();
    staleBadge = null;
  }
}

function aggregateChannels(passages) {
  const seen = new Set();
  for (const p of passages || []) {
    for (const c of p.channels || (p.explain && p.explain.channels) || []) seen.add(c);
  }
  return [...seen];
}

function normalizeResult(r, level) {
  if (level === "passage") {
    return {
      title: r.segment_title || `${cap(r.role || "message")}`,
      convId: r.conversation_id,
      url: null,
      score: r.score,
      date: r.created_at,
      reason: null,
      channels: r.channels || (r.explain && r.explain.channels) || [],
      passages: [r],
      metaExtra: null,
    };
  }
  if (level === "segment") {
    return {
      title: r.title || r.conversation_title || "(untitled segment)",
      sub: r.conversation_title,
      convId: r.conversation_id,
      url: null,
      score: r.score,
      date: (r.passages && r.passages[0] && r.passages[0].created_at) || null,
      reason: null,
      channels: aggregateChannels(r.passages),
      passages: r.passages || [],
      metaExtra: r.passages ? plural(r.passages.length, "passage") : null,
    };
  }
  return {
    title: r.title || "(untitled)",
    convId: r.conversation_id,
    url: chatgptUrl(r.url),
    score: r.score,
    date: r.updated_at || r.created_at,
    reason: r.reason,
    channels: aggregateChannels(r.passages),
    passages: r.passages || [],
    metaExtra: typeof r.distinct_message_count === "number" ? plural(r.distinct_message_count, "msg") : null,
  };
}

// Plain-language tooltips for each ranking channel — surfaced on the scorechart labels.
const CHANNEL_TIP = {
  keyword: "How closely the exact words in your query appear in this conversation.",
  meaning: "How close this conversation is in meaning, regardless of the exact wording.",
  title: "How well the conversation's title matches your query.",
  reranked: "A second-pass relevance model's judgement of how well this matches.",
  overall: "The combined, final ranking score across every signal.",
};

// Raw retrieval channel -> unified badge modifier (the .badge--* family in the design system).
const CHANNEL_BADGE = {
  lexical: "badge--keyword",
  exact: "badge--keyword",
  keyword: "badge--keyword",
  semantic: "badge--semantic",
  meaning: "badge--semantic",
  title: "badge--title",
  reranker: "",
  reranked: "",
};

function scorechartRow(label, value, { final = false } = {}) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  const row = el("div", { class: final ? "scorechart-row scorechart-row--final" : "scorechart-row" });
  row.append(el("span", { class: "scorechart-label", title: CHANNEL_TIP[label] || label, text: label }));
  const track = el("span", { class: "scorechart-track" });
  track.append(el("span", { class: "scorechart-fill", style: `width:${pct.toFixed(0)}%` }));
  row.append(track);
  row.append(el("span", { class: "scorechart-value", text: (Number(value) || 0).toFixed(2) }));
  return row;
}

/**
 * Score mini-chart: the four channels plus a weighted "overall" row that reads loudest. Replaces
 * the old .sbar / .explain rendering. Only channels the server actually scored are shown.
 */
function scorechart(explain) {
  const rows = [
    ["keyword", explain.lexical_score],
    ["meaning", explain.semantic_score],
    ["title", explain.title_score],
    ["reranked", explain.reranker_score],
  ].filter(([, v]) => typeof v === "number");
  const chart = el("div", { class: "scorechart" });
  for (const [l, v] of rows) chart.append(scorechartRow(l, v));
  const overall = explain.final_score != null ? explain.final_score : explain.fused_score;
  if (typeof overall === "number") chart.append(scorechartRow("overall", overall, { final: true }));
  return chart;
}

/**
 * The consolidated "why it matched" block. The reason sentence (with its "Ranked because" label)
 * shows by DEFAULT for any result that carries one; the detailed scorechart sits behind the
 * .why-toggle and auto-expands when the Explain checkbox is on.
 */
function buildWhy(n, explain, controls) {
  const hasReason = Boolean(n.reason);
  const hasDetail = Boolean(explain);
  if (!hasReason && !hasDetail) return null;
  const why = el("div", { class: "why" });
  if (hasReason) {
    why.append(el("span", { class: "why-label", text: "Ranked because" }));
    why.append(el("p", { class: "why-reason", text: n.reason }));
  }
  if (hasDetail) {
    const detail = el("div", { class: "why-detail" });
    detail.append(scorechart(explain));
    const expanded = Boolean(controls.explain);
    detail.hidden = !expanded;
    const toggle = el("button", {
      type: "button",
      class: "why-toggle",
      "aria-expanded": expanded ? "true" : "false",
      text: "Score breakdown",
    });
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = detail.hidden;
      detail.hidden = !open;
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    why.append(toggle, detail);
  }
  return why;
}

/** One passage, rendered as a nested step in the conversation > segment > passage hierarchy. */
function renderPassage(passage) {
  const wrap = el("div", { class: "result-hierarchy" });
  wrap.append(el("span", { class: "result-hierarchy-label", text: "Passage" }));
  const head = el("div", { class: "passage-head" });
  head.append(el("span", { class: "badge", text: passage.role || "message" }));
  const alternate = passage.is_primary_path === false;
  head.append(el("span", { class: alternate ? "badge badge--branch" : "badge badge--selected", text: alternate ? "alternate" : "on main path" }));
  if (passage.branch && passage.branch !== "main") {
    head.append(el("span", { class: "badge badge--branch", text: `branch ${passage.branch}` }));
  }
  wrap.append(head);
  wrap.append(quoteBlock(passage.text));
  return wrap;
}

function renderInspect(convId, region, toggleBtn) {
  clear(region);
  region.append(el("div", { class: "state" }, spinner()));
  ask({ type: "convsearch:conversation", id: convId }).then((res) => {
    clear(region);
    if (!res.ok) {
      region.append(errorState(res, () => renderInspect(convId, region, toggleBtn)));
      return;
    }
    const data = res.data || {};
    const head = el("div", { class: "result-head" });
    const url = chatgptUrl(data.url);
    head.append(el("strong", { class: "li-title", text: data.title || "Conversation" }));
    if (url) head.append(el("button", { type: "button", class: "btn btn-sm", text: "Open in ChatGPT", onclick: () => openTab(url) }));
    region.append(head);
    const msgs = data.messages || [];
    if (!msgs.length) {
      region.append(el("p", { class: "state-body", text: "No messages in this conversation." }));
      return;
    }
    for (const m of msgs) {
      const msg = el("div", { class: "msg" });
      const mh = el("div", { class: "msg-head" });
      mh.append(pill(m.role || "message", "role"));
      if (m.is_primary_path === false) mh.append(pill("alternate", "alt"));
      if (m.created_at) mh.append(el("span", { class: "tl-date", text: fmtDate(m.created_at) }));
      msg.append(mh);
      msg.append(quoteBlock(m.text, "msg-body"));
      region.append(msg);
    }
  });
}

/** Log that the user opened a result. Fire-and-forget; index is the 0-based list position. */
function logOpen(convId, index) {
  if (!convId) return;
  logFeedback({ event_type: "open", conversation_id: convId, position: index });
}

function renderResultCard(raw, index, controls) {
  const n = normalizeResult(raw, controls.level);
  const card = el("article", {
    class: "result",
    id: `result-${index}`,
    role: "option",
    "aria-selected": "false",
  });
  if (n.url) card.dataset.url = n.url;
  if (n.convId) card.dataset.conv = n.convId;
  card.dataset.index = String(index);

  // header — the title is a REAL focusable control (a link when there's a URL, else a button
  // that opens the inline inspector), no longer a tabindex="-1" span.
  const head = el("div", { class: "result-head" });
  const titleNode = n.url
    ? el("a", { class: "result-title", href: n.url, target: "_blank", rel: "noreferrer", text: n.title })
    : el("button", { type: "button", class: "result-title", text: n.title });
  if (n.url) {
    titleNode.addEventListener("click", () => logOpen(n.convId, index));
  } else {
    titleNode.addEventListener("click", (e) => {
      e.stopPropagation();
      if (card._inspect) card._inspect();
    });
  }
  head.append(titleNode);
  const metaBits = [fmtDate(n.date), n.metaExtra, fmtScore(n.score)].filter(Boolean);
  head.append(el("span", { class: "result-meta mono", text: metaBits.join(" · ") }));
  card.append(head);

  if (n.sub) card.append(el("div", { class: "result-sub", text: `in ${n.sub}` }));

  // unified match badges from the channels that fired
  if (n.channels && n.channels.length) {
    const badges = el("div", { class: "match-badges" });
    for (const c of n.channels) {
      const mod = CHANNEL_BADGE[c] != null ? CHANNEL_BADGE[c] : "";
      badges.append(el("span", { class: mod ? `badge ${mod}` : "badge", text: channelLabel(c) }));
    }
    card.append(badges);
  }

  // "Ranked because" — reason shown by default; scorechart behind the toggle (auto-open w/ Explain).
  const repExplain = (n.passages.find((p) => p && p.explain) || {}).explain || (raw && raw.explain) || null;
  const why = buildWhy(n, repExplain, controls);
  if (why) card.append(why);

  // passages, nested as the deepest step of the hierarchy
  for (const p of n.passages) card.append(renderPassage(p));

  // actions + inspect region
  const actions = el("div", { class: "result-actions" });
  if (n.url) actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Open", onclick: (e) => { e.stopPropagation(); logOpen(n.convId, index); openTab(n.url); } }));
  const inspectRegion = el("div", { class: "inspect", hidden: true });
  if (n.convId) {
    const inspectBtn = el("button", { type: "button", class: "btn btn-sm", text: "Inspect", "aria-expanded": "false" });
    inspectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = inspectRegion.hidden;
      inspectRegion.hidden = !open;
      inspectBtn.setAttribute("aria-expanded", open ? "true" : "false");
      inspectBtn.textContent = open ? "Hide" : "Inspect";
      if (open) {
        logFeedback({ event_type: "inspect", conversation_id: n.convId, position: index });
        renderInspect(n.convId, inspectRegion, inspectBtn);
      }
    });
    actions.append(inspectBtn);
    card.dataset.inspectable = "1";
    card._inspect = () => inspectBtn.click();
  }
  card.append(actions, inspectRegion);

  // click selects; open handled via keyboard/Open button
  card.addEventListener("mousedown", () => setSearchSelection(index, { scroll: false }));
  return card;
}

/* selection (keyboard driven combobox) */
function searchCards() {
  return Array.from(searchResultsEl.querySelectorAll(".result"));
}

function setSearchSelection(index, { scroll = true } = {}) {
  const cards = searchCards();
  if (!cards.length) {
    searchSelected = -1;
    searchInput.removeAttribute("aria-activedescendant");
    return;
  }
  const next = Math.max(-1, Math.min(index, cards.length - 1));
  searchSelected = next;
  cards.forEach((c, i) => {
    const on = i === next;
    c.classList.toggle("selected", on);
    c.setAttribute("aria-selected", on ? "true" : "false");
  });
  if (next < 0) {
    searchInput.removeAttribute("aria-activedescendant");
    return;
  }
  const active = cards[next];
  searchInput.setAttribute("aria-activedescendant", active.id);
  if (scroll) active.scrollIntoView({ block: "nearest" });
}

function moveSearchSelection(delta) {
  const cards = searchCards();
  if (!cards.length) return false;
  const from = searchSelected < 0 ? (delta > 0 ? -1 : 0) : searchSelected;
  let next = from + delta;
  if (next < 0) next = cards.length - 1;
  if (next >= cards.length) next = 0;
  setSearchSelection(next);
  return true;
}

function openSearchSelected() {
  const card = searchCards()[searchSelected];
  if (!card) return false;
  if (card.dataset.url) {
    logOpen(card.dataset.conv, Number(card.dataset.index));
    return openTab(card.dataset.url);
  }
  if (card._inspect) {
    card._inspect();
    return true;
  }
  return false;
}

function renderSearchIdle() {
  clear(searchResultsEl);
  clearAnswer(); // an emptied query drops the pinned answer too
  searchInput.setAttribute("aria-expanded", "false");
  setSearchSelection(-1);
  searchResultsEl.append(
    stateBlock({
      variant: "empty",
      title: "Ask or search your history.",
      body: "Type to search across every ChatGPT conversation convsearch has indexed — matching is by meaning, so describe what a conversation was about. Press Enter (or Ask) to also get a grounded answer with cited sources.",
    })
  );
}

async function runSearch({ force = false } = {}) {
  const controls = readSearchControls();
  const q = controls.q.trim();
  if (!q) {
    searchLastKey = null;
    searchStatus("");
    renderSearchIdle();
    return;
  }
  const key = JSON.stringify({ ...controls, q });
  if (!force && key === searchLastKey) return;
  searchLastKey = key;

  const seq = ++searchSeq;
  setSearchStale(true);
  searchStatus("Searching…");

  const res = await ask({
    type: "convsearch:search",
    params: {
      q,
      level: controls.level,
      explain: controls.explain ? 1 : 0,
      limit: 20,
      passages: 3,
      profile: controls.profile,
      branches: controls.branches ? 1 : 0,
    },
  });

  if (seq !== searchSeq) return; // a newer request already superseded this one

  if (!res.ok) {
    setSearchStale(false);
    searchStatus(res.status ? `Error ${res.status}` : "Server offline", true);
    clear(searchResultsEl);
    searchResultsEl.append(errorState(res, () => runSearch({ force: true })));
    refreshConn();
    return;
  }

  const data = res.data || {};
  const results = data.results || [];
  setSearchStale(false);
  searchStatus(`${plural(data.count != null ? data.count : results.length, "result")} · ${controls.level}`);

  // Log the settled query once results land — debounced by comparison so we record the query
  // the user actually rested on, not every keystroke along the way. Then refresh suggestions
  // so a just-searched query can surface as recent next time the box is empty.
  if (q !== searchLastLoggedQuery) {
    searchLastLoggedQuery = q;
    logFeedback({ event_type: "search", query: q });
    loadSuggestions({ force: true });
  }

  clear(searchResultsEl);
  if (!results.length) {
    searchInput.setAttribute("aria-expanded", "false");
    setSearchSelection(-1);
    searchResultsEl.append(
      stateBlock({
        variant: "empty",
        title: "No conversations matched that.",
        body: "Try describing the topic rather than quoting exact words, or switch the profile to Semantic. Recently captured conversations become searchable automatically once auto-indexing catches up.",
      })
    );
    return;
  }

  results.forEach((r, i) => searchResultsEl.append(renderResultCard(r, i, controls)));
  searchInput.setAttribute("aria-expanded", "true");
  // Leave nothing pre-selected so Enter runs the query (search + answer) rather than opening
  // the top result; ArrowDown selects a result, and Enter then opens the selected one.
  setSearchSelection(-1);
  refreshConn();
}

function scheduleSearch() {
  if (searchDebounce !== null) clearTimeout(searchDebounce);
  const q = searchInput.value.trim();
  if (!q) {
    searchSeq += 1;
    searchLastKey = null;
    searchStatus("");
    renderSearchIdle();
    return;
  }
  if (q.length < MIN_AUTO_QUERY) {
    setSearchStale(true);
    return;
  }
  setSearchStale(true);
  searchDebounce = setTimeout(() => {
    searchDebounce = null;
    runSearch();
  }, DEBOUNCE_MS);
}

// Explicit submit (Enter or the Ask button) runs BOTH: the fast search (results render as soon
// as they land) and the expensive grounded answer (pinned above, on its own spinner). The
// as-you-type path below only ever runs search — the answer is never fired on a keystroke.
$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (searchDebounce !== null) clearTimeout(searchDebounce);
  hideSuggestions();
  runSearch({ force: true });
  runAsk();
});
searchInput.addEventListener("input", scheduleSearch);
searchInput.addEventListener("input", () => {
  if (searchInput.value.trim() === "") {
    if (searchFocused) loadSuggestions();
  } else {
    hideSuggestions();
  }
});
for (const id of ["profile", "branches", "explain"]) {
  $(id).addEventListener("change", () => runSearch({ force: true }));
}
for (const radio of document.querySelectorAll('input[name="level"]')) {
  radio.addEventListener("change", () => runSearch({ force: true }));
}

/* ================================================================== */
/* ASK                                                                */
/* ================================================================== */

/*
 * The answer lives in a block pinned above the results on the unified Search tab. It is fired
 * only on explicit submit (Enter / the Ask button), never on every keystroke, because the real
 * model can take ~30s — so it must never block the fast search results, which render on their
 * own. `askSeq` guards it the same way `searchSeq` guards search: a stale reply can't clobber a
 * newer one, and clearing the slot bumps the sequence so an in-flight answer is dropped.
 */
const answerSlot = $("answer-slot");
let askSeq = 0;

/** Drop the pinned answer (and cancel any in-flight ask) — used when the query is emptied. */
function clearAnswer() {
  askSeq += 1;
  clear(answerSlot);
  answerSlot.hidden = true;
}

/**
 * A compact, non-blocking note for the answer slot (no-LLM / error). Unlike `stateBlock` this
 * does not take over the view — the search results below stay put. All text is untrusted and set
 * via `el`/textContent only.
 */
function answerNote(spec) {
  const note = el("div", { class: "answer-note", role: "status" });
  if (spec.title) note.append(el("strong", { class: "answer-note-title", text: spec.title }));
  if (spec.body) note.append(el("span", { class: "answer-note-body", text: spec.body }));
  if (spec.code) note.append(el("code", { text: spec.code }));
  if (spec.hint) note.append(el("span", { class: "answer-note-hint", text: spec.hint }));
  if (spec.onRetry) {
    const row = el("div", { class: "result-actions" });
    row.append(el("button", { type: "button", class: "btn btn-sm", text: "Try again", onclick: spec.onRetry }));
    note.append(row);
  }
  return note;
}

function renderAnswerText(answer, sourcesById) {
  const container = el("div", { class: "answer" });
  const parts = String(answer || "").split(/(\[\d+\])/g);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const num = m[1];
      const btn = el("button", {
        type: "button",
        class: "cite",
        text: num,
        "aria-label": `Jump to source ${num}`,
      });
      btn.addEventListener("click", () => {
        const target = sourcesById[num];
        if (!target) return;
        target.scrollIntoView({ block: "center", behavior: "smooth" });
        target.classList.add("flash");
        setTimeout(() => target.classList.remove("flash"), 1200);
      });
      container.append(btn);
    } else if (part) {
      container.append(document.createTextNode(part));
    }
  }
  return container;
}

async function runAsk() {
  const q = searchInput.value.trim();
  if (!q) return;
  const seq = ++askSeq;
  $("ask-submit").disabled = true;
  // The answer block gets its OWN spinner so it can lag behind the results without blocking them.
  answerSlot.hidden = false;
  clear(answerSlot);
  answerSlot.append(stateBlock({ spinner: true, title: "Answering…", body: "Synthesising a grounded answer from your history — this can take a moment. Results are ready below." }));

  const res = await ask({ type: "convsearch:ask", params: { q, limit: 8, passages: 3 } });
  if (seq !== askSeq) return; // superseded by a newer ask, or the slot was cleared
  $("ask-submit").disabled = false;
  clear(answerSlot);

  if (!res.ok) {
    // A missing model must NOT wipe the results: show a small non-blocking note in the slot.
    if (res.status === 503) {
      answerSlot.append(
        answerNote({
          title: "Answer needs a local model",
          body: "Run:",
          code: "ollama serve\nollama pull gemma3:1b",
          hint: "(or set --backend anthropic with a key)",
          onRetry: runAsk,
        })
      );
      return;
    }
    answerSlot.append(
      answerNote({
        title: "Couldn't get an answer.",
        body: res.status ? `The server returned an error (${res.status}). The results below are still current.` : "The server is unreachable. The results below may be stale.",
        onRetry: runAsk,
      })
    );
    return;
  }

  // An answer came back — record the asked question for the learning loop (fire-and-forget).
  logFeedback({ event_type: "ask", query: q });

  const data = res.data || {};
  const sources = data.sources || [];
  const sourcesById = {};

  // Build source cards first so citation markers can link to them.
  const sourcesWrap = el("div", { class: "output", style: "padding:0;gap:8px" });
  for (const s of sources) {
    const idx = String(s.index);
    const card = el("div", { class: "source", id: `source-${idx}` });
    const sh = el("div", { class: "source-head" });
    sh.append(el("span", { class: "source-index", text: `[${idx}]` }));
    sh.append(el("strong", { class: "li-title", text: s.title || "(untitled)" }));
    card.append(sh);
    const meta = [fmtDate(s.date), s.role].filter(Boolean).join(" · ");
    if (meta) card.append(el("div", { class: "li-meta", text: meta }));
    if (s.quote) card.append(quoteBlock(s.quote, "source-quote"));
    sourcesById[idx] = card;
    sourcesWrap.append(card);
  }

  const block = el("div", { class: "answer-block" });
  block.append(renderAnswerText(data.answer, sourcesById));
  if (data.backend || data.model) {
    block.append(el("p", { class: "answer-meta", text: `answered by ${[data.backend, data.model].filter(Boolean).join(":")}` }));
  }
  if (sources.length) {
    block.append(el("h2", { class: "sources-title", text: `Sources (${sources.length})` }));
    block.append(sourcesWrap);
  } else {
    block.append(el("p", { class: "answer-meta", text: "No sources were cited for this answer." }));
  }
  answerSlot.append(block);
}

/* ================================================================== */
/* PLAN                                                               */
/* ================================================================== */

/*
 * The planner asks the server to decide how to answer a question, then reports both the
 * grounded answer and the reasoning trail (intent -> steps -> tool calls -> findings). Every
 * string it returns is untrusted server text and is rendered with textContent / `el` only.
 */
const planInput = $("plan-input");
const planOutput = $("plan-output");
let planSeq = 0;

/** A collapsible "How it answered" section holding steps, calls and findings. */
function planHowSection(steps, calls, findings) {
  const details = el("details", { class: "plan-how" });
  details.append(el("summary", { class: "plan-how-summary", text: "How it answered" }));

  if (steps.length) {
    const block = el("div", { class: "dash-section" });
    block.append(el("h3", { class: "section-head" }, ["Steps", el("span", { class: "section-count", text: String(steps.length) })]));
    const ol = el("ol", { class: "plan-steps" });
    for (const s of steps) {
      const li = el("li", { class: "plan-step" });
      const head = el("div", { class: "plan-step-head" });
      if (s.order !== null && s.order !== undefined) head.append(el("span", { class: "plan-step-order mono", text: `${s.order}.` }));
      if (s.tool) head.append(pill(String(s.tool), "role"));
      li.append(head);
      if (s.rationale) li.append(el("div", { class: "plan-step-rationale", text: String(s.rationale) }));
      ol.append(li);
    }
    block.append(ol);
    details.append(block);
  }

  if (calls.length) {
    const block = el("div", { class: "dash-section" });
    block.append(el("h3", { class: "section-head" }, ["Tool calls", el("span", { class: "section-count", text: String(calls.length) })]));
    for (const c of calls) {
      const row = el("div", { class: "plan-call" });
      const head = el("div", { class: "plan-call-head" });
      if (c.tool) head.append(pill(String(c.tool), "accent"));
      if (c.result_count !== null && c.result_count !== undefined) {
        head.append(el("span", { class: "li-meta mono", text: plural(Number(c.result_count) || 0, "result") }));
      }
      row.append(head);
      if (c.result_summary) row.append(el("div", { class: "plan-call-summary", text: String(c.result_summary) }));
      block.append(row);
    }
    details.append(block);
  }

  if (findings.length) {
    const block = el("div", { class: "dash-section" });
    block.append(el("h3", { class: "section-head" }, ["Findings", el("span", { class: "section-count", text: String(findings.length) })]));
    block.append(el("div", { class: "dash-block" }, statementList(findings)));
    details.append(block);
  }

  return details;
}

async function runPlan() {
  const q = planInput.value.trim();
  if (!q) return;
  const seq = ++planSeq;
  $("plan-submit").disabled = true;
  clear(planOutput);
  planOutput.append(stateBlock({ spinner: true, title: "Planning…", body: "Working out how to answer, then gathering grounded evidence from your history." }));

  const res = await ask({ type: "convsearch:plan", params: { q } });
  if (seq !== planSeq) return;
  $("plan-submit").disabled = false;
  clear(planOutput);

  if (!res.ok) {
    if (res.status === 503) {
      planOutput.append(
        stateBlock({
          variant: "error",
          title: "No language model is available.",
          body: "The planner needs a local or cloud model to synthesise a grounded answer. Run a local one with Ollama, then try again:",
          code: "ollama serve\nollama pull llama3.2:1b",
          actions: [{ label: "Try again", primary: true, onClick: runPlan }],
        })
      );
      return;
    }
    planOutput.append(errorState(res, runPlan));
    return;
  }

  const data = res.data || {};
  const answer = String(data.answer || "").trim();
  const steps = Array.isArray(data.steps) ? data.steps : [];
  const calls = Array.isArray(data.calls) ? data.calls : [];
  const findings = (Array.isArray(data.findings) ? data.findings : []).filter((f) => typeof f === "string" && f.trim());

  if (!answer && !steps.length && !calls.length && !findings.length) {
    planOutput.append(
      stateBlock({
        variant: "empty",
        title: "The planner had nothing to add.",
        body: "It couldn't ground an answer to that question from your captured history. Try rephrasing, or capture more of ChatGPT first.",
        actions: [{ label: "Try again", primary: true, onClick: runPlan }],
      })
    );
    return;
  }

  // Grounded answer, prominent at the top. When the planner gathered a trail but produced no
  // answer text, say so explicitly with a next action rather than rendering a blank/placeholder
  // line — the reasoning trail (if any) still renders below.
  if (answer) {
    planOutput.append(el("div", { class: "answer plan-answer", text: answer }));
  } else {
    planOutput.append(
      stateBlock({
        variant: "empty",
        title: "No grounded answer for that.",
        body: "The planner gathered evidence but couldn't ground a direct answer. Try rephrasing, ask on the Ask & Search tab instead, or capture more of ChatGPT first.",
        actions: [
          { label: "Try again", primary: true, onClick: runPlan },
          { label: "Switch to Ask", onClick: () => setView("search") },
        ],
      })
    );
  }

  // Detected intent.
  if (data.intent) {
    planOutput.append(el("div", { class: "plan-intent" }, [el("span", { class: "plan-intent-label", text: "Intent" }), pill(String(data.intent), "accent")]));
  }

  // Collapsible reasoning trail.
  if (steps.length || calls.length || findings.length) {
    planOutput.append(planHowSection(steps, calls, findings));
  }
}

$("plan-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runPlan();
});
planInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    runPlan();
  }
});

/* ================================================================== */
/* MEMORIES                                                           */
/* ================================================================== */

const memOutput = $("memories-output");
let memDebounce = null;
let memSeq = 0;

function readMemFilters() {
  return {
    q: $("mem-query").value.trim(),
    kind: $("mem-kind").value,
    status: $("mem-status").value,
    project: $("mem-project").value.trim(),
  };
}

/** Keep kind/status selects in sync with whatever the current result set contains. */
function syncSelect(select, values) {
  const current = select.value;
  const wanted = new Set(values.filter(Boolean));
  const existing = new Set(Array.from(select.options).map((o) => o.value));
  for (const v of wanted) {
    if (!existing.has(v)) select.append(el("option", { value: v, text: v }));
  }
  select.value = current;
}

function memStatusVariant(status) {
  const s = String(status || "").toLowerCase();
  if (s === "active" || s === "confirmed") return "ok";
  if (s === "superseded" || s === "deprecated") return "warn";
  if (s === "conflicted" || s === "rejected") return "bad";
  return "accent";
}

async function loadMemories() {
  const seq = ++memSeq;
  const f = readMemFilters();
  loadingInto(memOutput, 4);

  const res = await ask({ type: "convsearch:memories", params: { q: f.q, kind: f.kind, status: f.status, project: f.project, limit: 60 } });
  if (seq !== memSeq) return;
  clear(memOutput);

  if (!res.ok) {
    memOutput.append(errorState(res, loadMemories));
    return;
  }

  const data = res.data || {};
  const memories = data.memories || [];
  syncSelect($("mem-kind"), memories.map((m) => m.kind));
  syncSelect($("mem-status"), memories.map((m) => m.status));

  if (!memories.length) {
    memOutput.append(
      stateBlock({
        variant: "empty",
        title: "No memories yet.",
        body: "Memories are facts, preferences and decisions the server distills from your conversations. Browse and capture more of ChatGPT, or clear the filters above.",
      })
    );
    return;
  }

  memOutput.append(el("h2", { class: "section-head" }, ["Memories", el("span", { class: "section-count", text: String(data.count != null ? data.count : memories.length) })]));
  for (const m of memories) {
    const item = el("div", { class: "list-item", role: "button", tabindex: "0" });
    item.append(el("div", { class: "li-title", text: m.statement || "(no statement)" }));
    const badges = el("div", { class: "badges" });
    if (m.kind) badges.append(pill(m.kind, "role"));
    if (m.status) badges.append(pill(m.status, memStatusVariant(m.status)));
    if (m.project) badges.append(pill(m.project, "accent"));
    item.append(badges);
    const meta = el("div", { class: "li-meta" });
    if (typeof m.confidence === "number") meta.append(el("span", { class: "mono", text: `confidence ${(m.confidence * 100).toFixed(0)}%` }));
    if (m.conversation_title) meta.append(el("span", { text: m.conversation_title }));
    if (m.created_at) meta.append(el("span", { text: fmtDate(m.created_at) }));
    item.append(meta);
    const open = () => openMemory(m.memory_id);
    item.addEventListener("click", open);
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });
    memOutput.append(item);
  }
}

async function openMemory(id) {
  loadingInto(memOutput, 2);
  const res = await ask({ type: "convsearch:memory", id });
  clear(memOutput);
  if (!res.ok) {
    memOutput.append(errorState(res, () => openMemory(id)));
    return;
  }
  const m = res.data || {};
  memOutput.append(el("button", { type: "button", class: "btn btn-ghost btn-sm back-btn", text: "‹ All memories", onclick: loadMemories }));
  memOutput.append(el("div", { class: "detail-statement", text: m.statement || "(no statement)" }));

  const badges = el("div", { class: "badges" });
  if (m.kind) badges.append(pill(m.kind, "role"));
  if (m.status) badges.append(pill(m.status, memStatusVariant(m.status)));
  if (m.project) badges.append(pill(m.project, "accent"));
  if (typeof m.confidence === "number") badges.append(el("span", { class: "mono li-meta", text: `confidence ${(m.confidence * 100).toFixed(0)}%` }));
  memOutput.append(badges);

  if (m.conversation_title || m.conversation_id) {
    const row = el("div", { class: "li-meta" });
    row.append(document.createTextNode(`From: ${m.conversation_title || "(untitled)"}${m.created_at ? ` · ${fmtDate(m.created_at)}` : ""}`));
    if (m.conversation_id) {
      const convId = m.conversation_id;
      const inspectRegion = el("div", { class: "inspect", hidden: true });
      const inspectBtn = el("button", { type: "button", class: "btn btn-ghost btn-sm", text: "View conversation", "aria-expanded": "false" });
      inspectBtn.addEventListener("click", () => {
        const open = inspectRegion.hidden;
        inspectRegion.hidden = !open;
        inspectBtn.setAttribute("aria-expanded", open ? "true" : "false");
        inspectBtn.textContent = open ? "Hide conversation" : "View conversation";
        if (open) renderInspect(convId, inspectRegion, inspectBtn);
      });
      row.append(inspectBtn);
      memOutput.append(row, inspectRegion);
    } else {
      memOutput.append(row);
    }
  }

  const evidence = m.evidence || [];
  const evidenceBlock = el("div", { class: "dash-section" });
  evidenceBlock.append(el("h2", { class: "section-head" }, ["Evidence", el("span", { class: "section-count", text: String(evidence.length) })]));
  if (evidence.length) {
    for (const e of evidence) evidenceBlock.append(el("div", { class: "dash-block" }, quoteBlock(e.quote, "source-quote")));
  } else {
    evidenceBlock.append(el("p", { class: "state-body", text: "No evidence recorded for this memory yet." }));
  }
  memOutput.append(evidenceBlock);

  const relations = m.relations || [];
  const relationsBlock = el("div", { class: "dash-section" });
  relationsBlock.append(el("h2", { class: "section-head" }, ["Relations", el("span", { class: "section-count", text: String(relations.length) })]));
  if (relations.length) {
    for (const r of relations) {
      const row = el("div", { class: "dash-block" });
      const head = el("div", { class: "badges" });
      head.append(pill(`${r.direction || ""} ${r.relation || ""}`.trim(), r.relation === "conflicts" ? "bad" : "warn"));
      row.append(head);
      row.append(el("div", { class: "source-quote", text: r.other_statement || r.other_memory_id || "" }));
      relationsBlock.append(row);
    }
  } else {
    relationsBlock.append(el("p", { class: "state-body", text: "No related memories yet." }));
  }
  memOutput.append(relationsBlock);

  const history = m.status_history || m.history || [];
  const historyBlock = el("div", { class: "dash-section" });
  historyBlock.append(el("h2", { class: "section-head" }, ["Status history", el("span", { class: "section-count", text: String(history.length) })]));
  if (history.length) {
    const tl = el("div", { class: "timeline" });
    for (const h of history) {
      const item = el("div", { class: "tl-item" });
      item.append(el("div", {}, [pill(h.old_status || "—", "warn"), " → ", pill(h.new_status || "—", memStatusVariant(h.new_status))]));
      if (h.reason) item.append(el("div", { class: "source-quote", text: h.reason }));
      if (h.changed_at) item.append(el("div", { class: "tl-date", text: fmtDate(h.changed_at) }));
      tl.append(item);
    }
    historyBlock.append(tl);
  } else {
    historyBlock.append(el("p", { class: "state-body", text: "No status changes recorded yet." }));
  }
  memOutput.append(historyBlock);
}

$("mem-query").addEventListener("input", () => {
  if (memDebounce !== null) clearTimeout(memDebounce);
  memDebounce = setTimeout(loadMemories, DEBOUNCE_MS);
});
for (const id of ["mem-kind", "mem-status"]) $(id).addEventListener("change", loadMemories);
$("mem-project").addEventListener("input", () => {
  if (memDebounce !== null) clearTimeout(memDebounce);
  memDebounce = setTimeout(loadMemories, DEBOUNCE_MS);
});

/* ================================================================== */
/* REVIEW · memory review queue                                       */
/* ================================================================== */

const reviewOutput = $("review-output");
let reviewSeq = 0;
let reviewDebounce = null;

function readReviewFilters() {
  return {
    kind: $("review-kind").value,
    project: $("review-project").value.trim(),
    includeReviewed: $("review-include-reviewed").checked,
  };
}

/** Same shape as `dashSection`'s conflicts-with rendering: what it conflicts with, not just
 * that it conflicts. */
function reviewConflictsBlock(conflicts) {
  if (!conflicts || !conflicts.length) return null;
  const body = el("div", { class: "dash-section" });
  for (const c of conflicts) {
    const row = el("div", { class: "dash-block" });
    const head = el("div", { class: "badges" });
    if (c.status) head.append(pill(c.status, memStatusVariant(c.status)));
    row.append(head);
    row.append(el("div", { class: "source-quote", text: c.statement || `memory ${c.memory_id}` }));
    if (c.reason) row.append(el("p", { class: "state-body", text: c.reason }));
    body.append(row);
  }
  return dashSection("Conflicts with", conflicts.length, body);
}

function reviewSupersededByBlock(superseded) {
  if (!superseded || !superseded.length) return null;
  return dashSection("Superseded by", superseded.length, el("div", { class: "dash-block" }, statementList(superseded)));
}

/**
 * One review-queue card, built fresh from a review item. Confirm/Invalidate/Pin mutate that
 * one memory server-side and return the updated item; on success we rebuild this exact card
 * from the response and swap it in, rather than refetching the whole queue. On failure the
 * server's error is shown inline and the item is left exactly as it was — no optimistic state.
 */
function buildReviewCard(item) {
  const card = el("div", { class: "list-item" });
  card.append(el("div", { class: "li-title", text: item.statement || "(no statement)" }));

  const badges = el("div", { class: "badges" });
  if (item.kind) badges.append(pill(item.kind, "role"));
  if (item.status) badges.append(pill(item.status, memStatusVariant(item.status)));
  if (item.project) badges.append(pill(item.project, "accent"));
  if (item.pinned) badges.append(pill("pinned", "sel"));
  card.append(badges);

  // The whole point of this queue: why this item needs a look, stated plainly and up front.
  if (item.review_reason) {
    card.append(el("p", { class: "state-body review-reason", text: item.review_reason }));
  }

  const meta = el("div", { class: "li-meta" });
  if (item.conversation_title) meta.append(el("span", { text: item.conversation_title }));
  meta.append(el("span", { text: item.created_at ? fmtDate(item.created_at) : "date unknown" }));
  if (typeof item.confidence === "number") meta.append(el("span", { class: "mono", text: `confidence ${(item.confidence * 100).toFixed(0)}%` }));
  if (item.reviewed_at) meta.append(el("span", { text: `reviewed ${fmtDate(item.reviewed_at)}` }));
  card.append(meta);

  // Evidence — same .source/.source-quote drawer Tasks and Timeline use. Review evidence has
  // no per-entry conversation/role/timestamp (it's always this item's own conversation), so
  // evidenceDrawer renders just the quotes, which is exactly right here.
  attachEvidenceToggle(card, item.evidence, false);

  const conflictsBlock = reviewConflictsBlock(item.conflicts);
  if (conflictsBlock) card.append(conflictsBlock);
  const supersededBlock = reviewSupersededByBlock(item.superseded_by);
  if (supersededBlock) card.append(supersededBlock);

  const errNote = el("p", { class: "state-body is-error", hidden: true });

  const actions = el("div", { class: "result-actions" });
  const confirmBtn = el("button", { type: "button", class: "btn btn-sm", text: "Confirm" });
  const pinBtn = el("button", { type: "button", class: "btn btn-sm", text: item.pinned ? "Unpin" : "Pin" });
  const invalidateBtn = el("button", { type: "button", class: "btn btn-sm", text: "Invalidate" });

  const setBusy = (busy) => {
    confirmBtn.disabled = busy;
    pinBtn.disabled = busy;
    invalidateBtn.disabled = busy;
  };

  async function mutate(type, params) {
    errNote.hidden = true;
    setBusy(true);
    const res = await ask({ type, id: item.memory_id, params: params || {} });
    if (!res.ok) {
      setBusy(false);
      errNote.textContent = (res && res.error) || "The request didn't succeed. The item is unchanged.";
      errNote.hidden = false;
      return;
    }
    card.replaceWith(buildReviewCard(res.data || item));
  }

  confirmBtn.addEventListener("click", () => mutate("convsearch:memoryConfirm", {}));
  pinBtn.addEventListener("click", () => mutate("convsearch:memoryPin", { pinned: !item.pinned }));

  // Invalidate marks a memory wrong — require an inline "are you sure?" before it fires.
  let invalidateArmed = false;
  invalidateBtn.addEventListener("click", () => {
    if (!invalidateArmed) {
      invalidateArmed = true;
      invalidateBtn.textContent = "Confirm invalidate?";
      return;
    }
    mutate("convsearch:memoryInvalidate", {});
  });

  actions.append(confirmBtn, pinBtn, invalidateBtn);
  card.append(actions, errNote);

  return card;
}

async function loadReview() {
  const seq = ++reviewSeq;
  const f = readReviewFilters();
  loadingInto(reviewOutput, 4);

  const res = await ask({
    type: "convsearch:memoriesReview",
    params: { limit: 100, kind: f.kind, project: f.project, include_reviewed: f.includeReviewed ? 1 : undefined },
  });
  if (seq !== reviewSeq) return;
  clear(reviewOutput);

  if (!res.ok) {
    reviewOutput.append(errorState(res, loadReview));
    return;
  }

  const data = res.data || {};
  const items = data.items || [];
  syncSelect($("review-kind"), items.map((i) => i.kind));

  const head = el("div", { class: "li-meta" });
  head.append(
    el("span", {
      class: "mono",
      text: `${plural(data.total_pending || 0, "pending")} · ${plural(data.total_pinned || 0, "pinned")} · ${plural(data.total_contested || 0, "contested")} · ${plural(data.total_invalidated || 0, "invalidated")}`,
    })
  );
  reviewOutput.append(head);

  if (!items.length) {
    reviewOutput.append(
      stateBlock({
        title: "Nothing needs review.",
        body:
          f.includeReviewed || f.kind || f.project
            ? "Nothing matches these filters right now — try clearing them."
            : "Every captured memory is confirmed, pinned, or hasn't raised a flag yet. That's good news, not an error.",
      })
    );
    return;
  }

  reviewOutput.append(el("h2", { class: "section-head" }, ["Review queue", el("span", { class: "section-count", text: String(data.count != null ? data.count : items.length) })]));
  for (const item of items) reviewOutput.append(buildReviewCard(item));
}

$("review-kind").addEventListener("change", loadReview);
$("review-include-reviewed").addEventListener("change", loadReview);
$("review-project").addEventListener("input", () => {
  if (reviewDebounce !== null) clearTimeout(reviewDebounce);
  reviewDebounce = setTimeout(loadReview, DEBOUNCE_MS);
});

/* ================================================================== */
/* TASKS                                                              */
/* ================================================================== */

const tasksOutput = $("tasks-output");
let tasksSeq = 0;

function readTaskFilters() {
  const stateInput = document.querySelector('input[name="task-state"]:checked');
  return {
    state: (stateInput && stateInput.value) || "open",
    project: $("task-project").value,
    expandEvidence: $("task-evidence").checked,
  };
}

/** Evidence drawer shared by Tasks and Timeline: verbatim quote, conversation, role, timestamp. */
function evidenceDrawer(evidence) {
  const wrap = el("div", { class: "dash-block" });
  if (!evidence || !evidence.length) {
    wrap.append(el("p", { class: "state-body", text: "No evidence recorded for this yet." }));
    return wrap;
  }
  for (const e of evidence) {
    const card = el("div", { class: "source" });
    const head = el("div", { class: "source-head" });
    if (e.conversation_title) head.append(el("strong", { class: "li-title", text: e.conversation_title }));
    card.append(head);
    const meta = [e.role, e.timestamp ? fmtDate(e.timestamp) : "date unknown"].filter(Boolean).join(" · ");
    if (meta) card.append(el("div", { class: "li-meta", text: meta }));
    card.append(quoteBlock(e.quote, "source-quote"));
    wrap.append(card);
  }
  return wrap;
}

/**
 * Expandable evidence affordance.
 *
 * IMPORTANT: `has_evidence` from the server was measured against the running instance and found
 * to just mirror "was `evidence=1` requested", not an independent truthful signal (a
 * memory_id came back `has_evidence: false` with no `evidence` param and `true` for the exact
 * same memory_id with `evidence=1`, every time). That's the same "empty evidence[] misread as
 * no evidence" bug the task brief warned about, except it has leaked into the field meant to
 * guard against it. Since we don't own the server, we route around it here: Tasks and Timeline
 * both always request `evidence=1`, so `evidence` is always the real list for the item in hand,
 * and "no evidence" is decided from that list being empty — never from the unreliable flag.
 */
function attachEvidenceToggle(container, evidence, startExpanded) {
  if (!evidence || !evidence.length) {
    container.append(el("p", { class: "li-meta", text: "No evidence recorded." }));
    return;
  }
  const region = el("div", { class: "inspect" }, evidenceDrawer(evidence));
  region.hidden = !startExpanded;
  const btn = el("button", {
    type: "button",
    class: "btn btn-sm",
    text: startExpanded ? "Hide evidence" : "Show evidence",
    "aria-expanded": startExpanded ? "true" : "false",
  });
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = region.hidden;
    region.hidden = !open;
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.textContent = open ? "Hide evidence" : "Show evidence";
  });
  container.append(btn, region);
}

function taskStateVariant(state) {
  const s = String(state || "").toLowerCase();
  if (s === "completed") return "ok";
  if (s === "open") return "accent";
  return "role";
}

async function loadTasks() {
  const seq = ++tasksSeq;
  const f = readTaskFilters();
  loadingInto(tasksOutput, 4);

  // Always request evidence — see attachEvidenceToggle's note on why has_evidence alone isn't
  // trustworthy here. Item counts are small (single digits to a few dozen), so this is cheap.
  const res = await ask({
    type: "convsearch:tasks",
    params: { state: f.state, project: f.project, limit: 100, evidence: 1 },
  });
  if (seq !== tasksSeq) return;
  clear(tasksOutput);

  if (!res.ok) {
    // A bad `state` is a 400 with a specific server message — surface it verbatim, not a generic failure.
    tasksOutput.append(errorState(res, loadTasks));
    return;
  }

  const data = res.data || {};
  const items = data.items || [];
  syncSelect($("task-project"), data.projects || []);

  // Mutable so a Complete/Reopen click can adjust these in place without refetching the whole
  // list — the header span is updated from `taskCounts`, not re-read from the stale `data`.
  const taskCounts = { open: data.total_open || 0, completed: data.total_completed || 0 };
  const countsLabel = el("span", { class: "mono" });
  const renderCounts = () => {
    countsLabel.textContent = `${plural(taskCounts.open, "open task")} · ${plural(taskCounts.completed, "completed task")}`;
  };
  renderCounts();
  const head = el("div", { class: "li-meta" }, [countsLabel]);
  tasksOutput.append(head);

  /** Called after a successful complete/reopen with the state before and after the transition,
   * so the header counts track reality without a full re-fetch. */
  function onTaskStateChange(prevState, nextState) {
    if (prevState === nextState) return;
    if (prevState === "open") taskCounts.open = Math.max(0, taskCounts.open - 1);
    if (prevState === "completed") taskCounts.completed = Math.max(0, taskCounts.completed - 1);
    if (nextState === "open") taskCounts.open += 1;
    if (nextState === "completed") taskCounts.completed += 1;
    renderCounts();
  }

  if (!items.length) {
    if (f.state === "open" && (data.total_open || 0) === 0) {
      tasksOutput.append(
        stateBlock({
          title: "No open tasks — you're caught up.",
          body: "Nothing outstanding matches this filter right now. Switch to \"All\" to see completed and past tasks.",
        })
      );
    } else if ((data.total_open || 0) === 0 && (data.total_completed || 0) === 0) {
      tasksOutput.append(
        stateBlock({
          title: "No conversations indexed yet.",
          body: "Tasks are distilled from your ChatGPT history as it's captured and indexed. Browse chatgpt.com with the server running — your conversations are captured and indexed automatically. Optional: import a ChatGPT export to backfill your existing history.",
          actions: [{ label: "Go to Status", onClick: () => setView("status") }],
        })
      );
    } else {
      tasksOutput.append(stateBlock({ title: "No tasks match these filters.", body: "Try a different project or state." }));
    }
    return;
  }

  for (const t of items) tasksOutput.append(buildTaskCard(t, f.expandEvidence, onTaskStateChange));
}

/**
 * One task-list row, built fresh from a task item. Complete/Reopen mutate that one task
 * server-side and return the updated item; on success we rebuild this exact card from the
 * response and swap it in — same pattern as buildReviewCard. On failure the server's message
 * is shown inline and the row is left exactly as it was, no optimistic state.
 *
 * Filter-mismatch decision: completing a task while the "open" filter is active would make the
 * row stop matching that filter. Rather than yank the row out from under the user the instant
 * they click (surprising, and indistinguishable from an error), we leave it in place — re-rendered
 * with its new "completed" pill — until the next `loadTasks()` (filter change, or manual refresh)
 * naturally drops it. Same tradeoff the review queue already makes for its own filters.
 */
function buildTaskCard(t, expandEvidence, onStateChange) {
  const item = el("div", { class: "list-item" });
  item.append(el("div", { class: "li-title", text: t.statement || "(no statement)" }));
  const badges = el("div", { class: "badges" });
  if (t.project) badges.append(pill(t.project, "accent"));
  if (t.task_state) badges.append(pill(t.task_state, taskStateVariant(t.task_state)));
  if (t.status) badges.append(pill(t.status, memStatusVariant(t.status)));
  item.append(badges);

  const meta = el("div", { class: "li-meta" });
  if (t.conversation_title) meta.append(el("span", { text: t.conversation_title }));
  meta.append(el("span", { text: t.created_at ? fmtDate(t.created_at) : "date unknown" }));
  // Distinguish a state the user actually set from one the extractor guessed — task_state_source
  // is "user" only when task_state_changed_at is set (a real transition through complete/reopen).
  if (t.task_state_source === "user") {
    meta.append(
      el("span", {
        class: "mono",
        text: t.task_state_changed_at ? `you set this ${fmtDate(t.task_state_changed_at)}` : "you set this",
      })
    );
  }
  item.append(meta);

  const errNote = el("p", { class: "state-body is-error", hidden: true });

  const actions = el("div", { class: "result-actions" });
  const stateBtn = el("button", {
    type: "button",
    class: "btn btn-sm",
    text: t.task_state === "completed" ? "Reopen" : "Complete",
  });
  const setBusy = (busy) => {
    stateBtn.disabled = busy;
  };
  stateBtn.addEventListener("click", async () => {
    errNote.hidden = true;
    setBusy(true);
    const prevState = t.task_state;
    const endpoint = t.task_state === "completed" ? "convsearch:taskReopen" : "convsearch:taskComplete";
    const res = await ask({ type: endpoint, id: t.memory_id, params: {} });
    if (!res.ok) {
      setBusy(false);
      errNote.textContent = (res && res.error) || "The request didn't succeed. The task is unchanged.";
      errNote.hidden = false;
      return;
    }
    const updated = res.data || t;
    if (onStateChange) onStateChange(prevState, updated.task_state);
    item.replaceWith(buildTaskCard(updated, expandEvidence, onStateChange));
  });
  actions.append(stateBtn);

  if (t.conversation_id) {
    const inspectRegion = el("div", { class: "inspect", hidden: true });
    const inspectBtn = el("button", { type: "button", class: "btn btn-sm", text: "View conversation", "aria-expanded": "false" });
    inspectBtn.addEventListener("click", () => {
      const open = inspectRegion.hidden;
      inspectRegion.hidden = !open;
      inspectBtn.setAttribute("aria-expanded", open ? "true" : "false");
      inspectBtn.textContent = open ? "Hide conversation" : "View conversation";
      if (open) renderInspect(t.conversation_id, inspectRegion, inspectBtn);
    });
    actions.append(inspectBtn);
    item.append(actions, inspectRegion);
  } else {
    item.append(actions);
  }

  item.append(errNote);
  attachEvidenceToggle(item, t.evidence, expandEvidence);

  return item;
}

$("task-project").addEventListener("change", loadTasks);
$("task-evidence").addEventListener("change", loadTasks);
for (const input of document.querySelectorAll('input[name="task-state"]')) {
  input.addEventListener("change", loadTasks);
}

/* ================================================================== */
/* PROJECTS                                                           */
/* ================================================================== */

const projectsOutput = $("projects-output");

async function loadProjects() {
  loadingInto(projectsOutput, 3);
  const res = await ask({ type: "convsearch:projects" });
  clear(projectsOutput);
  if (!res.ok) {
    projectsOutput.append(errorState(res, loadProjects));
    return;
  }
  const data = res.data || {};
  const projects = data.projects || [];
  if (!projects.length) {
    projectsOutput.append(
      stateBlock({
        title: "No projects yet.",
        body: "The server groups related decisions, tasks and conversations into projects as it distills your history. There's nothing to show until it finds one.",
      })
    );
    return;
  }
  projectsOutput.append(el("h2", { class: "section-head" }, ["Projects", el("span", { class: "section-count", text: String(data.count != null ? data.count : projects.length) })]));
  for (const p of projects) {
    const item = el("div", { class: "list-item", role: "button", tabindex: "0" });
    item.append(el("div", { class: "li-title", text: p.name }));
    const meta = el("div", { class: "li-meta" });
    meta.append(el("span", { text: plural(p.memory_count || 0, "memory").replace("memorys", "memories") }));
    meta.append(el("span", { text: plural(p.decision_count || 0, "decision") }));
    meta.append(el("span", { text: `${p.open_task_count || 0} open` }));
    meta.append(el("span", { text: plural(p.conversation_count || 0, "conversation") }));
    if (p.last_activity) meta.append(el("span", { text: timeAgo(p.last_activity) || fmtDate(p.last_activity) }));
    item.append(meta);
    const open = () => openProject(p.name);
    item.addEventListener("click", open);
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        open();
      }
    });
    projectsOutput.append(item);
  }
}

function statementList(items, variant) {
  const ul = el("ul", { class: `mini-list${variant ? ` ${variant}` : ""}` });
  for (const it of items) {
    const text = typeof it === "string" ? it : it.statement || "";
    ul.append(el("li", { text }));
  }
  return ul;
}

function dashSection(title, count, body) {
  if (!body) return null;
  const sec = el("div", { class: "dash-section" });
  const head = el("h2", { class: "section-head" }, [title]);
  if (count != null) head.append(el("span", { class: "section-count", text: String(count) }));
  sec.append(head);
  sec.append(body);
  return sec;
}

/** Strips path separators and characters Windows rejects in filenames, so a project name of
 * any shape becomes a safe local file. Falls back to "project" if nothing printable survives. */
function sanitizeFilename(name) {
  const cleaned = String(name || "")
    .replace(/[\\/:*?"<>|]+/g, "_")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || "project";
}

/**
 * Downloads the project's evidence-backed Markdown export via Blob + createObjectURL + a
 * programmatic `<a download>` click — the reliable route for triggering a file save from a
 * Chrome side-panel page (verified working: side panels are ordinary extension pages, not
 * content-script contexts, so blob: URLs and the download attribute behave as in any tab).
 * On error the server's message is shown inline next to the button, and the button always
 * returns to its normal enabled state so a retry is possible.
 */
async function exportProjectMarkdown(name, btn, errEl, container) {
  if (btn.disabled) return;
  btn.disabled = true;
  const prevText = btn.textContent;
  btn.textContent = "Exporting…";
  clear(errEl);

  const res = await ask({ type: "convsearch:projectExport", name });

  btn.disabled = false;
  btn.textContent = prevText;

  if (!res || !res.ok) {
    errEl.textContent = (res && res.error) || "Export failed. Check the server terminal and try again.";
    return;
  }

  const data = res.data || {};
  const markdown = typeof data.markdown === "string" ? data.markdown : "";
  const filename = `${sanitizeFilename(data.name || name)}.md`;

  try {
    const blob = new Blob([markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (err) {
    // Genuinely can't trigger a download here — fall back to an inline, scrollable, copyable
    // rendering rather than leaving the user with nothing.
    errEl.textContent = "Download isn't available here — showing the export below instead.";
    if (container) container.append(exportFallbackBlock(markdown, filename));
  }
}

/** Fallback when a real download can't be triggered: a scrollable `<pre>` with the raw
 * Markdown, plus a copy-to-clipboard button. The container scrolls its own overflow rather
 * than letting a long export widen the panel. */
function exportFallbackBlock(markdown, filename) {
  const wrap = el("div", { class: "dash-block diag-remediation" });
  wrap.append(el("h3", { class: "section-head", text: filename }));
  const pre = el("pre", { class: "mono diag-remediation-code", text: markdown });
  pre.style.maxHeight = "40vh";
  pre.style.overflowY = "auto";
  wrap.append(pre);
  const copyBtn = el("button", { type: "button", class: "btn btn-sm", text: "Copy Markdown" });
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(markdown);
      copyBtn.textContent = "Copied";
    } catch {
      copyBtn.textContent = "Couldn't copy — select the text above";
    } finally {
      setTimeout(() => {
        copyBtn.textContent = "Copy Markdown";
      }, 1600);
    }
  });
  wrap.append(el("div", { class: "result-actions" }, [copyBtn]));
  return wrap;
}

async function openProject(name) {
  loadingInto(projectsOutput, 3);
  const res = await ask({ type: "convsearch:project", name });
  clear(projectsOutput);
  if (!res.ok) {
    projectsOutput.append(errorState(res, () => openProject(name)));
    return;
  }
  const p = res.data || {};
  projectsOutput.append(el("button", { type: "button", class: "btn btn-ghost btn-sm back-btn", text: "‹ All projects", onclick: loadProjects }));
  projectsOutput.append(el("div", { class: "detail-statement", text: p.name || name }));

  // export — a Markdown copy of everything below, for pasting elsewhere.
  {
    const row = el("div", { class: "result-actions" });
    const exportBtn = el("button", { type: "button", class: "btn btn-sm", text: "Export Markdown" });
    const exportErr = el("span", { class: "state-body", role: "status", "aria-live": "polite" });
    row.append(exportBtn, exportErr);
    exportBtn.addEventListener("click", () => exportProjectMarkdown(p.name || name, exportBtn, exportErr, projectsOutput));
    projectsOutput.append(row);
  }

  // summary
  if (p.summary) projectsOutput.append(el("div", { class: "dash-block", text: p.summary }));

  // stat row
  const decisions = p.decisions || [];
  const openTasks = p.open_tasks || [];
  const completedTasks = p.completed_tasks || [];
  const risks = p.risks || [];
  const stats = el("div", { class: "stat-grid" });
  const statDefs = [
    [decisions.length, "decisions"],
    [openTasks.length, "open tasks"],
    [completedTasks.length, "done"],
    [risks.length, "risks"],
    [p.evidence_count || 0, "evidence"],
  ];
  for (const [num, label] of statDefs) {
    stats.append(el("div", { class: "stat" }, [el("div", { class: "stat-num", text: String(num) }), el("div", { class: "stat-label", text: label })]));
  }
  projectsOutput.append(stats);

  const add = (sec) => {
    if (sec) projectsOutput.append(sec);
  };

  // architecture
  if ((p.architecture || []).length) add(dashSection("Current architecture", p.architecture.length, el("div", { class: "dash-block" }, statementList(p.architecture))));

  // decisions (active) — each with evidence quotes. Always shown (with a "none recorded" line)
  // so the section's presence isn't mistaken for a broken load.
  {
    const body = el("div", { class: "dash-section" });
    if (decisions.length) {
      for (const d of decisions) {
        const block = el("div", { class: "dash-block" });
        const head = el("div", { class: "badges" });
        if (d.status) head.append(pill(d.status, memStatusVariant(d.status)));
        block.append(head);
        block.append(el("div", { class: "li-title", text: d.statement || "" }));
        const dEvidence = d.evidence || [];
        if (dEvidence.length) {
          for (const ev of dEvidence) block.append(quoteBlock(ev.quote, "source-quote"));
        } else {
          block.append(el("p", { class: "state-body", text: "No evidence recorded." }));
        }
        body.append(block);
      }
    } else {
      body.append(el("p", { class: "state-body", text: "No decisions recorded for this project yet." }));
    }
    add(dashSection("Decisions", decisions.length, body));
  }

  // superseded decisions
  if ((p.superseded_decisions || []).length) add(dashSection("Superseded decisions", p.superseded_decisions.length, el("div", { class: "dash-block" }, statementList(p.superseded_decisions, "muted"))));

  // rejected alternatives
  if ((p.rejected_alternatives || []).length) add(dashSection("Rejected alternatives", p.rejected_alternatives.length, el("div", { class: "dash-block" }, statementList(p.rejected_alternatives, "muted"))));

  // tasks — open tasks always shown so an empty backlog reads as "none", not "broken".
  add(
    dashSection(
      "Open tasks",
      openTasks.length,
      el("div", { class: "dash-block" }, openTasks.length ? statementList(openTasks) : el("p", { class: "state-body", text: "No open tasks recorded." }))
    )
  );
  if (completedTasks.length) add(dashSection("Completed tasks", completedTasks.length, el("div", { class: "dash-block" }, statementList(completedTasks, "done"))));

  // risks + known bugs — risks always shown.
  add(
    dashSection(
      "Risks",
      risks.length,
      el("div", { class: "dash-block" }, risks.length ? statementList(risks) : el("p", { class: "state-body", text: "No risks recorded." }))
    )
  );
  if ((p.known_bugs || []).length) add(dashSection("Known bugs", p.known_bugs.length, el("div", { class: "dash-block" }, statementList(p.known_bugs))));
  if ((p.next_milestones || []).length) add(dashSection("Next milestones", p.next_milestones.length, el("div", { class: "dash-block" }, statementList(p.next_milestones))));

  // timeline
  if ((p.timeline || []).length) {
    const tl = el("div", { class: "timeline" });
    for (const t of p.timeline) {
      const item = el("div", { class: "tl-item" });
      const head = el("div", {}, []);
      if (t.kind) head.append(pill(t.kind, "role"));
      if (t.status) head.append(pill(t.status, memStatusVariant(t.status)));
      item.append(head);
      item.append(el("div", { class: "source-quote", text: t.statement || "" }));
      if (t.created_at) item.append(el("div", { class: "tl-date", text: fmtDate(t.created_at) }));
      tl.append(item);
    }
    add(dashSection("Timeline", p.timeline.length, tl));
  }

  // relevant conversations
  const convs = p.conversations || [];
  if (convs.length) {
    const list = el("div", { class: "dash-block" });
    const inner = el("div", { class: "mini-list" });
    for (const c of convs) {
      const id = Array.isArray(c) ? c[0] : c.conversation_id;
      const title = Array.isArray(c) ? c[1] : c.title;
      const btn = el("button", { type: "button", class: "link-btn", text: title || id });
      btn.addEventListener("click", () => openConversationTab(id));
      inner.append(el("div", {}, btn));
    }
    list.append(inner);
    add(dashSection("Relevant conversations", convs.length, list));
  }
}

/** Projects list conversations by id only; fetch to learn the URL, then open (or inspect). */
async function openConversationTab(id) {
  if (!id) return;
  const res = await ask({ type: "convsearch:conversation", id });
  if (res.ok && chatgptUrl(res.data && res.data.url)) {
    openTab(res.data.url);
  }
}

/* ================================================================== */
/* TIMELINE                                                           */
/* ================================================================== */

const timelineOutput = $("timeline-output");
const timelineQuery = $("timeline-query");
const timelineHint = $("timeline-hint");
let timelineSeq = 0;

const TIMELINE_PARTITIONS = [
  ["active", "Active", "This is the current, standing decision or fact."],
  ["contested", "Contested", "Conflicting statements exist and haven't been resolved."],
  ["superseded", "Superseded", "Replaced by a later decision — see what replaced it below."],
  ["rejected", "Rejected", "Considered and explicitly turned down."],
];

function timelineNodeCard(node) {
  const item = el("div", { class: "tl-item" });
  const head = el("div", { class: "badges" });
  if (node.kind) head.append(pill(node.kind, "role"));
  if (node.status) head.append(pill(node.status, memStatusVariant(node.status)));
  if (node.project) head.append(pill(node.project, "accent"));
  item.append(head);
  item.append(el("div", { class: "li-title", text: node.statement || "" }));

  const meta = el("div", { class: "li-meta" });
  if (node.conversation_title) meta.append(el("span", { text: node.conversation_title }));
  item.append(meta);
  item.append(el("div", { class: "tl-date", text: node.created_at ? fmtDate(node.created_at) : "date unknown" }));

  // The "what replaced it and why" payoff for superseded/rejected nodes.
  const supersededBy = node.superseded_by || [];
  if (supersededBy.length) {
    item.append(el("div", { class: "li-meta", text: `Superseded by: ${supersededBy.join(", ")}` }));
  }
  const reasons = node.reasons || [];
  if (reasons.length) {
    item.append(el("div", { class: "source-quote", text: reasons.join(" · ") }));
  } else if (node.status === "superseded" || node.status === "rejected") {
    item.append(el("p", { class: "state-body", text: "The reason was not recorded." }));
  }

  // Timeline is always fetched with evidence=1 (see runTimeline), so the array is the truthful signal here.
  attachEvidenceToggle(item, node.evidence, false);
  return item;
}

function timelinePartition(label, description, nodes) {
  if (!nodes || !nodes.length) return null;
  const sec = el("div", { class: "dash-section" });
  const head = el("h2", { class: "section-head" }, [label, el("span", { class: "section-count", text: String(nodes.length) })]);
  sec.append(head);
  sec.append(el("p", { class: "state-body", text: description }));
  for (const n of nodes) sec.append(timelineNodeCard(n));
  return sec;
}

function updateTimelineHint() {
  const q = timelineQuery.value.trim();
  $("timeline-submit").disabled = !q;
  timelineHint.textContent = q ? "" : "Enter a topic to search — a bare Run won't fire without one.";
}

async function runTimeline() {
  const q = timelineQuery.value.trim();
  if (!q) {
    updateTimelineHint();
    return;
  }
  const seq = ++timelineSeq;
  const project = $("timeline-project").value.trim();
  loadingInto(timelineOutput, 3);

  const res = await ask({ type: "convsearch:timeline", params: { q, project, limit: 60, evidence: 1 } });
  if (seq !== timelineSeq) return;
  clear(timelineOutput);

  if (!res.ok) {
    // A missing/blocked `q` or bad filters come back as a 400 with a specific message.
    timelineOutput.append(errorState(res, runTimeline));
    return;
  }

  const data = res.data || {};
  if (!data.matched_count) {
    timelineOutput.append(
      stateBlock({
        title: `No memories matched "${data.topic || q}".`,
        body: "Try a broader term, or check the Memories view to see what's actually been captured.",
        actions: [{ label: "Open Memories", onClick: () => setView("memories") }],
      })
    );
    return;
  }

  const context = el("div", { class: "li-meta" });
  context.append(
    el(
      "span",
      { class: "mono" },
      `${plural(data.matched_count, "match").replace("matchs", "matches")} · first seen ${data.first_seen ? fmtDate(data.first_seen) : "date unknown"} · last seen ${data.last_seen ? fmtDate(data.last_seen) : "date unknown"}`
    )
  );
  timelineOutput.append(context);

  for (const [key, label, description] of TIMELINE_PARTITIONS) {
    const sec = timelinePartition(label, description, data[key]);
    if (sec) timelineOutput.append(sec);
  }
}

$("timeline-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runTimeline();
});
timelineQuery.addEventListener("input", updateTimelineHint);
updateTimelineHint();

/* ================================================================== */
/* STATUS                                                             */
/* ================================================================== */

const statusOutput = $("status-output");
let reindexing = false;
let statusSeq = 0;

/*
 * Ambient auto-index readout. Indexing is automatic, so the normal path is a quiet status line
 * (dot + label) driven by /health — never a prominent "Rebuild index" button. Only when the
 * index is stale/failed do we surface the quiet, tertiary .index-rebuild control as recovery,
 * which routes through the same reindex plumbing (runReindex / homeRebuildIndex / …).
 * `health` is the /health payload; `stale` can be forced true (e.g. the captures response's own
 * stale_index flag). `onRebuild` is the recovery handler for the stale case.
 */
function indexStatusEl(health, onRebuild, { stale: forceStale = false } = {}) {
  const h = health || {};
  const indexing = Boolean(h.indexing);
  const stale = forceStale || Boolean(h.stale_index);
  let variant, label;
  if (indexing) {
    variant = "busy";
    label = "Indexing…";
  } else if (stale) {
    variant = "stale";
    label = "Some captures aren't indexed yet";
  } else {
    variant = "ok";
    label = "Index up to date";
  }
  const wrap = el("div", { class: `index-status index-status--${variant}`, role: "status" });
  wrap.append(el("span", { text: label }));
  if (variant === "stale" && onRebuild) {
    const btn = el("button", { type: "button", class: "index-rebuild", text: reindexing ? "Rebuilding…" : "Rebuild now" });
    btn.disabled = reindexing;
    btn.addEventListener("click", onRebuild);
    wrap.append(btn);
  }
  return wrap;
}

/*
 * The "Learning" panel surfaces the interaction counts the server has accumulated locally. The
 * fetch is fired after the panel is in the DOM and its result guarded by statusSeq so a slow
 * reply can't land in a status view the user has already re-rendered.
 */
const LEARN_TILE_DEFS = [
  ["search", "searches logged"],
  ["open", "results opened"],
  ["distinct_queries", "distinct queries"],
  ["learned_preferences", "learned preferences"],
];

/** Whether the user just now ran a learn pass (kept across a status re-render). */
let learning = false;

/** Refresh only the stat tiles, guarded by statusSeq so a stale reply can't land. */
function refreshLearnStats(seq, numEls, note) {
  ask({ type: "convsearch:learnStats" }).then((res) => {
    if (seq !== statusSeq) return;
    const stats = (res && res.ok && res.data && res.data.stats) || {};
    LEARN_TILE_DEFS.forEach(([key], i) => {
      const v = stats[key];
      numEls[i].textContent = typeof v === "number" ? String(v) : "0";
    });
    if ((!res || !res.ok) && note) {
      note.textContent = "Learning stats are unavailable right now — interactions are still logged locally as you search, open and ask.";
    }
  });
}

/** A single learned-preference row: the note plus a subtle weight indicator. */
function prefRow(pref) {
  const row = el("div", { class: "pref-item" });
  row.append(el("div", { class: "pref-note", text: pref.note || "(no note)" }));
  const meta = el("div", { class: "pref-meta" });
  if (typeof pref.weight === "number") {
    const pct = Math.max(0, Math.min(1, pref.weight)) * 100;
    meta.append(
      el("span", { class: "pref-weight", title: `weight ${pref.weight.toFixed(2)}`, "aria-label": `weight ${pref.weight.toFixed(2)}` }, [
        el("span", { class: "pref-weight-track" }, el("span", { class: "pref-weight-fill", style: `width:${pct.toFixed(0)}%` })),
        el("span", { class: "pref-weight-val mono", text: pref.weight.toFixed(2) }),
      ])
    );
  }
  if (pref.created_at) meta.append(el("span", { class: "tl-date", text: timeAgo(pref.created_at) || fmtDate(pref.created_at) }));
  if (meta.firstChild) row.append(meta);
  return row;
}

/** Learned preferences list, with its own loading / empty / error state and refresh. */
function loadLearnPrefs(seq, listEl) {
  clear(listEl);
  listEl.append(el("div", { class: "state" }, spinner()));
  ask({ type: "convsearch:learnPrefs", params: { limit: 20 } }).then((res) => {
    if (seq !== statusSeq) return;
    clear(listEl);
    if (!res || !res.ok) {
      listEl.append(errorState(res, () => loadLearnPrefs(seq, listEl)));
      return;
    }
    const prefs = (res.data && Array.isArray(res.data.preferences) ? res.data.preferences : []).filter((p) => p && p.note);
    if (!prefs.length) {
      listEl.append(el("p", { class: "learn-note", text: "No learned preferences yet. Run a learn pass once you've searched, opened and asked a few times." }));
      return;
    }
    for (const p of prefs) listEl.append(prefRow(p));
  });
}

function renderLearningPanel(seq) {
  const panel = el("div", { class: "learn-panel dash-section" });
  panel.append(el("h2", { class: "section-head" }, ["Learning", el("span", { class: "section-count", text: "local" })]));
  const grid = el("div", { class: "stat-grid" });
  const numEls = LEARN_TILE_DEFS.map(([, label]) => {
    const numEl = el("div", { class: "stat-num", text: "—" });
    grid.append(el("div", { class: "stat" }, [numEl, el("div", { class: "stat-label", text: label })]));
    return numEl;
  });
  panel.append(grid);
  const note = el("p", {
    class: "learn-note",
    text: "The local model uses these interactions to improve ranking and suggestions. All of this data stays on your machine.",
  });
  panel.append(note);

  // "Learn now" controls: run a self-improvement pass, optionally with the local LLM.
  const llmToggle = el("input", { type: "checkbox", id: "learn-use-llm", checked: true });
  const llmLabel = el("label", { class: "toggle" }, [llmToggle, el("span", { text: "Use local model only" })]);

  const actions = el("div", { class: "result-actions learn-actions" });
  const learnBtn = el("button", { type: "button", class: "btn btn-primary btn-sm", text: learning ? "Learning…" : "Learn now" });
  learnBtn.disabled = learning;
  actions.append(learnBtn);
  actions.append(llmLabel);
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Refresh", onclick: () => loadStatus() }));
  panel.append(actions);

  // Where the run's result line + notes land.
  const result = el("div", { class: "learn-result", role: "status", "aria-live": "polite" });
  panel.append(result);

  learnBtn.addEventListener("click", () => runLearn(seq, learnBtn, llmToggle.checked, result, numEls, note, prefsList));

  // Learned preferences sub-section, with its own refresh.
  const prefsSection = el("div", { class: "dash-section learn-prefs-section" });
  const prefsHead = el("h3", { class: "section-head" }, ["Learned preferences"]);
  prefsHead.append(el("button", { type: "button", class: "btn btn-ghost btn-sm", text: "Refresh", onclick: () => loadLearnPrefs(statusSeq, prefsList) }));
  prefsSection.append(prefsHead);
  const prefsList = el("div", { class: "pref-list" });
  prefsSection.append(prefsList);
  panel.append(prefsSection);

  statusOutput.append(panel);

  refreshLearnStats(seq, numEls, note);
  loadLearnPrefs(seq, prefsList);
}

async function runLearn(seq, btn, useLlm, result, numEls, note, prefsList) {
  if (learning) return;
  learning = true;
  btn.disabled = true;
  btn.textContent = "Learning…";
  clear(result);
  result.append(stateBlock({ spinner: true, title: "Running a learn pass…", body: "Reviewing your recent interactions to distill preferences. This can take a while if the local model runs." }));

  const res = await ask({ type: "convsearch:learnRun", params: { use_llm: Boolean(useLlm) } });
  learning = false;
  // The status view may have been re-rendered (a Refresh) while the long request ran; if so,
  // these nodes are detached — writing to them is harmless, but skip the shared refreshes.
  const live = seq === statusSeq;
  btn.disabled = false;
  btn.textContent = "Learn now";
  clear(result);

  if (!res || !res.ok) {
    result.append(errorState(res, () => runLearn(seq, btn, useLlm, result, numEls, note, prefsList)));
    return;
  }

  const d = res.data || {};
  const written = Number(d.notes_written || 0);
  const via = [d.backend, d.model].filter(Boolean).join(":");
  result.append(
    el("div", { class: "banner banner-ok" }, `Wrote ${plural(written, "preference")}${via ? ` via ${via}` : ""}${typeof d.events_read === "number" ? ` from ${plural(d.events_read, "event")}` : ""}.`)
  );

  const notes = (Array.isArray(d.notes) ? d.notes : []).filter((n) => typeof n === "string" && n.trim());
  if (notes.length) result.append(el("div", { class: "dash-block learn-notes" }, statementList(notes)));

  if (live) {
    refreshLearnStats(seq, numEls, note);
    loadLearnPrefs(seq, prefsList);
  }
}

/**
 * Read the background worker's capture-queue health directly from chrome.storage.local.
 * background.js owns writing `queueState`/`captureQueue`; this is a read-only peek so the panel
 * can surface queue trouble that today only shows up in the popup. Never throws.
 */
function readQueueState() {
  const fallback = { nextAttemptAt: 0, failures: 0, offline: false, lastError: null };
  return new Promise((resolve) => {
    try {
      chrome.storage.local.get({ queueState: fallback }, (stored) => {
        if (chrome.runtime.lastError) {
          resolve(fallback);
          return;
        }
        resolve((stored && stored.queueState) || fallback);
      });
    } catch {
      resolve(fallback);
    }
  });
}

/** Banner surfacing queued/failed captures with a next action; nothing is shown when healthy. */
function captureQueueBanner(pending, state) {
  const failures = Number(state.failures) || 0;
  const bits = [];
  if (pending > 0) bits.push(`${plural(pending, "capture")} queued`);
  if (state.offline) bits.push("the capture queue can't reach the server");
  else if (failures > 0) bits.push(`${plural(failures, "failed attempt")}`);
  let text = bits.length ? `${cap(bits.join(" — "))}.` : "The capture queue is having trouble.";
  if (state.lastError) text += ` Last error: ${state.lastError}.`;
  text += " It retries automatically once the server is reachable — use Retry now to check immediately.";
  const banner = el("div", { class: `banner ${state.offline ? "banner-bad" : "banner-warn"}` }, el("span", {}, text));
  banner.append(el("button", { type: "button", class: "btn btn-sm", text: "Retry now", onclick: loadStatus }));
  return banner;
}

async function loadStatus() {
  const seq = ++statusSeq;
  loadingInto(statusOutput, 2);
  const res = await refreshConn();
  if (seq !== statusSeq) return;
  clear(statusOutput);

  const online = res && res.online;
  const health = (res && res.health) || {};
  const serverUrl = (res && res.serverUrl) || "http://127.0.0.1:8756";

  // headline banner
  if (!online) {
    statusOutput.append(el("div", { class: "banner banner-bad" }, `Server offline — not reachable at ${serverUrl}. Start it, then refresh.`));
  } else if (health.problem) {
    statusOutput.append(el("div", { class: "banner banner-warn" }, String(health.problem)));
  } else if (health.indexing) {
    statusOutput.append(el("div", { class: "banner banner-warn" }, "Indexing in progress — new conversations become searchable in a few seconds."));
  } else if (health.stale_index) {
    statusOutput.append(el("div", { class: "banner banner-warn" }, "Some captured conversations aren't indexed yet. Auto-indexing will catch up; use the rebuild below only if it stays stale."));
  } else if (health.indexed) {
    statusOutput.append(el("div", { class: "banner banner-ok" }, "Everything captured so far is indexed and searchable."));
  }

  // capture-queue health — silent when the queue is empty and healthy, so it never adds noise.
  const queueState = await readQueueState();
  if (seq !== statusSeq) return;
  const pendingCaptures = (res && res.pending) || 0;
  if (pendingCaptures > 0 || queueState.offline || (Number(queueState.failures) || 0) > 0) {
    statusOutput.append(captureQueueBanner(pendingCaptures, queueState));
  }

  // stat tiles
  const stats = el("div", { class: "stat-grid" });
  const tile = (num, label) => el("div", { class: "stat" }, [el("div", { class: "stat-num", text: String(num) }), el("div", { class: "stat-label", text: label })]);
  stats.append(tile(health.conversations || 0, "conversations"));
  stats.append(tile(health.messages || 0, "messages"));
  stats.append(tile(health.captured_conversations || 0, "captured live"));
  if (res && res.pending) stats.append(tile(res.pending, "queued"));
  statusOutput.append(stats);

  // detail table
  const detail = el("div", { class: "dash-block" });
  const kv = (k, v) => detail.append(el("div", { class: "kv" }, [el("span", { class: "kv-key", text: k }), el("span", { class: "kv-val", text: v })]));
  kv("Server URL", serverUrl);
  kv("Reachable", online ? "yes" : "no");
  if (health.workspace) kv("Workspace", health.workspace);
  if (health.database) kv("Database", health.database);
  kv("Indexed", health.indexed ? "yes" : "no");
  kv("Index up to date", health.stale_index ? "no — stale" : "yes");
  kv("Indexing now", health.indexing ? "yes" : "no");
  kv("Auto-index", health.auto_index === false ? "off" : "on");
  if (res && res.lastCaptureAt) kv("Last capture", timeAgo(res.lastCaptureAt) || fmtDate(res.lastCaptureAt));
  if (res && res.error) kv("Last error", res.error);
  statusOutput.append(detail);

  // Auto-index is ambient: a quiet status line, with the demoted rebuild only surfaced on stale.
  if (online) statusOutput.append(indexStatusEl(health, runReindex));
  const actions = el("div", { class: "result-actions" });
  actions.append(el("button", { type: "button", class: "btn", text: "Refresh", onclick: loadStatus }));
  statusOutput.append(actions);

  // Local-model setup assistant — its own async slot so a slow LLM check never blocks the
  // health/rebuild UI above it.
  renderDiagnosticsPanel(seq);

  // Learning panel — interaction counts the server has gathered from this device.
  renderLearningPanel(seq);

  if (!online) {
    statusOutput.append(stateBlock({ variant: "offline", title: "Start the server", body: "convsearch runs against a local server that isn't responding. Start it with the launcher — or run the command below in a terminal from the convsearch folder — then refresh.", code: "convsearch serve --workspace ./workspace" }));
  }
}

/** One check row: a pill for pass/fail plus the name and the server's free-text detail. */
function diagnosticsCheckRow(check) {
  const ok = Boolean(check && check.ok);
  const row = el("div", { class: "kv" });
  row.append(el("span", { class: "kv-key" }, [pill(ok ? "OK" : "Fail", ok ? "ok" : "bad"), el("span", { text: ` ${(check && check.name) || ""}` })]));
  row.append(el("span", { class: "kv-val", text: (check && check.detail) || "" }));
  return row;
}

function diagnosticsCheckGroup(title, checks) {
  const body = el("div", { class: "dash-block" });
  if (checks.length) {
    for (const c of checks) body.append(diagnosticsCheckRow(c));
  } else {
    body.append(el("p", { class: "state-body", text: "No checks reported." }));
  }
  return dashSection(title, checks.length, body);
}

/**
 * Copy-pasteable remediation block — same code-styled look as the offline-server hint in
 * `errorState`/`stateBlock` (a `.mono` block on `--code-bg`), but rendered as its own element
 * since remediation is a multi-line, multi-command list rather than one string.
 */
function diagnosticsRemediationBlock(lines) {
  const wrap = el("div", { class: "dash-block diag-remediation" });
  wrap.append(el("pre", { class: "mono diag-remediation-code", text: lines.join("\n") }));
  const copyBtn = el("button", { type: "button", class: "btn btn-sm", text: "Copy" });
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      copyBtn.textContent = "Copied";
    } catch {
      copyBtn.textContent = "Couldn't copy — select the text above";
    } finally {
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 1600);
    }
  });
  wrap.append(el("div", { class: "result-actions" }, [copyBtn]));
  return wrap;
}

/** Builds the filled-in diagnostics body (banner + both check groups + remediation-when-not-ready). */
function fillDiagnosticsBody(body, data) {
  clear(body);
  const d = data || {};
  const ready = Boolean(d.ready);
  const bannerText = [d.summary, d.backend ? `Backend: ${d.backend}.` : ""].filter(Boolean).join(" ") || (ready ? "Ready." : "Not ready.");
  body.append(el("div", { class: `banner ${ready ? "banner-ok" : "banner-bad"}` }, bannerText));

  const llmChecks = Array.isArray(d.llm_checks) ? d.llm_checks : [];
  const doctorChecks = Array.isArray(d.doctor_checks) ? d.doctor_checks : [];
  body.append(diagnosticsCheckGroup("LLM readiness", llmChecks));
  body.append(diagnosticsCheckGroup("Workspace", doctorChecks));

  if (!ready) {
    const remediation = (Array.isArray(d.remediation) ? d.remediation : []).filter((line) => typeof line === "string" && line.trim());
    if (remediation.length) body.append(dashSection("Fix it", remediation.length, diagnosticsRemediationBlock(remediation)));
  }
}

async function loadDiagnosticsInto(seq, body) {
  clear(body);
  body.append(skeletonList(2));
  const res = await ask({ type: "convsearch:diagnostics" });
  if (seq !== statusSeq) return;
  clear(body);
  if (!res || !res.ok) {
    body.append(errorState(res, () => loadDiagnosticsInto(seq, body)));
    return;
  }
  fillDiagnosticsBody(body, res.data || {});
}

function renderDiagnosticsPanel(seq) {
  const panel = el("div", { class: "dash-section diagnostics-panel" });
  panel.append(el("h2", { class: "section-head" }, ["Local model setup"]));
  const body = el("div", { class: "dash-section" });
  panel.append(body);
  statusOutput.append(panel);
  loadDiagnosticsInto(seq, body);
}

async function runReindex() {
  if (reindexing) return;
  reindexing = true;
  await loadStatus();
  const res = await ask({ type: "convsearch:reindex" });
  reindexing = false;
  await loadStatus();
  if (res && res.ok) {
    statusOutput.append(el("div", { class: "banner banner-ok" }, typeof res.indexed_passages === "number" ? `Rebuilt — ${plural(res.indexed_passages, "passage")} indexed.` : "Index rebuilt."));
  } else {
    statusOutput.append(el("div", { class: "banner banner-bad" }, (res && res.error) || "Rebuild failed. Check the server terminal and try again."));
  }
}

/* ================================================================== */
/* CAPTURES                                                           */
/* ================================================================== */

const capturesOutput = $("captures-output");
let capturesSeq = 0;

function readCaptureFilters() {
  const sourceInput = document.querySelector('input[name="capture-source"]:checked');
  return {
    source: (sourceInput && sourceInput.value) || "all",
    problems: $("captures-problems").checked,
  };
}

/** Same reindex call/timeout/guard as Status/Home's runReindex — reuses the one shared
 * `reindexing` flag rather than adding a third independent in-flight state. */
async function capturesRebuildIndex() {
  if (reindexing) return;
  reindexing = true;
  await loadCaptures();
  const res = await ask({ type: "convsearch:reindex" });
  reindexing = false;
  await loadCaptures();
  if (res && res.ok) {
    capturesOutput.append(
      el("div", { class: "banner banner-ok" }, typeof res.indexed_passages === "number" ? `Rebuilt — ${plural(res.indexed_passages, "passage")} indexed.` : "Index rebuilt.")
    );
  } else {
    capturesOutput.append(el("div", { class: "banner banner-bad" }, (res && res.error) || "Rebuild failed. Check the server terminal and try again."));
  }
}

function capturePipelinePill(label, ok) {
  return pill(label, ok ? "ok" : "warn");
}

async function loadCaptures() {
  const seq = ++capturesSeq;
  const f = readCaptureFilters();
  loadingInto(capturesOutput, 4);

  const res = await ask({
    type: "convsearch:captures",
    params: { source: f.source, limit: 100, problems: f.problems ? 1 : undefined },
  });
  if (seq !== capturesSeq) return;
  clear(capturesOutput);

  if (!res.ok) {
    // A bad `source` is a 400 with a specific server message — surface it verbatim.
    capturesOutput.append(errorState(res, loadCaptures));
    return;
  }

  const data = res.data || {};
  const items = data.items || [];

  // Auto-index status; the demoted rebuild recovery shows only because this response is stale.
  capturesOutput.append(indexStatusEl(null, capturesRebuildIndex, { stale: Boolean(data.stale_index) }));

  const stats = el("div", { class: "stat-grid" });
  const tile = (num, label) => el("div", { class: "stat" }, [el("div", { class: "stat-num", text: String(num) }), el("div", { class: "stat-label", text: label })]);
  stats.append(tile(data.total || 0, "total"));
  stats.append(tile(data.live_captured || 0, "live captured"));
  stats.append(tile(data.imported || 0, "imported"));
  stats.append(tile(data.not_indexed || 0, "not indexed"));
  stats.append(tile(data.not_segmented || 0, "not segmented"));
  capturesOutput.append(stats);

  if (!items.length) {
    capturesOutput.append(
      stateBlock({
        variant: "empty",
        title: f.problems ? "No captures with problems." : "No captures yet.",
        body: f.problems
          ? "Nothing matches this filter right now — clear \"Problems only\" to see everything captured."
          : "Captures appear automatically while the server runs and you browse chatgpt.com. Optional: import a ChatGPT export from the CLI to backfill your existing history.",
      })
    );
    return;
  }

  capturesOutput.append(el("h2", { class: "section-head" }, ["Captures", el("span", { class: "section-count", text: String(data.count != null ? data.count : items.length) })]));

  for (const c of items) {
    const item = el("div", { class: "list-item" });
    item.append(el("div", { class: "li-title", text: c.title || "(untitled)" }));

    const badges = el("div", { class: "badges" });
    badges.append(pill(c.source === "live-capture" ? "live capture" : "import", c.source === "live-capture" ? "accent" : "role"));
    badges.append(capturePipelinePill("indexed", c.indexed));
    badges.append(capturePipelinePill("segmented", c.segmented));
    badges.append(capturePipelinePill("memories extracted", c.memories_extracted));
    item.append(badges);

    const meta = el("div", { class: "li-meta" });
    meta.append(el("span", { text: c.captured_at ? fmtDate(c.captured_at) : "date unknown" }));
    meta.append(el("span", { text: plural(c.message_count || 0, "message") }));
    if (typeof c.passage_count === "number") meta.append(el("span", { text: plural(c.passage_count, "passage") }));
    if (typeof c.memory_count === "number") meta.append(el("span", { text: plural(c.memory_count, "memory").replace("memorys", "memories") }));
    item.append(meta);

    const warnings = c.warnings || [];
    if (warnings.length) {
      const chips = el("div", { class: "chips" });
      for (const w of warnings) chips.append(chip(w));
      item.append(chips);
    }

    const actions = el("div", { class: "result-actions" });
    const url = chatgptUrl(c.source_url);
    if (url) actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Open in ChatGPT", onclick: () => openTab(url) }));
    if (actions.childNodes.length) item.append(actions);

    capturesOutput.append(item);
  }
}

$("captures-problems").addEventListener("change", loadCaptures);
for (const input of document.querySelectorAll('input[name="capture-source"]')) {
  input.addEventListener("change", loadCaptures);
}

/* ================================================================== */
/* PRIVACY                                                            */
/* ================================================================== */

const privacyOutput = $("privacy-output");
let privacySeq = 0;

/**
 * The whole point of this screen is that it's verified, not asserted: the banner reflects
 * `local_only`/`cloud_would_be_used` straight from the server response — never a hardcoded
 * "you are private" claim — because those fields come from the same backend-resolution logic
 * `/ask` and `/plan` actually use.
 */
async function loadPrivacy() {
  const seq = ++privacySeq;
  loadingInto(privacyOutput, 3);

  const res = await ask({ type: "convsearch:privacy" });
  if (seq !== privacySeq) return;
  clear(privacyOutput);

  if (!res.ok) {
    privacyOutput.append(errorState(res, loadPrivacy));
    return;
  }

  const d = res.data || {};
  const llm = d.llm || {};

  if (d.cloud_would_be_used) {
    privacyOutput.append(
      el(
        "div",
        { class: "banner banner-warn" },
        `Cloud backend (${llm.effective_backend || "cloud"}) would be used for the next ask/plan request — the note below states exactly what that sends.`
      )
    );
  } else if (d.local_only) {
    privacyOutput.append(el("div", { class: "banner banner-ok" }, "Local only — your raw conversations stay on this machine."));
  }

  const pathsBlock = el("div", { class: "dash-block" });
  const kv = (k, v) => pathsBlock.append(el("div", { class: "kv" }, [el("span", { class: "kv-key", text: k }), el("span", { class: "kv-val mono", text: v || "—" })]));
  kv("Workspace", d.workspace_path);
  kv("Database", d.database_path);
  kv("Index", d.index_path);
  privacyOutput.append(dashSection("Storage paths", null, pathsBlock));

  const netBlock = el("div", { class: "dash-block" });
  netBlock.append(el("div", { class: "kv" }, [el("span", { class: "kv-key", text: "Server bind" }), el("span", { class: "kv-val mono", text: d.server_bind || "—" })]));
  netBlock.append(el("div", { class: "badges" }, pill(d.loopback_only ? "loopback only" : "not loopback-only", d.loopback_only ? "ok" : "bad")));
  privacyOutput.append(dashSection("Network", null, netBlock));

  const llmBlock = el("div", { class: "dash-block" });
  const kv2 = (k, v) => llmBlock.append(el("div", { class: "kv" }, [el("span", { class: "kv-key", text: k }), el("span", { class: "kv-val", text: v })]));
  kv2("Backend mode", llm.backend_mode || "—");
  kv2("Effective backend", llm.effective_backend || "—");
  kv2("Ollama host", llm.ollama_host || "—");
  // Presence-only — this never implies the key value itself is readable.
  kv2("Cloud key configured", llm.cloud_configured ? "yes" : "no");
  kv2("Cloud would be used now", llm.cloud_would_be_used ? "yes" : "no");
  privacyOutput.append(dashSection("LLM backend", null, llmBlock));

  if (d.cloud_payload_note) {
    // Rendered verbatim — this text was derived from the real prompt-building code and must
    // not be paraphrased here.
    privacyOutput.append(dashSection("What would be sent in cloud mode", null, el("p", { class: "state-body", text: d.cloud_payload_note })));
  }

  const counts = d.counts || {};
  const stats = el("div", { class: "stat-grid" });
  const tile = (num, label) => el("div", { class: "stat" }, [el("div", { class: "stat-num", text: String(num || 0) }), el("div", { class: "stat-label", text: label })]);
  stats.append(tile(counts.conversations, "conversations"));
  stats.append(tile(counts.messages, "messages"));
  stats.append(tile(counts.memories, "memories"));
  privacyOutput.append(stats);
}

/* ================================================================== */
/* HOME · dashboard                                                   */
/* ================================================================== */

const homeOutput = $("home-output");
let homeSeq = 0;

/** Same reindex call/timeout/guard as the Status tab's runReindex(), rendered into home-output. */
async function homeRebuildIndex() {
  if (reindexing) return;
  reindexing = true;
  await loadHome();
  const res = await ask({ type: "convsearch:reindex" });
  reindexing = false;
  await loadHome();
  if (res && res.ok) {
    homeOutput.append(
      el("div", { class: "banner banner-ok" }, typeof res.indexed_passages === "number" ? `Rebuilt — ${plural(res.indexed_passages, "passage")} indexed.` : "Index rebuilt.")
    );
  } else {
    homeOutput.append(el("div", { class: "banner banner-bad" }, (res && res.error) || "Rebuild failed. Check the server terminal and try again."));
  }
}

function homeStat(num, label) {
  return el("div", { class: "stat" }, [el("div", { class: "stat-num", text: String(num) }), el("div", { class: "stat-label", text: label })]);
}

/** A real problem still warrants a banner; ordinary index state (indexing / stale / up-to-date)
 * is handled by the ambient .index-status readout rendered separately, not a rebuild button. */
function homeStatusBanner(health) {
  if (health.problem) return el("div", { class: "banner banner-warn" }, String(health.problem));
  return null;
}

/** Genuinely useful onboarding for a workspace with nothing imported yet — the real CLI command,
 * not a bare empty state. */
function homeOnboarding(health) {
  const ws = (health && health.workspace) || "./workspace";
  return stateBlock({
    variant: "empty-index",
    title: "No conversations indexed yet.",
    body: "Browse chatgpt.com with the server running — new conversations you open are captured and indexed automatically, no export needed. Optional: import a ChatGPT export to backfill your existing history using the command below.",
    code: `convsearch import <export.zip> -w ${ws}`,
    actions: [{ label: "Setup help", onClick: () => setView("status") }],
  });
}

async function loadHome() {
  const seq = ++homeSeq;
  loadingInto(homeOutput, 3);

  const [statusRes, learnRes, projectsRes, suggestRes] = await Promise.all([
    refreshConn(),
    ask({ type: "convsearch:learnStats" }),
    ask({ type: "convsearch:projects" }),
    ask({ type: "convsearch:suggestions", params: { limit: 8 } }),
  ]);
  if (seq !== homeSeq) return;

  const online = statusRes && statusRes.online;
  const health = (statusRes && statusRes.health) || {};
  const serverUrl = (statusRes && statusRes.serverUrl) || "http://127.0.0.1:8756";

  clear(homeOutput);

  if (!online) {
    homeOutput.append(el("div", { class: "banner banner-bad" }, `Server offline — not reachable at ${serverUrl}.`));
    homeOutput.append(
      stateBlock({
        variant: "offline",
        title: "Start the server",
        body: "convsearch runs against a local server that isn't responding. Start it with the launcher — or run the command below in a terminal from the convsearch folder — then retry.",
        code: "convsearch serve --workspace ./workspace",
        actions: [{ label: "Retry", primary: true, onClick: loadHome }],
      })
    );
    return;
  }

  const banner = homeStatusBanner(health);
  if (banner) homeOutput.append(banner);
  // Ambient auto-index readout (quiet rebuild recovery only when stale).
  homeOutput.append(indexStatusEl(health, homeRebuildIndex));

  // capture-queue health — silent when empty and healthy, same rule as Status.
  const queueState = await readQueueState();
  if (seq !== homeSeq) return;
  const pendingCaptures = statusRes.pending || 0;
  if (pendingCaptures > 0 || queueState.offline || (Number(queueState.failures) || 0) > 0) {
    homeOutput.append(captureQueueBanner(pendingCaptures, queueState));
  }

  const conversations = typeof health.conversations === "number" ? health.conversations : null;
  if (conversations === 0) homeOutput.append(homeOnboarding(health));

  // stat tiles — only for fields the server actually sent.
  const learnStats = (learnRes && learnRes.ok && learnRes.data && learnRes.data.stats) || {};
  const stats = el("div", { class: "stat-grid" });
  let anyStat = false;
  const addStat = (value, label) => {
    if (typeof value !== "number") return;
    stats.append(homeStat(value, label));
    anyStat = true;
  };
  addStat(health.conversations, "conversations");
  addStat(health.messages, "messages");
  addStat(health.captured_conversations, "captured live");
  addStat(learnStats.learned_preferences, "learned preferences");
  addStat(learnStats.search, "searches logged");
  if (anyStat) homeOutput.append(stats);

  // top projects
  const projects = (projectsRes && projectsRes.ok && projectsRes.data && projectsRes.data.projects) || [];
  if (projects.length) {
    const sec = el("div", { class: "dash-section" });
    sec.append(el("h2", { class: "section-head" }, ["Top projects", el("span", { class: "section-count", text: String(projects.length) })]));
    const list = el("div", { class: "dash-block" });
    for (const p of projects.slice(0, 5)) {
      const item = el("div", { class: "list-item", role: "button", tabindex: "0" });
      item.append(el("div", { class: "li-title", text: p.name }));
      const meta = el("div", { class: "li-meta" });
      meta.append(el("span", { text: plural(p.conversation_count || 0, "conversation") }));
      meta.append(el("span", { text: `${p.open_task_count || 0} open` }));
      if (p.last_activity) meta.append(el("span", { text: timeAgo(p.last_activity) || fmtDate(p.last_activity) }));
      item.append(meta);
      const open = () => {
        setView("projects");
        openProject(p.name);
      };
      item.addEventListener("click", open);
      item.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      });
      list.append(item);
    }
    sec.append(list);
    homeOutput.append(sec);
  }

  // suggested searches — server /suggestions recent (this user's own past queries) then
  // popular, de-duplicated, recent first.
  const recent = (suggestRes && suggestRes.ok && suggestRes.data && suggestRes.data.recent) || [];
  const popular = (suggestRes && suggestRes.ok && suggestRes.data && suggestRes.data.popular) || [];
  const seenQ = new Set();
  const merged = [];
  for (const it of [...recent, ...popular]) {
    const q = String(Array.isArray(it) ? it[0] : it || "").trim();
    if (!q || seenQ.has(q.toLowerCase())) continue;
    seenQ.add(q.toLowerCase());
    merged.push(q);
  }
  if (merged.length) {
    const sec = el("div", { class: "dash-section" });
    sec.append(el("h2", { class: "section-head" }, "Suggested searches"));
    const chips = el("div", { class: "suggest-chips" });
    for (const q of merged.slice(0, 8)) {
      const btn = el("button", { type: "button", class: "suggest-chip", text: q, title: q });
      btn.addEventListener("click", () => {
        setView("search");
        searchInput.value = q;
        focusSearch();
        runSearch({ force: true });
      });
      chips.append(btn);
    }
    sec.append(chips);
    homeOutput.append(sec);
  }

  // quick actions — every button does something real today.
  const actions = el("div", { class: "result-actions" });
  actions.append(el("button", { type: "button", class: "btn btn-primary btn-sm", text: "Search", onclick: () => { setView("search"); focusSearch({ select: true }); } }));
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Ask", onclick: () => { setView("search"); focusSearch({ select: true }); } }));
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "View projects", onclick: () => setView("projects") }));
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "View memories", onclick: () => setView("memories") }));
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Run diagnostics", onclick: () => setView("status") }));
  actions.append(el("button", { type: "button", class: "btn btn-sm", text: "Setup help", onclick: () => setView("status") }));
  homeOutput.append(el("div", { class: "dash-section" }, [el("h2", { class: "section-head" }, "Quick actions"), actions]));
}

/* ------------------------------------------------------------------ */
/* HOME · universal command bar                                       */
/* ------------------------------------------------------------------ */

/*
 * Local-only parsing — no LLM, no new endpoint. Every recognised verb maps to an action that
 * already exists elsewhere in this file; unrecognised structured input (a known verb used
 * wrong) gets a helpful inline list of commands rather than a silent no-op.
 */
const cmdInput = $("cmd-input");
const cmdHintEl = $("cmd-hint");
const COMMAND_HELP =
  "search <query> · ask <question> · find/show memories <query> · project <name> · " +
  "tasks/open tasks · timeline <topic> · captures · review · privacy · " +
  "rebuild index · diagnostics/status · help";
const CMD_KNOWN_VERBS = new Set([
  "search",
  "ask",
  "find",
  "show",
  "project",
  "open",
  "rebuild",
  "diagnostics",
  "status",
  "help",
  "tasks",
  "timeline",
  "captures",
  "capture",
  "review",
  "privacy",
]);

function parseCommand(raw) {
  const text = String(raw || "").trim().replace(/\s+/g, " ");
  if (!text) return null;
  const lower = text.toLowerCase();
  let m;

  if ((m = /^search\s+(.+)$/i.exec(text))) return { kind: "search", query: m[1].trim(), hint: `→ search: ${m[1].trim()}` };
  if ((m = /^ask\s+(.+)$/i.exec(text))) return { kind: "ask", query: m[1].trim(), hint: `→ ask: ${m[1].trim()}` };
  if ((m = /^(?:find|show)\s+memories(?:\s+(.+))?$/i.exec(text))) {
    const q = (m[1] || "").trim();
    return { kind: "memories", query: q, hint: q ? `→ memories: ${q}` : "→ memories" };
  }
  if ((m = /^(?:open\s+project|project)\s+(.+)$/i.exec(text))) return { kind: "project", name: m[1].trim(), hint: `→ project: ${m[1].trim()}` };
  if ((m = /^timeline\s+(.+)$/i.exec(text))) return { kind: "timeline", query: m[1].trim(), hint: `→ timeline: ${m[1].trim()}` };
  if (lower === "tasks" || lower === "open tasks" || lower === "show open tasks") return { kind: "tasks", hint: "→ tasks" };
  if (lower === "captures" || lower === "capture inbox") return { kind: "captures", hint: "→ captures" };
  if (lower === "review" || lower === "memories review") return { kind: "review", hint: "→ review" };
  if (lower === "privacy") return { kind: "privacy", hint: "→ privacy" };
  if (lower === "rebuild index") return { kind: "reindex", hint: "→ rebuild index" };
  if (lower === "diagnostics" || lower === "status") return { kind: "status", hint: "→ status" };
  if (lower === "help") return { kind: "help", hint: "→ help" };

  const firstWord = lower.split(" ")[0];
  if (CMD_KNOWN_VERBS.has(firstWord)) return { kind: "invalid", hint: `→ unrecognized command: "${text}"` };

  return { kind: "search", query: text, hint: `→ search: ${text}` };
}

function cmdHint(text, isError) {
  cmdHintEl.textContent = text || "";
  cmdHintEl.classList.toggle("is-error", Boolean(isError));
}

async function runCommandBar() {
  const parsed = parseCommand(cmdInput.value);
  if (!parsed) {
    cmdHint("");
    return;
  }
  cmdHint(parsed.hint);
  switch (parsed.kind) {
    case "search":
      setView("search");
      searchInput.value = parsed.query;
      focusSearch();
      runSearch({ force: true });
      break;
    case "ask":
      setView("search");
      searchInput.value = parsed.query;
      focusSearch();
      runSearch({ force: true });
      runAsk();
      break;
    case "memories":
      setView("memories");
      $("mem-query").value = parsed.query;
      loadMemories();
      $("mem-query").focus();
      break;
    case "project":
      setView("projects");
      openProject(parsed.name);
      break;
    case "tasks": {
      setView("tasks");
      const openRadio = document.querySelector('input[name="task-state"][value="open"]');
      if (openRadio) openRadio.checked = true;
      loadTasks();
      break;
    }
    case "timeline":
      setView("timeline");
      $("timeline-query").value = parsed.query;
      updateTimelineHint();
      runTimeline();
      break;
    case "captures":
      setView("captures");
      break;
    case "review":
      setView("review");
      break;
    case "privacy":
      setView("privacy");
      break;
    case "reindex":
      cmdHint("→ rebuild index — starting…");
      await homeRebuildIndex();
      cmdHint("→ rebuild index — done. See Home or Status for details.");
      break;
    case "status":
      setView("status");
      break;
    case "help":
      cmdHint(`Commands: ${COMMAND_HELP}`);
      break;
    case "invalid":
      cmdHint(`${parsed.hint}. Commands: ${COMMAND_HELP}`, true);
      break;
  }
}

$("cmd-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runCommandBar();
});
cmdInput.addEventListener("input", () => {
  const parsed = parseCommand(cmdInput.value);
  cmdHint(parsed ? parsed.hint : "");
});
cmdInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    e.preventDefault();
    cmdInput.value = "";
    cmdHint("");
  }
});

/* ================================================================== */
/* global keyboard                                                    */
/* ================================================================== */

function isTextField(node) {
  if (!node) return false;
  const tag = node.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || node.isContentEditable === true;
}

function focusSearch({ select = false } = {}) {
  searchInput.focus();
  if (select) searchInput.select();
}

document.addEventListener("keydown", (e) => {
  if (e.isComposing) return;
  const key = e.key;
  const inText = isTextField(e.target);

  // "/" focuses search from anywhere outside a text field
  if (key === "/" && !inText && !e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    if (currentView !== "search") setView("search");
    focusSearch({ select: true });
    return;
  }

  if (currentView !== "search") return;

  if (key === "ArrowDown") {
    if (moveSearchSelection(1)) {
      e.preventDefault();
      focusSearch();
      return;
    }
    if (e.target === searchInput && focusFirstSuggestion()) {
      e.preventDefault();
    }
    return;
  }
  if (key === "ArrowUp") {
    if (moveSearchSelection(-1)) {
      e.preventDefault();
      focusSearch();
    }
    return;
  }
  if (key === "Enter" && e.target === searchInput) {
    if (searchSelected >= 0 && openSearchSelected()) {
      e.preventDefault();
      return;
    }
  }
  if (key === "Escape") {
    if (searchInput.value !== "") {
      e.preventDefault();
      searchInput.value = "";
      searchSeq += 1;
      searchLastKey = null;
      searchStatus("");
      renderSearchIdle();
      focusSearch();
      searchFocused = true;
      loadSuggestions();
    }
  }
});

/* ================================================================== */
/* boot                                                               */
/* ================================================================== */

async function boot() {
  initTheme();
  renderSearchIdle();
  refreshConn();

  // A context-menu selection is stashed by the SW just before it opens the panel.
  chrome.storage.local.get({ pendingQuery: null }, (stored) => {
    const pending = stored && stored.pendingQuery;
    if (pending && pending.text && Date.now() - (pending.at || 0) < PENDING_MAX_AGE_MS) {
      chrome.storage.local.remove("pendingQuery");
      setView("search");
      searchInput.value = pending.text;
      runSearch({ force: true });
      focusSearch();
    } else {
      if (pending) chrome.storage.local.remove("pendingQuery");
      setView("home");
      $("cmd-input").focus();
    }
  });
}

boot();
