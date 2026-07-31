"use strict";

/**
 * Content script glue for chatgpt.com.
 *
 * All extraction lives in capture.js (loaded first, in the same isolated world,
 * exposing globalThis.convsearchExtract). This file only decides *when* to
 * extract and hands the result to the background worker, which owns all HTTP.
 *
 * Rules of the house:
 *  - never modify the host page's DOM;
 *  - never let an exception escape into chatgpt.com.
 */

(() => {
  /** Quiet period after the last DOM mutation before we consider a turn settled. */
  const SETTLE_MS = 1500;
  /** How often we re-check the URL — the SPA navigates without a page load. */
  const URL_POLL_MS = 1000;
  /** Upper bound on how long streaming can defer a capture. */
  const MAX_DEFER_MS = 20000;

  let captureEnabled = true;
  let settleTimer = null;
  let deferSince = 0;
  let lastUrl = location.href;
  /** conversation id → hash of the last payload the background accepted. */
  const sentHashes = new Map();

  /** Small, fast, non-cryptographic string hash (FNV-1a, 32-bit, hex). */
  function hashString(value) {
    let hash = 0x811c9dc5;
    for (let i = 0; i < value.length; i += 1) {
      hash ^= value.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16);
  }

  /**
   * Hash only the content-bearing fields. `updated_at` is stamped at extraction
   * time, so including it would make every capture look new and defeat dedup.
   */
  function conversationHash(conversation) {
    return hashString(
      JSON.stringify({
        id: conversation.source_conversation_id,
        title: conversation.title,
        messages: conversation.messages.map((message) => [message.role, message.text]),
      })
    );
  }

  function sendCapture(conversation, hash) {
    try {
      chrome.runtime.sendMessage({ type: "convsearch:capture", conversation }, (response) => {
        // Reading lastError is what suppresses the "unchecked runtime.lastError"
        // console noise when the worker is asleep or the extension reloaded.
        if (chrome.runtime.lastError) return;
        // Only remember the hash on success, so a failed send retries later.
        if (response && response.ok) {
          sentHashes.set(conversation.source_conversation_id, hash);
        }
      });
    } catch {
      /* Extension context invalidated (reload/update); the next mutation retries. */
    }
  }

  function capture() {
    settleTimer = null;
    deferSince = 0;
    if (!captureEnabled) return;

    const extract = globalThis.convsearchExtract;
    if (typeof extract !== "function") return;

    const conversation = extract(document, location.href);
    if (!conversation) return; // not a conversation page, or nothing usable yet

    const hash = conversationHash(conversation);
    if (sentHashes.get(conversation.source_conversation_id) === hash) return;

    sendCapture(conversation, hash);
  }

  /**
   * Debounce: while tokens stream in, mutations fire continuously and each one
   * pushes the capture out. MAX_DEFER_MS caps that so a very long response (or a
   * page with a permanently animating element) still gets captured periodically
   * rather than never.
   */
  function scheduleCapture() {
    if (!captureEnabled) return;
    const now = Date.now();
    if (!deferSince) deferSince = now;
    if (settleTimer !== null) {
      if (now - deferSince >= MAX_DEFER_MS) return; // let the pending timer fire
      clearTimeout(settleTimer);
    }
    settleTimer = setTimeout(capture, SETTLE_MS);
  }

  function onUrlMaybeChanged() {
    if (location.href === lastUrl) return;
    lastUrl = location.href;
    // A navigation replaces the whole thread; capture the new one once it renders.
    scheduleCapture();
  }

  function start() {
    const observer = new MutationObserver(scheduleCapture);
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    setInterval(onUrlMaybeChanged, URL_POLL_MS);
    // History API navigations in the SPA also surface here on modern Chrome.
    window.addEventListener("popstate", onUrlMaybeChanged);

    scheduleCapture(); // first load
  }

  let started = false;
  function startOnce() {
    if (started) return;
    started = true;
    try {
      start();
    } catch {
      /* never break the host page */
    }
  }

  try {
    chrome.storage.local.get({ captureEnabled: true }, (stored) => {
      if (!chrome.runtime.lastError && stored) {
        captureEnabled = stored.captureEnabled !== false;
      }
      startOnce();
    });
  } catch {
    startOnce(); // storage unavailable: fall back to the default (enabled)
  }

  // Toggling capture in the options page takes effect without a reload.
  try {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== "local" || !changes.captureEnabled) return;
      captureEnabled = changes.captureEnabled.newValue !== false;
      if (captureEnabled) scheduleCapture();
    });
  } catch {
    /* storage.onChanged unavailable — capture still works, just not live-toggled */
  }
})();
