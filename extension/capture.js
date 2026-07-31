"use strict";

/**
 * Pure extraction of the conversation currently rendered on chatgpt.com.
 *
 * This file must stay free of chrome.* APIs and of any DOM globals it does not
 * receive as arguments: the Playwright suite loads it into a plain fixture page
 * and calls convsearchExtract(document, location.href) directly.
 *
 * Content scripts are not ES modules, so the entry point hangs off globalThis.
 */

/** ChatGPT conversation URLs are /c/<uuid>; anything else is not a conversation. */
const CONVERSATION_PATH = /\/c\/([0-9a-fA-F-]{8,})/;

/** document.title is "<title> - ChatGPT" (some builds use an en dash). */
const TITLE_SUFFIX = /\s*[-–—]\s*ChatGPT\s*$/;

/** Turn containers, in priority order (contract selector table). */
const TURN_SELECTORS = ['[data-testid^="conversation-turn"]', "[data-message-author-role]"];

/**
 * innerText is what we want (it collapses the way the user sees the message and
 * skips hidden nodes), but it is layout-dependent and undefined in some
 * non-browser DOMs, so fall back to textContent.
 */
function readText(element) {
  if (!element) return "";
  const value = typeof element.innerText === "string" ? element.innerText : element.textContent;
  return typeof value === "string" ? value.trim() : "";
}

function attr(element, name) {
  if (!element || typeof element.getAttribute !== "function") return null;
  const value = element.getAttribute(name);
  return typeof value === "string" && value.length ? value : null;
}

/** `location.pathname` matching /c/<uuid>; null (skip capture) if it does not. */
function conversationIdFromUrl(url) {
  if (typeof url !== "string" || !url) return null;
  let pathname = url;
  try {
    // url may be a full href or a bare pathname; URL() handles only the former.
    pathname = new URL(url, "https://chatgpt.com").pathname;
  } catch {
    /* fall through and match against the raw string */
  }
  const match = CONVERSATION_PATH.exec(pathname);
  return match ? match[1] : null;
}

/**
 * Title: the active sidebar link's text, else document.title minus " - ChatGPT".
 * The link whose href points at this conversation *is* the active sidebar entry,
 * which is far more stable than whatever class ChatGPT uses for "selected".
 */
function extractTitle(doc, conversationId) {
  const candidates = [
    `nav a[href$="/c/${conversationId}"]`,
    `a[href$="/c/${conversationId}"]`,
    'nav a[aria-current="page"]',
    'a[aria-current="page"]',
  ];
  for (const selector of candidates) {
    let link = null;
    try {
      link = doc.querySelector(selector);
    } catch {
      continue; // malformed id would make an invalid selector; just move on
    }
    const text = readText(link);
    if (text) return text;
  }
  const documentTitle = typeof doc.title === "string" ? doc.title : "";
  const stripped = documentTitle.replace(TITLE_SUFFIX, "").trim();
  return stripped || null;
}

/** All turn elements, using the first selector that matches anything. */
function collectTurns(doc) {
  for (const selector of TURN_SELECTORS) {
    let nodes = [];
    try {
      nodes = Array.from(doc.querySelectorAll(selector));
    } catch {
      continue;
    }
    if (nodes.length) return nodes;
  }
  return [];
}

/**
 * The role attribute may live on the turn element itself or on a descendant
 * (ChatGPT wraps each turn around an inner [data-message-author-role] node).
 */
function findRoleElement(turn) {
  if (attr(turn, "data-message-author-role")) return turn;
  if (typeof turn.querySelector !== "function") return null;
  return turn.querySelector("[data-message-author-role]");
}

/** Message text: `.markdown` innerText, else the whole element's innerText. */
function extractTurnText(turn, roleElement) {
  const scope = roleElement || turn;
  if (typeof scope.querySelector === "function") {
    const markdown = scope.querySelector(".markdown");
    const markdownText = readText(markdown);
    if (markdownText) return markdownText;
  }
  return readText(scope);
}

function extractMessage(turn, conversationId, index) {
  const roleElement = findRoleElement(turn);
  const role = attr(roleElement, "data-message-author-role");
  if (!role) return null; // no role → skip this turn

  const text = extractTurnText(turn, roleElement);
  if (!text) return null; // empty turn (e.g. a placeholder while streaming starts)

  const sourceMessageId =
    attr(roleElement, "data-message-id") ||
    attr(turn, "data-message-id") ||
    `${conversationId}-${index}`;

  return {
    source_message_id: sourceMessageId,
    role: role,
    text: text,
    order: 0, // assigned after filtering so orders stay contiguous
    created_at: null,
  };
}

/**
 * Extract the conversation rendered in `doc`.
 *
 * @param {Document} doc  the page document
 * @param {string} url    the page URL (href or pathname)
 * @returns {object|null} conversation object, or null when this page is not a
 *                        usable conversation. Never throws.
 */
function convsearchExtract(doc, url) {
  try {
    if (!doc || typeof doc.querySelectorAll !== "function") return null;

    const conversationId = conversationIdFromUrl(url);
    if (!conversationId) return null;

    const messages = [];
    const turns = collectTurns(doc);
    for (let index = 0; index < turns.length; index += 1) {
      const message = extractMessage(turns[index], conversationId, index);
      if (!message) continue;
      message.order = messages.length;
      messages.push(message);
    }
    if (!messages.length) return null; // nothing usable → skip the conversation

    return {
      source_conversation_id: conversationId,
      title: extractTitle(doc, conversationId),
      created_at: null,
      updated_at: new Date().toISOString(),
      messages: messages,
    };
  } catch {
    // An unfamiliar page shape must never surface as an exception on chatgpt.com.
    return null;
  }
}

globalThis.convsearchExtract = convsearchExtract;
