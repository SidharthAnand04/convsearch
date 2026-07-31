"use strict";

const DEFAULT_SERVER = "http://127.0.0.1:8756";

const form = document.getElementById("options-form");
const input = document.getElementById("server-url");
const captureToggle = document.getElementById("capture-enabled");
const shortcutsButton = document.getElementById("open-shortcuts");
const status = document.getElementById("status");

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

chrome.storage.local.get(
  { serverUrl: DEFAULT_SERVER, captureEnabled: true },
  ({ serverUrl, captureEnabled }) => {
    input.value = serverUrl || DEFAULT_SERVER;
    captureToggle.checked = captureEnabled !== false;
  }
);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = input.value.trim().replace(/\/+$/, "");
  let url;
  try {
    url = new URL(value);
  } catch {
    setStatus("That is not a valid URL.", true);
    return;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    setStatus("Use an http:// address.", true);
    return;
  }
  if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
    setStatus("Only 127.0.0.1 and localhost are permitted.", true);
    return;
  }
  chrome.storage.local.set({ serverUrl: value }, () => {
    setStatus("Saved.");
  });
});

// A chrome:// URL cannot be a link target from an extension page, but tabs.create may
// navigate there, which is the only way to reach the shortcut editor.
shortcutsButton.addEventListener("click", () => {
  if (chrome.tabs && chrome.tabs.create) {
    chrome.tabs.create({ url: "chrome://extensions/shortcuts" });
  } else {
    setStatus("Open chrome://extensions/shortcuts manually to change the shortcut.", true);
  }
});

captureToggle.addEventListener("change", () => {
  const captureEnabled = captureToggle.checked;
  chrome.storage.local.set({ captureEnabled }, () => {
    setStatus(
      captureEnabled
        ? "Capture on — open chatgpt.com and browse to add conversations."
        : "Capture off — nothing new will be saved until you turn this back on."
    );
  });
});
