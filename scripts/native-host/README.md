# convsearch native messaging host

Lets the Chrome extension **auto-start the local `convsearch serve` server** whenever it
is offline. Chrome's MV3 service worker cannot launch a process, so a small registered
"native messaging host" (`native_host.py`, run through a generated launcher) does it.

There is a one-time registration step. After that the extension spins the server up on
its own.

## Prerequisites

- `convsearch` installed (in a checkout: `uv sync --extra ml --extra llm`).
- For a **fresh** workspace, import and index your history first, otherwise the server
  will start but serve empty results:

  ```
  convsearch init ./workspace
  convsearch import <your-ChatGPT-export.zip> -w ./workspace
  convsearch index -w ./workspace
  ```

## One-time install

1. Load the unpacked `extension/` folder in `chrome://extensions` (Developer mode on).
2. Copy the extension's **ID** from that page.
3. Run the installer for your OS from the repo root:

   **Windows**
   ```
   powershell -ExecutionPolicy Bypass -File scripts/install-native-host.ps1 -ExtensionId <id> -Workspace .\workspace
   ```

   **macOS / Linux**
   ```
   bash scripts/install-native-host.sh <id> ./workspace
   ```

4. **Reload the extension** in `chrome://extensions` (or restart the browser).

Thereafter, on browser startup / extension install and whenever a health check finds the
server down, the extension asks this host to start it — no terminal needed.

## What the installer creates

All under `scripts/native-host/`:

- `host-config.json` — how to run the server: `{command, workspace, port, log}`.
- `convsearch-host.bat` (Windows) / `convsearch-host.sh` (POSIX) — the launcher Chrome
  runs; it invokes `native_host.py` with the resolved interpreter.
- `com.convsearch.host.json` — the native messaging host manifest, rendered from
  `com.convsearch.host.json.template` with the launcher path and your extension ID.

Registration:

- **Windows** — registry values under
  `HKCU\Software\Google\Chrome\NativeMessagingHosts\com.convsearch.host` (and the Edge
  equivalent) pointing at the rendered manifest.
- **macOS / Linux** — the manifest is copied into Chrome's per-user
  `NativeMessagingHosts` directory.

## How it works

```
extension (background.js)
  --> chrome.runtime.connectNative("com.convsearch.host")
      --> convsearch-host.bat/.sh  --> native_host.py
          - {action:"status"}         -> GET /health, report running/health
          - {action:"ensure_server"}  -> if down, spawn `convsearch serve` DETACHED,
                                         poll /health up to ~25s, report started/pid
  <-- framed JSON response
```

The server is launched detached so it **outlives** the host (Chrome kills the host when
the message port closes). The host guards against double-launch: if `/health` already
answers, it does not spawn.

## Re-running / uninstall

The installer is idempotent — re-run it to change the workspace, port, or extension ID.
To uninstall, delete the registry keys (Windows) or the copied
`com.convsearch.host.json` (macOS/Linux).

## Troubleshooting

- Check `tmp/native-host.log` in the repo for host and server-start errors.
- If the extension reports the host is unavailable, confirm the ID you passed matches
  `chrome://extensions` and that you reloaded the extension after installing.
