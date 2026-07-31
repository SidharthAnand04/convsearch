# scripts/

Four unrelated toolchains share this flat folder. Grouped by the workflow they belong to.

## Run it

`convsearch-up.sh` / `convsearch-up.ps1` — init, import, index and serve in one command.

```bash
bash scripts/convsearch-up.sh [WORKSPACE] [CHATGPT_EXPORT_ZIP]
```

Workspace defaults to `./workspace`. The export ZIP is optional and may instead come from
`CONVSEARCH_EXPORT_ZIP` or `CHATGPT_EXPORT_ZIP` (`convsearch-up.sh:3-9`); without it you get an
empty initialized workspace. In a checkout it prefers `uv run --no-sync convsearch`, falling
back to a bare `convsearch` on PATH — `--no-sync` so it does not block on the uv lock while an
earlier server is still running (`convsearch-up.sh:14-19`).

## Auto-start

`install-native-host.ps1` / `install-native-host.sh`, `native_host.py`, `native-host/`. One-time
registration that lets the extension start the server itself. See
[`native-host/README.md`](native-host/README.md).

## Ship it

`package-extension.sh` / `package-extension.ps1` — builds
`dist/convsearch-extension-v<version>.zip`, reading the version from `extension/manifest.json`
and archiving the *contents* of `extension/` so `manifest.json` lands at the ZIP root, as the
Chrome Web Store requires.

## Dev

- `bench.py` — capture, index, search and first-search latency at several corpus sizes
  (default `10,50,200`), so scaling problems surface instead of hiding behind a toy dataset.
  Uses a disposable workspace under `tmp/bench-ws` and port 8801, never the default 8756.
  Deterministic embeddings by default; `--real-model` pulls in sentence-transformers for
  production-like numbers (`bench.py:1-20`).
- `live.js` — `npm run live`, opens the real chatgpt.com with the extension loaded and a
  persisted profile. A manual tool, not a test.

## Generated files — do not edit or commit

`native-host/host-config.json`, `native-host/convsearch-host.bat` (Windows),
`native-host/convsearch-host.sh` (POSIX) and `native-host/com.convsearch.host.json` are all
written by the installer and are overwritten on every re-run. Only the `.template` files in
`native-host/` are source. Note that the POSIX launcher has no template — `install-native-host.sh`
generates it inline (`:109-120`).
