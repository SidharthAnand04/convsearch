#!/usr/bin/env python3
"""Chrome Native Messaging host for convsearch.

Chrome's MV3 service worker cannot launch a local process. This tiny host program
*can*: Chrome is permitted to start it (once it is registered — see the installers in
this directory), and it speaks the Native Messaging stdio protocol to the extension.
Its whole job is to (a) report whether the local `convsearch serve` HTTP server is up
and (b) start it, detached, when it is not.

Wire protocol (both directions, on stdin/stdout):
    [4 bytes: uint32 little-endian message length][that many bytes of UTF-8 JSON]

Only the standard library is used, so the host runs under whatever Python the
installer resolved (bare `python`, or `uv run` which provides its own interpreter).

Configuration is read from a sibling file written by the installer:
    scripts/native-host/host-config.json
        {
          "command":   ["uv", "run", "--no-sync", "convsearch"],  # argv prefix
          "workspace": "<abs path>",
          "port":      8756,
          "log":       "<abs log path>"
        }

Robustness contract: this process must NEVER write anything to stdout that is not a
correctly framed message — a stray print would corrupt the protocol and break the
extension. Every code path therefore funnels through send_message(); errors are logged
to the configured log file and also returned as framed JSON error objects.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import UTC, datetime
from typing import Any

# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #

# scripts/native_host.py -> repo root is two levels up from this file's dir.
_HOST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HOST_DIR)
CONFIG_PATH = os.path.join(_HOST_DIR, "native-host", "host-config.json")

# Fallback log location, used only if we cannot even read the config (which is where
# the real log path lives). Kept inside the repo's tmp/ so it is easy to find.
_FALLBACK_LOG = os.path.join(REPO_ROOT, "tmp", "native-host.log")

# How long ensure_server waits for /health after a spawn before giving up.
_START_TIMEOUT_S = 25.0
_POLL_INTERVAL_S = 0.5
# Short timeout for the status probe so a dead port answers immediately.
_HEALTH_TIMEOUT_S = 1.5


# --------------------------------------------------------------------------- #
# logging                                                                      #
# --------------------------------------------------------------------------- #


def _log_path() -> str:
    """Best-effort resolution of the log file, tolerant of a missing/broken config."""
    try:
        cfg = _read_config()
        log = cfg.get("log")
        if isinstance(log, str) and log.strip():
            return log
    except Exception:
        pass
    return _FALLBACK_LOG


def log(message: str) -> None:
    """Append a timestamped line to the log file. Never raises — logging must not
    take the host down, and it must never touch stdout."""
    line = f"{datetime.now(UTC).isoformat()} {message}\n"
    try:
        path = _log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        # Truly nothing we can safely do here; swallow it rather than crash.
        pass


# --------------------------------------------------------------------------- #
# native messaging framing                                                     #
# --------------------------------------------------------------------------- #


def read_message() -> dict[str, Any] | None:
    """Read one framed message from stdin. Returns None at end-of-stream (Chrome
    closed the pipe), which is the normal signal to exit."""
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length or len(raw_length) < 4:
        return None
    (length,) = struct.unpack("<I", raw_length)
    if length == 0:
        return {}
    payload = sys.stdin.buffer.read(length)
    if len(payload) < length:
        return None
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:  # malformed input — report, don't crash
        log(f"read_message: could not decode payload: {exc!r}")
        return {"__decode_error__": str(exc)}


def send_message(obj: dict[str, Any]) -> None:
    """Write one framed message to stdout. This is the ONLY function permitted to
    write to stdout; everything the extension sees goes through here."""
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #


def _read_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_config() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load and validate host-config.json.

    Returns (config, error_response). On success error_response is None; on failure
    config is None and error_response is a framed-ready error object explaining what
    to do (usually: run the installer)."""
    if not os.path.exists(CONFIG_PATH):
        return None, {
            "ok": False,
            "error": "config_missing",
            "detail": (
                f"host config not found at {CONFIG_PATH}. Run the installer "
                "(scripts/install-native-host.ps1 or .sh) to create it."
            ),
        }
    try:
        cfg = _read_config()
    except Exception as exc:
        return None, {
            "ok": False,
            "error": "config_unreadable",
            "detail": f"could not parse {CONFIG_PATH}: {exc}",
        }

    command = cfg.get("command")
    if not isinstance(command, list) or not command:
        return None, {
            "ok": False,
            "error": "config_invalid",
            "detail": "host config 'command' must be a non-empty array",
        }
    workspace = cfg.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        return None, {
            "ok": False,
            "error": "config_invalid",
            "detail": "host config 'workspace' must be an absolute path string",
        }
    # Port defaults to the project standard if the installer omitted it.
    cfg.setdefault("port", 8756)
    return cfg, None


# --------------------------------------------------------------------------- #
# health probe                                                                 #
# --------------------------------------------------------------------------- #


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def probe_health(port: int) -> tuple[bool, dict[str, Any] | None]:
    """GET /health. Returns (running, parsed_body_or_None). A refused connection or
    timeout simply means the server is not up — that is the normal, expected case."""
    url = health_url(port)
    try:
        with urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_S) as response:
            body = response.read()
            if response.status != 200:
                return False, None
            try:
                return True, json.loads(body.decode("utf-8"))
            except Exception:
                # Reachable but unparseable body still counts as "running".
                return True, None
    except Exception:
        return False, None


# --------------------------------------------------------------------------- #
# server launch                                                                #
# --------------------------------------------------------------------------- #


def spawn_server(cfg: dict[str, Any]) -> int | None:
    """Launch `<command> serve -w <workspace> --port <port>` fully detached so it
    OUTLIVES this host process (Chrome kills the host when the port closes). Returns
    the child PID, or None if the spawn itself failed."""
    command = list(cfg["command"])
    workspace = cfg["workspace"]
    port = int(cfg.get("port", 8756))
    argv = [*command, "serve", "-w", workspace, "--port", str(port)]

    log(f"spawn_server: launching {argv} (cwd={REPO_ROOT})")

    # Send the server's own stdout/stderr to the log file so a crash-on-start is
    # diagnosable. Opened append so successive launches accumulate.
    log_file_path = cfg.get("log") or _FALLBACK_LOG
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    popen_kwargs: dict[str, Any] = {
        "cwd": REPO_ROOT,
        "stdin": subprocess.DEVNULL,
    }
    try:
        # Deliberately not a context manager: this handle is inherited by the detached
        # child as its stdout/stderr and must stay open after this function returns.
        out = open(log_file_path, "a", encoding="utf-8")  # noqa: SIM115
        popen_kwargs["stdout"] = out
        popen_kwargs["stderr"] = out
    except Exception:
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL

    if os.name == "nt":
        # DETACHED_PROCESS: no console; CREATE_NEW_PROCESS_GROUP: not killed when the
        # host's group is signalled. Together they let the server survive host exit.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        # POSIX: a new session detaches the child from the host's process group and
        # controlling terminal, so it keeps running after the host exits.
        popen_kwargs["start_new_session"] = True

    try:
        child = subprocess.Popen(argv, **popen_kwargs)
        return child.pid
    except Exception as exc:
        log(f"spawn_server: failed to launch: {exc!r}\n{traceback.format_exc()}")
        return None


# --------------------------------------------------------------------------- #
# request handlers                                                             #
# --------------------------------------------------------------------------- #


def handle_status(cfg: dict[str, Any]) -> dict[str, Any]:
    port = int(cfg.get("port", 8756))
    running, health = probe_health(port)
    return {
        "ok": True,
        "running": running,
        "url": health_url(port),
        "health": health,
    }


def handle_ensure_server(cfg: dict[str, Any]) -> dict[str, Any]:
    port = int(cfg.get("port", 8756))

    # Guard against launching twice: if the server already answers, do nothing.
    running, health = probe_health(port)
    if running:
        return {
            "ok": True,
            "running": True,
            "started": False,
            "url": health_url(port),
            "pid": None,
            "detail": "server already running",
        }

    pid = spawn_server(cfg)
    if pid is None:
        return {
            "ok": False,
            "running": False,
            "started": False,
            "url": health_url(port),
            "pid": None,
            "error": "spawn_failed",
            "detail": "could not launch convsearch serve; see the native-host log",
        }

    # Poll /health until the server answers or we time out. The server has to load
    # models on first start, so this can legitimately take a few seconds.
    deadline = time.monotonic() + _START_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        running, health = probe_health(port)
        if running:
            log(f"ensure_server: server up (pid={pid})")
            return {
                "ok": True,
                "running": True,
                "started": True,
                "url": health_url(port),
                "pid": pid,
                "health": health,
                "detail": "server started",
            }

    log(f"ensure_server: timed out waiting for health (pid={pid})")
    return {
        "ok": True,
        "running": False,
        "started": True,
        "url": health_url(port),
        "pid": pid,
        "detail": (
            "launched convsearch serve but /health did not respond within "
            f"{int(_START_TIMEOUT_S)}s; it may still be loading — check the log"
        ),
    }


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    """Route a single request to its handler, loading config lazily so a status/ensure
    request with a missing config returns a clean error instead of crashing."""
    action = request.get("action") if isinstance(request, dict) else None

    if action not in ("status", "ensure_server"):
        return {
            "ok": False,
            "error": "unknown_action",
            "detail": f"unsupported action: {action!r}",
        }

    cfg, err = load_config()
    if err is not None:
        return err

    assert cfg is not None
    try:
        if action == "status":
            return handle_status(cfg)
        return handle_ensure_server(cfg)
    except Exception as exc:  # never let a handler bubble raw text to stdout
        log(f"dispatch: handler for {action} raised: {exc!r}\n{traceback.format_exc()}")
        return {"ok": False, "error": "handler_exception", "detail": str(exc)}


# --------------------------------------------------------------------------- #
# main loop                                                                    #
# --------------------------------------------------------------------------- #


def main() -> int:
    log("native host started")
    try:
        while True:
            request = read_message()
            if request is None:
                # Chrome closed the pipe: normal shutdown.
                log("native host: stdin closed, exiting")
                return 0
            try:
                response = dispatch(request)
            except Exception as exc:
                log(f"main: dispatch failed: {exc!r}\n{traceback.format_exc()}")
                response = {"ok": False, "error": "internal_error", "detail": str(exc)}
            send_message(response)
    except Exception as exc:
        # Last-resort guard: log, try to emit one framed error, then exit non-zero.
        log(f"native host: fatal: {exc!r}\n{traceback.format_exc()}")
        with contextlib.suppress(Exception):
            send_message({"ok": False, "error": "fatal", "detail": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
