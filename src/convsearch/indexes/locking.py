"""Cross-process advisory locks over a workspace's vector index files.

`AutoIndexer.lock` is a `threading.Lock`: it serialises threads inside one server process and
nothing else. Two convsearch processes on the same workspace — a second `convsearch serve` on
another port, or a hand-run `convsearch index` while a server's auto-indexer is mid-pass —
would otherwise read-modify-write `passages.faiss` and `passage_vectors.json` concurrently.
The last writer wins and silently discards the other's vectors, even though that process
already committed matching `embedding_records` rows, so those conversations stop being
searchable with no error anywhere.

Two locks, deliberately separated so no one waits longer than they must:

* **mutate** wraps a whole index pass (snapshot → encode → write). It makes the
  read-modify-write atomic across processes, so an append can never be lost. It can be held
  for as long as encoding takes, which is why readers do not take it.
* **swap** wraps only the file replacement itself — a few milliseconds — and is also taken by
  searches. The index file and the map file are separate files replaced one after the other,
  so without it a concurrent reader could pair a *new* index with an *old* map. After a full
  rebuild the map is reordered, so that pairing returns the WRONG passage for a query, which
  is far worse than an error.

The OS drops both locks when the holding process dies, so there is no stale-PID problem and
nothing to clean up after a crash.

Locks are reentrant within a process: `update_indexes` calls `build_indexes`, which takes the
same `mutate` lock. Reentrancy is tracked per lock file, and threads of the same process
serialise on a `threading.RLock` before ever touching the OS lock (a second OS-level lock
request from the same process on a different file descriptor would otherwise deadlock).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

MUTATE_LOCK = "mutate"
SWAP_LOCK = "swap"

_POLL_SECONDS = 0.05


class IndexLockTimeout(RuntimeError):
    """Another process held an index lock too long. The message names what to do about it."""


@dataclass
class _Held:
    rlock: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    fd: int | None = None


_registry: dict[str, _Held] = {}
_registry_guard = threading.Lock()


def lock_path(workspace: Path, name: str) -> Path:
    return workspace / "indexes" / f".index-{name}.lock"


def _entry(key: str) -> _Held:
    with _registry_guard:
        held = _registry.get(key)
        if held is None:
            held = _Held()
            _registry[key] = held
        return held


def _try_lock(fd: int) -> bool:
    """Take an exclusive OS lock on byte 0 without blocking. False when someone else has it."""
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)


def _open_lock_file(path: Path) -> int | None:
    """Open the lock file, creating it if needed. None when the workspace is not writable.

    Every search takes the swap lock, so the common case — the file already exists — must be
    a single syscall. Doing `mkdir(exist_ok=True)` first cost about as much again (measured
    ~0.7ms each on this Windows filesystem), for a directory that is there on all but the
    very first call.
    """
    try:
        return os.open(path, os.O_RDWR)
    except FileNotFoundError:
        pass
    except OSError:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None
    try:
        if os.fstat(fd).st_size == 0:
            # msvcrt.locking() locks a byte range from the current offset; give it a byte to
            # lock. Best effort: if another process got there first the byte already exists.
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        pass
    return fd


def _wait_for_lock(fd: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if _try_lock(fd):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)


def _busy_message(path: Path, name: str, timeout: float) -> str:
    return (
        f"another convsearch process has held the {name!r} index lock on this workspace for "
        f"more than {timeout:g}s ({path}). Only one process may index a workspace at a time: "
        "stop the other `convsearch serve` or `convsearch index` and try again."
    )


@contextmanager
def index_lock(
    workspace: Path,
    name: str,
    *,
    timeout: float = 120.0,
    required: bool = True,
) -> Iterator[bool]:
    """Hold the named cross-process index lock for the duration of the block.

    Yields True when the lock is actually held. With `required=False` a lock that cannot be
    taken (read-only workspace, deleted directory, contention past `timeout`) yields False
    instead of raising — used by searches, which must degrade rather than fail. Writers pass
    `required=True` and get `IndexLockTimeout`, whose message names the fix.
    """
    path = lock_path(workspace, name)
    key = os.path.normcase(os.path.abspath(path))
    held = _entry(key)
    if not held.rlock.acquire(timeout=timeout):
        if not required:
            yield False
            return
        raise IndexLockTimeout(_busy_message(path, name, timeout))
    try:
        if held.depth > 0:
            # Reentrant: this thread already owns the OS lock (update_indexes -> build_indexes).
            held.depth += 1
            try:
                yield True
            finally:
                held.depth -= 1
            return
        fd = _open_lock_file(path)
        if fd is None:
            if not required:
                yield False
                return
            raise IndexLockTimeout(
                f"cannot create the index lock file {path}. Check that the workspace still "
                "exists and is writable."
            )
        try:
            if not _wait_for_lock(fd, timeout):
                if not required:
                    yield False
                    return
                raise IndexLockTimeout(_busy_message(path, name, timeout))
            held.fd = fd
            held.depth = 1
            try:
                yield True
            finally:
                held.depth = 0
                held.fd = None
                _unlock(fd)
        finally:
            os.close(fd)
    finally:
        held.rlock.release()
