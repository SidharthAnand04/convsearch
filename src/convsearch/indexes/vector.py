"""The vector index on disk, and the map that makes it meaningful.

`vector_map_path` holds `{"passage_ids": [...]}` where a list **position is the FAISS vector
id**. Any drift between that list and the index returns the wrong passage for a query — a
plausible-looking but wrong answer, which is worse than a visible failure. Everything in this
module exists to keep the two aligned, or to make a misalignment loud:

* Both files are written to a temp file, fsynced, and then `os.replace`d, so an interrupted
  write (disk full, forced shutdown, antivirus grabbing the handle) leaves the previous
  known-good index intact instead of a truncated one.
* The two replaces happen back to back under the cross-process `swap` lock, which searches
  also take, so a reader cannot pair a new index with an old map.
* A crash between the two replaces is still possible, so `vector_index_size` lets the next
  index pass compare `ntotal` against the map length and fall back to a full rebuild.
* `search_vector_index` bounds-checks every vector id against the map, so an index that is
  longer than its map drops the orphaned tail rather than raising `IndexError` — or worse,
  reading past the end of a list that happens to be long enough.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from convsearch.config.settings import faiss_index_path, vector_map_path
from convsearch.indexes.locking import SWAP_LOCK, index_lock

SWAP_LOCK_TIMEOUT = 60.0


class VectorIndexError(RuntimeError):
    """The on-disk vector index could not be read or written, with an actionable message."""


@dataclass(frozen=True)
class VectorSearchResult:
    passage_id: int
    score: float
    rank: int


def normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def _faiss() -> Any | None:
    """The faiss module, or None when only the numpy fallback backend is available."""
    try:
        return cast(Any, importlib.import_module("faiss"))
    except ImportError:
        return None


def _temp_path(target: Path) -> Path:
    # Same directory as the target so os.replace stays on one filesystem, and pid-tagged so
    # two processes cannot clobber each other's half-written file. The suffix is preserved
    # because np.save() silently appends ".npy" to any path that does not already end in it.
    return target.with_name(f"{target.stem}.tmp-{os.getpid()}{target.suffix}")


def _fsync(path: Path) -> None:
    """Flush a finished temp file to disk. Best effort — a failure here is not a bad write."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_temp(target: Path, write: Callable[[Path], None]) -> Path:
    """Write `target`'s new contents into a sibling temp file and flush it to disk."""
    tmp = _temp_path(target)
    try:
        write(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise VectorIndexError(
            f"could not write {target} (via {tmp.name}): {type(exc).__name__}: {exc}. The "
            "previous index is untouched. Free up disk space or release any lock on the "
            "workspace, then rebuild with `convsearch index` or POST /reindex."
        ) from exc
    _fsync(tmp)
    return tmp


def _commit(
    workspace: Path,
    index_target: Path,
    write_index: Callable[[Path], None],
    passage_ids: list[int],
) -> None:
    """Publish a new index and its map together, as close to atomically as two files allow."""
    map_target = vector_map_path(workspace)
    payload = json.dumps({"passage_ids": passage_ids})

    def write_map(path: Path) -> None:
        path.write_text(payload, encoding="utf-8")

    index_tmp = _write_temp(index_target, write_index)
    try:
        map_tmp = _write_temp(map_target, write_map)
    except Exception:
        index_tmp.unlink(missing_ok=True)
        raise
    try:
        # Both temp files are complete by now, so the lock is held only for two renames.
        with index_lock(workspace, SWAP_LOCK, timeout=SWAP_LOCK_TIMEOUT):
            os.replace(index_tmp, index_target)
            os.replace(map_tmp, map_target)
    except OSError as exc:
        raise VectorIndexError(
            f"could not publish the new index into {index_target.parent}: {exc}. The previous "
            "index is untouched; rebuild with `convsearch index` or POST /reindex."
        ) from exc
    finally:
        index_tmp.unlink(missing_ok=True)
        map_tmp.unlink(missing_ok=True)


def write_vector_index(
    workspace: Path, vectors: NDArray[np.float32], passage_ids: list[int]
) -> tuple[str, int]:
    index_path = faiss_index_path(workspace)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape((0, 0))
    vectors = normalize(vectors) if vectors.size else vectors
    dimension = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    faiss = _faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(dimension)
        if len(vectors):
            index.add(vectors)
        _commit(
            workspace,
            index_path,
            lambda path: faiss.write_index(index, str(path)),
            passage_ids,
        )
        return "faiss", dimension
    written = vectors
    _commit(
        workspace,
        index_path.with_suffix(".npy"),
        lambda path: np.save(str(path), written, allow_pickle=False),
        passage_ids,
    )
    return "numpy-fallback", dimension


def read_vector_map(workspace: Path) -> list[int]:
    """Passage ids in vector-id order. Position in this list IS the FAISS vector id."""
    map_path = vector_map_path(workspace)
    if not map_path.exists():
        return []
    try:
        data = json.loads(map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [int(value) for value in data.get("passage_ids", [])]


def vector_index_size(workspace: Path) -> int | None:
    """Vectors actually on disk, or None when the index is absent or unreadable.

    Callers use this to catch the two states no single-file write can rule out: a crash
    between the index replace and the map replace, and a corrupt index file. Both are
    recoverable — but only by rebuilding, so they must be detected rather than appended to.
    """
    index_path = faiss_index_path(workspace)
    faiss = _faiss()
    if faiss is None:
        npy_path = index_path.with_suffix(".npy")
        if not npy_path.exists():
            return None
        try:
            array = np.load(str(npy_path), mmap_mode="r", allow_pickle=False)
        except Exception:
            return None
        return int(array.shape[0]) if array.ndim == 2 else None
    if not index_path.exists():
        return None
    try:
        return int(faiss.read_index(str(index_path)).ntotal)
    except Exception:
        return None


def append_vector_index(
    workspace: Path, vectors: NDArray[np.float32], passage_ids: list[int]
) -> tuple[str, int]:
    """Append vectors to the existing index without re-encoding what is already there.

    Only safe when nothing already indexed has been deleted or edited — callers must decide
    that before getting here, because a vector left pointing at a deleted passage row shows
    up as a search hit for text that no longer exists.
    """
    if not passage_ids:
        raise ValueError("append_vector_index requires at least one passage")
    index_path = faiss_index_path(workspace)
    existing_ids = read_vector_map(workspace)
    vectors = normalize(np.asarray(vectors, dtype=np.float32))
    dimension = int(vectors.shape[1])
    combined_ids = [*existing_ids, *passage_ids]
    faiss = _faiss()
    if faiss is not None:
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorIndexError(
                f"the vector index at {index_path} could not be read ({type(exc).__name__}: "
                f"{exc}), so new passages cannot be appended. Rebuild it with "
                "`convsearch index -w <workspace>` or POST /reindex."
            ) from exc
        if int(index.d) != dimension:
            raise ValueError(f"index dimension {index.d} does not match vectors {dimension}")
        if int(index.ntotal) != len(existing_ids):
            # The map and the index have diverged; appending would misalign every id.
            raise ValueError(
                f"vector map has {len(existing_ids)} ids but index holds {index.ntotal}"
            )
        index.add(vectors)
        _commit(
            workspace, index_path, lambda path: faiss.write_index(index, str(path)), combined_ids
        )
        return "faiss", dimension
    npy_path = index_path.with_suffix(".npy")
    try:
        existing = (
            np.load(str(npy_path), allow_pickle=False)
            if npy_path.exists()
            else np.zeros((0, dimension), dtype=np.float32)
        )
    except Exception as exc:
        raise VectorIndexError(
            f"the vector index at {npy_path} could not be read ({type(exc).__name__}: {exc}), "
            "so new passages cannot be appended. Rebuild it with `convsearch index "
            "-w <workspace>` or POST /reindex."
        ) from exc
    if int(existing.shape[0]) != len(existing_ids):
        raise ValueError(
            f"vector map has {len(existing_ids)} ids but index holds {existing.shape[0]}"
        )
    combined = np.vstack([np.asarray(existing, dtype=np.float32), vectors])
    _commit(
        workspace,
        npy_path,
        lambda path: np.save(str(path), combined, allow_pickle=False),
        combined_ids,
    )
    return "numpy-fallback", dimension


def _query_index(
    index_path: Path, query: NDArray[np.float32], limit: int
) -> list[tuple[int, float]]:
    faiss = _faiss()
    if faiss is not None:
        try:
            index = faiss.read_index(str(index_path))
            scores, indices = index.search(query, limit)
        except Exception as exc:
            raise VectorIndexError(
                f"the vector index at {index_path} could not be read ({type(exc).__name__}: "
                f"{exc}). Rebuild it with `convsearch index -w <workspace>` or POST /reindex."
            ) from exc
        return [
            (int(vector_id), float(score))
            for vector_id, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False)
        ]
    npy_path = index_path.with_suffix(".npy")
    try:
        vectors = np.load(str(npy_path), allow_pickle=False)
    except Exception as exc:
        raise VectorIndexError(
            f"the vector index at {npy_path} could not be read ({type(exc).__name__}: {exc}). "
            "Rebuild it with `convsearch index -w <workspace>` or POST /reindex."
        ) from exc
    scores = np.dot(vectors, query[0])
    ranked = np.argsort(-scores)[:limit]
    return [(int(vector_id), float(scores[vector_id])) for vector_id in ranked]


def search_vector_index(
    workspace: Path, query_vector: NDArray[np.float32], limit: int
) -> list[VectorSearchResult]:
    query = normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    index_path = faiss_index_path(workspace)
    # Read the map and the index inside the swap lock so another process publishing a rebuilt
    # index cannot hand us a new index paired with the old (differently ordered) map.
    with index_lock(workspace, SWAP_LOCK, timeout=SWAP_LOCK_TIMEOUT, required=False):
        passage_ids = read_vector_map(workspace)
        if not passage_ids:
            return []
        pairs = _query_index(index_path, query, min(limit, len(passage_ids)))
    results: list[VectorSearchResult] = []
    for rank, (vector_index, score) in enumerate(pairs, start=1):
        if vector_index < 0 or vector_index >= len(passage_ids):
            # FAISS pads short result sets with -1. Ids past the end of the map mean an
            # interrupted write left the index longer than its map: those vectors have no
            # passage id, so the only safe answer is to drop them. The next index pass sees
            # the same mismatch through vector_index_size() and rebuilds.
            continue
        results.append(VectorSearchResult(passage_ids[vector_index], float(score), rank))
    return results
