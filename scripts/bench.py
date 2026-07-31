"""Latency benchmark for the convsearch capture → index → search pipeline.

Measures the things a user actually waits on, at several corpus sizes so scaling problems
show up rather than hiding behind a toy dataset:

  capture      POST /capture round trip. Happens while the user browses, so it must stay
               flat and single-digit milliseconds regardless of corpus size.
  index        one incremental indexing pass. Determines how long after opening a
               conversation it becomes searchable.
  search       GET /search round trip, warm. The number the UI lives or dies by.
  first-search cost of the very first query after startup (model load included).

Run against a DISPOSABLE workspace on a non-default port:

    ./.venv/Scripts/python.exe scripts/bench.py
    ./.venv/Scripts/python.exe scripts/bench.py --sizes 10,100,500 --real-model

By default it uses deterministic embeddings so the numbers isolate convsearch's own
overhead. --real-model includes sentence-transformers, which is what production feels like.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parent.parent
CONVSEARCH = REPO / ".venv" / "Scripts" / "convsearch.exe"

# Deliberately not 8756: never benchmark against a server holding real data.
DEFAULT_PORT = 8801
WORKSPACE = REPO / "tmp" / "bench-ws"

# Vocabulary pools kept disjoint per conversation. `passages.content_hash` is globally
# UNIQUE with INSERT OR IGNORE, so repeated text would silently collapse into one passage
# row and make the corpus smaller than the benchmark claims.
SUBJECTS = [
    "kiln cooling",
    "turf insulation",
    "derailleur wrap",
    "kelp corridor",
    "sourdough crumb",
    "borrow checker",
    "vacuum tube",
    "tide gauge",
    "loom warp",
    "aquifer recharge",
    "basalt column",
    "monsoon onset",
    "lens coating",
    "gear hobbing",
    "yeast pitching",
    "delta blues",
    "quartz inversion",
    "cold forging",
    "seed vault",
    "kite aerodynamics",
]
VERBS = ["governs", "constrains", "amplifies", "dampens", "reshapes", "delays", "sharpens"]
OBJECTS = [
    "the annealing schedule",
    "the load path",
    "the drive ratio",
    "the migration route",
    "the fermentation curve",
    "the ownership graph",
    "the bias point",
    "the datum plane",
]


@dataclass
class Samples:
    label: str
    values: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.values.append(seconds * 1000.0)

    def summary(self) -> str:
        if not self.values:
            return f"{self.label:<14} no samples"
        ordered = sorted(self.values)
        p50 = statistics.median(ordered)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        return (
            f"{self.label:<14} n={len(ordered):<4} "
            f"p50={p50:8.1f}ms  p95={p95:8.1f}ms  max={ordered[-1]:8.1f}ms"
        )


def conversation(index: int, turns: int = 4) -> dict[str, object]:
    """A synthetic conversation whose every passage is textually unique."""
    subject = SUBJECTS[index % len(SUBJECTS)]
    messages = []
    for turn in range(turns):
        verb = VERBS[(index + turn) % len(VERBS)]
        obj = OBJECTS[(index * 3 + turn) % len(OBJECTS)]
        role = "user" if turn % 2 == 0 else "assistant"
        messages.append(
            {
                "source_message_id": f"bench-{index}-{turn}",
                "role": role,
                "text": (
                    f"Conversation {index} turn {turn}: how {subject} {verb} {obj} "
                    f"under sustained load, and what instrumentation number {index * 7 + turn} "
                    f"would reveal about the interaction over a long observation window."
                ),
                "order": turn,
                "created_at": None,
            }
        )
    return {
        "source_conversation_id": f"bench{index:06d}-0000-4000-a000-{index:012d}",
        "title": f"Bench {index}: {subject}",
        "created_at": None,
        "updated_at": "2026-07-29T18:00:00Z",
        "messages": messages,
    }


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def post(self, path: str, payload: object | None = None, timeout: float = 600) -> object:
        body = json.dumps(payload).encode() if payload is not None else b""
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def get(self, path: str, timeout: float = 600) -> object:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=timeout) as response:
            return json.load(response)

    def health(self) -> dict[str, object]:
        return self.get("/health")  # type: ignore[return-value]

    def search(self, query: str, **params: str) -> dict[str, object]:
        qs = urlencode({"q": query, "limit": "10", **params})
        return self.get(f"/search?{qs}")  # type: ignore[return-value]


def wait_for_idle(client: Client, timeout: float = 900) -> float:
    """Block until no indexing pass is running or queued. Returns seconds waited."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if client.health().get("indexing") is False:
            return time.perf_counter() - started
        time.sleep(0.05)
    raise TimeoutError("indexing never settled")


def run(sizes: list[int], real_model: bool, port: int, keep: bool) -> int:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    subprocess.run([str(CONVSEARCH), "init", str(WORKSPACE)], check=True, capture_output=True)

    args = [
        str(CONVSEARCH),
        "serve",
        "--workspace",
        str(WORKSPACE),
        "--port",
        str(port),
        # A tight debounce so the benchmark is not dominated by deliberate waiting.
        "--auto-index-delay",
        "0.2",
    ]
    if not real_model:
        args.append("--test-embeddings")

    log_path = REPO / "tmp" / "bench-server.log"
    with log_path.open("w", encoding="utf-8") as log:
        return _measure(args, log, log_path, sizes, real_model, port, keep)


def _measure(
    args: list[str],
    log: object,
    log_path: Path,
    sizes: list[int],
    real_model: bool,
    port: int,
    keep: bool,
) -> int:
    server = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)  # type: ignore[arg-type]
    client = Client(f"http://127.0.0.1:{port}")

    try:
        for _ in range(240):
            try:
                client.health()
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("server never came up; see", log_path)
            return 1

        model = "real sentence-transformers" if real_model else "deterministic vectors"
        print(f"convsearch latency benchmark — {model}, port {port}")
        print(f"corpus sizes: {sizes}\n")

        # The first query pays for model load and index open. Reported separately because
        # averaging it into the warm numbers would hide both.
        seed = conversation(0)
        client.post("/capture", {"conversations": [seed]})
        wait_for_idle(client)
        started = time.perf_counter()
        client.search("instrumentation observation window")
        print(f"first search after startup: {(time.perf_counter() - started) * 1000:.1f}ms\n")

        written = 1
        for target in sizes:
            capture = Samples("capture")
            index_pass = Samples("index 1 conv")
            search_warm = Samples("search")

            # Build the corpus up to `target` first, timing only the capture round trip.
            while written < target:
                payload = {"conversations": [conversation(written)]}
                started = time.perf_counter()
                client.post("/capture", payload)
                capture.add(time.perf_counter() - started)
                written += 1
            wait_for_idle(client)

            # Then measure the number a user actually feels: open ONE conversation, how long
            # until it is searchable? Measuring passes during the bulk load instead would
            # conflate how many captures happened to coalesce into a pass with how large the
            # corpus is, and the two scale completely differently.
            for _ in range(5):
                client.post("/capture", {"conversations": [conversation(written)]})
                written += 1
                index_pass.add(wait_for_idle(client))

            queries = [
                "instrumentation observation window",
                '"quartz inversion"',
                "load path -derailleur",
                f"how {SUBJECTS[3]} governs the migration route",
            ]
            for _ in range(5):
                for query in queries:
                    started = time.perf_counter()
                    client.search(query)
                    search_warm.add(time.perf_counter() - started)

            health = client.health()
            print(
                f"--- corpus: {health['conversations']} conversations, "
                f"{health['messages']} messages ---"
            )
            print("  " + capture.summary())
            print("  " + index_pass.summary())
            print("  " + search_warm.summary())
            print()

        print("Notes")
        print("  capture must stay flat as the corpus grows — it is a plain SQLite write.")
        print("  index 1 conv = debounce + encode + FAISS append for a SINGLE new")
        print("      conversation on top of the stated corpus. Should stay flat if")
        print("      indexing is genuinely incremental.")
        print("  search is warm: model and index already loaded.")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        if not keep and WORKSPACE.exists():
            shutil.rmtree(WORKSPACE, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10,50,200", help="comma-separated corpus sizes")
    parser.add_argument("--real-model", action="store_true", help="use sentence-transformers")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--keep", action="store_true", help="keep the benchmark workspace")
    options = parser.parse_args()
    sizes = sorted(int(value) for value in options.sizes.split(",") if value.strip())
    return run(sizes, options.real_model, options.port, options.keep)


if __name__ == "__main__":
    sys.exit(main())
