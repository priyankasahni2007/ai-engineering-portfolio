"""Run searchers over a golden set: quality metrics plus per-query latency percentiles."""

from __future__ import annotations

import time

import numpy as np

from papertrail.eval.metrics import evaluate


def run_config(
    searcher,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    k: int = 10,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Metrics + latency_p50_ms / latency_p95_ms (wall-clock, single-threaded, per query)."""
    runs: dict[str, list[str]] = {}
    lat: list[float] = []
    for qid, q in queries.items():
        t0 = time.perf_counter()
        hits = searcher.search(q, k)
        lat.append((time.perf_counter() - t0) * 1000)
        runs[qid] = [d for d, _ in hits]
    out = evaluate(runs, qrels, ks)
    out["latency_p50_ms"] = float(np.percentile(lat, 50))
    out["latency_p95_ms"] = float(np.percentile(lat, 95))
    out["n_queries"] = len(queries)
    return out


def compare(configs: dict[str, object], queries, qrels, k: int = 10) -> dict[str, dict[str, float]]:
    return {name: run_config(s, queries, qrels, k) for name, s in configs.items()}


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    cols = ["recall@5", "recall@10", "mrr", "ndcg@10", "latency_p50_ms", "latency_p95_ms"]
    lines = ["| config | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)
