"""Retrieval metrics with binary relevance. `ranked` is a list of doc ids, best first."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top k. 0.0 if there are no relevant docs."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant doc, 0.0 if none retrieved."""
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """DCG = sum rel_i / log2(i+1); normalised by the best possible DCG."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(
    runs: dict[str, list[str]], qrels: dict[str, set[str]], ks: tuple[int, ...] = (5, 10)
) -> dict[str, float]:
    """Mean metrics over queries in qrels (a query missing from runs counts as an empty ranking)."""
    out: dict[str, float] = {}
    n = len(qrels)
    if n == 0:
        raise ValueError("qrels is empty")
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(runs.get(q, []), r, k) for q, r in qrels.items()) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(runs.get(q, []), r, k) for q, r in qrels.items()) / n
    out["mrr"] = sum(reciprocal_rank(runs.get(q, []), r) for q, r in qrels.items()) / n
    return out
