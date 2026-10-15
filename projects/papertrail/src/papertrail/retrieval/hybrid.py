"""Reciprocal Rank Fusion (Cormack et al., 2009) and a hybrid retriever.

RRF score(d) = sum over rankers r of weight_r / (c + rank_r(d)), rank starting at 1.
It uses ranks only, so BM25 scores (unbounded) and cosine scores (-1..1) need no calibration.
"""

from __future__ import annotations

from typing import Protocol


class Searcher(Protocol):
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]: ...


def rrf_fuse(
    rankings: list[list[str]], c: int = 60, weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ties broken by first appearance for determinism."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (c + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class HybridRetriever:
    def __init__(
        self,
        sparse: Searcher,
        dense: Searcher,
        c: int = 60,
        candidates: int = 100,
        weights: tuple[float, float] = (1.0, 1.0),
    ):
        self.sparse, self.dense = sparse, dense
        self.c, self.candidates, self.weights = c, candidates, list(weights)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        n = max(self.candidates, k)
        lists = [[d for d, _ in r.search(query, n)] for r in (self.sparse, self.dense)]
        return rrf_fuse(lists, self.c, self.weights)[:k]
