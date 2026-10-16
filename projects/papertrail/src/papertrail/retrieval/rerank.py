"""Second-stage reranking.

A bi-encoder (dense.py) embeds query and document separately, so it is fast but coarse.
A cross-encoder reads (query, document) together and scores the pair directly: more accurate,
but one model forward pass per candidate, so it is only run on the top-N of a cheap first stage.
"""

from __future__ import annotations

from typing import Protocol

from papertrail.retrieval.hybrid import Searcher


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher = more relevant)."""
        ...


class OverlapReranker:
    """Offline stand-in: fraction of query tokens present in the text. Not semantic; for tests."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(t.lower().split())) / len(q) for t in texts]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper (needs the `embed` extra + model download)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class RerankedRetriever:
    """First-stage searcher -> rerank top `top_n` -> return top k. Same `search` interface."""

    def __init__(
        self, first_stage: Searcher, reranker: Reranker, texts: dict[str, str], top_n: int = 50
    ):
        self.first_stage, self.reranker, self.texts, self.top_n = (
            first_stage,
            reranker,
            texts,
            top_n,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        cands = [d for d, _ in self.first_stage.search(query, max(self.top_n, k))]
        scores = self.reranker.score(query, [self.texts[d] for d in cands])
        # stable sort: ties keep first-stage order
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        return [(cands[i], scores[i]) for i in order[:k]]
