"""Okapi BM25 from scratch: tokenizer, inverted index, scoring.

score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))
idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (the non-negative "Lucene" variant)
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Small stoplist: keeps ML jargon ("is" is dropped, "lora" is not).
STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has in is it its of on or that the this to was we with"
    ).split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, keep hyphen/dot-joined terms ("kv-cache", "bert-base") as one token."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_idx, tf)]
        self.avgdl = 0.0

    def build(self, doc_ids: list[str], texts: list[str]) -> BM25Index:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.postings = {}
        self.doc_len = []
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        return self

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        n = len(self.doc_ids)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, score), best first; ties broken by doc order for determinism."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):  # repeated query terms don't double count
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]
