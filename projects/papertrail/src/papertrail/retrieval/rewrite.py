"""Rule-based query rewriting: expand known ML abbreviations so BM25 can match either form.

Deliberately simple and deterministic. An LLM rewriter can implement the same `rewrite` callable
later; the eval runner treats both identically so the gain (or loss) is measured, not assumed.
"""

from __future__ import annotations

import re

ABBREVIATIONS: dict[str, str] = {
    "llm": "large language model",
    "lora": "low-rank adaptation",
    "rag": "retrieval-augmented generation",
    "rlhf": "reinforcement learning from human feedback",
    "cnn": "convolutional neural network",
    "gnn": "graph neural network",
    "kv": "key value",
    "moe": "mixture of experts",
    "nlp": "natural language processing",
    "vit": "vision transformer",
}
_WORD = re.compile(r"[A-Za-z0-9]+")


def expand_query(query: str, table: dict[str, str] | None = None) -> str:
    """Append the long form after each known abbreviation; unknown text is left untouched."""
    table = ABBREVIATIONS if table is None else table
    extra = [table[w.lower()] for w in _WORD.findall(query) if w.lower() in table]
    return query if not extra else f"{query} {' '.join(extra)}"


class RewritingRetriever:
    """Wrap any searcher with a query rewrite step."""

    def __init__(self, inner, rewrite=expand_query):
        self.inner, self.rewrite = inner, rewrite

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.inner.search(self.rewrite(query), k)
