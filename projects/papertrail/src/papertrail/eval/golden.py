"""Golden-set IO. JSONL, one object per line: {"qid", "query", "relevant": [arxiv_id, ...]}."""

from __future__ import annotations

import json
from pathlib import Path


def load_golden(path: str | Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return (queries, qrels) keyed by qid. Rejects duplicate qids and empty relevance lists."""
    queries: dict[str, str] = {}
    qrels: dict[str, set[str]] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row["qid"]
        if qid in queries:
            raise ValueError(f"line {n}: duplicate qid {qid!r}")
        if not row["relevant"]:
            raise ValueError(f"line {n}: qid {qid!r} has no relevant docs")
        queries[qid] = row["query"]
        qrels[qid] = set(row["relevant"])
    return queries, qrels


def run_retriever(search, queries: dict[str, str], k: int = 10) -> dict[str, list[str]]:
    """Apply `search(query, k) -> [(doc_id, score)]` to every query; keep only the id ranking."""
    return {qid: [d for d, _ in search(q, k)] for qid, q in queries.items()}
