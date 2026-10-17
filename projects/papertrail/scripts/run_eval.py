"""Compare retrieval configs on the golden sets; writes results JSON + markdown tables.

Usage: python scripts/run_eval.py data/papers.parquet eval_data results
Corpus = abstracts only (titles excluded so the title golden set isn't trivially lexical).
Latency = wall-clock per query, single process, CPU, in this sandbox (see `env` in output).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

from papertrail.eval.golden import load_golden
from papertrail.eval.runner import compare, to_markdown
from papertrail.io import read_papers
from papertrail.retrieval.bm25 import BM25Index
from papertrail.retrieval.dense import DenseIndex, SentenceTransformerEmbedder
from papertrail.retrieval.hybrid import HybridRetriever
from papertrail.retrieval.rerank import CrossEncoderReranker, RerankedRetriever
from papertrail.retrieval.rewrite import RewritingRetriever

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> int:
    corpus, golden_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    papers = read_papers(corpus)
    ids, texts = [p.arxiv_id for p in papers], [p.abstract for p in papers]
    build = {}

    t0 = time.perf_counter()
    bm25 = BM25Index().build(ids, texts)
    build["bm25_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense = DenseIndex(SentenceTransformerEmbedder(EMBED_MODEL)).build(ids, texts)
    build["dense_s"] = time.perf_counter() - t0  # includes model load + encoding all abstracts

    hybrid = HybridRetriever(bm25, dense)
    reranker = CrossEncoderReranker(RERANK_MODEL)
    by_id = dict(zip(ids, texts, strict=True))
    configs = {
        "bm25": bm25,
        "bm25+rewrite": RewritingRetriever(bm25),
        "dense": dense,
        "hybrid_rrf": hybrid,
        "hybrid_rrf+rewrite": RewritingRetriever(hybrid),
        "hybrid_rrf+rerank@50": RerankedRetriever(hybrid, reranker, by_id, top_n=50),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus": {"path": str(corpus), "n_docs": len(papers), "text": "abstract"},
        "models": {"embedder": EMBED_MODEL, "reranker": RERANK_MODEL},
        "build_seconds": build,
        "env": {
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "machine": platform.machine(),
        },
        "results": {},
    }
    md = []
    for name in ("titles", "doc2query"):
        queries, qrels = load_golden(golden_dir / f"golden_{name}.jsonl")
        res = compare(configs, queries, qrels, k=10)
        report["results"][name] = res
        md += [f"### Golden set: {name} ({len(queries)} queries)", "", to_markdown(res), ""]
        print(md[-4], md[-2], sep="\n", flush=True)
    (out_dir / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "retrieval.md").write_text("\n".join(md), encoding="utf-8")
    print("build:", build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
