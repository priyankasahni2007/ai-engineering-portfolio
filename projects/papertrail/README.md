# PaperTrail

Evaluated hybrid retrieval-augmented research assistant over arXiv ML papers.

**Status:** retrieval stack (BM25, dense, hybrid RRF, cross-encoder rerank, query rewrite) is evaluated on real arXiv data. Generation (M5) has code but no LLM is wired in yet. API, Docker and the full README come next (M6–M7).

**Headline (2,000 cs.CL abstracts, 200 synthetic questions):** reranking the hybrid top 50 with a cross-encoder raised MRR from 0.716 (BM25) to **0.838** and Recall@10 from 0.825 to **0.925**. The cost is about **2 s per query on a 4-vCPU CPU**, compared with under 1 ms for BM25.

## Results — retrieval (2026-10-17)
Corpus: 2,000 newest cs.CL abstracts (2026-09-29 → 2026-10-15). The index holds abstracts only; IDs are in `eval_data/corpus_ids.txt`. Latency is per-query wall-clock time, single process, on a 4-vCPU CPU sandbox. Each query has one labelled relevant paper, so **all scores are lower bounds** (see ADR 0006).

**Golden set A — doc2query** (200 questions generated from abstracts by a T5 query generator)

| config | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.795 | 0.825 | 0.716 | 0.742 | 0.4 | 1.7 |
| bm25 + rewrite | 0.795 | 0.825 | 0.713 | 0.740 | 0.3 | 1.8 |
| dense (bge-small) | 0.700 | 0.755 | 0.588 | 0.628 | 16.3 | 25.2 |
| hybrid RRF | 0.825 | 0.875 | 0.703 | 0.745 | 15.0 | 20.8 |
| hybrid RRF + rewrite | 0.825 | 0.875 | 0.689 | 0.734 | 15.7 | 22.9 |
| **hybrid RRF + rerank@50** | **0.900** | **0.925** | **0.838** | **0.859** | 1929 | 2365 |

**Golden set B — titles** (200 paper titles used as queries; titles are not indexed)

| config | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| bm25 | 0.945 | 0.960 | 0.925 | 0.934 | 0.9 | 2.4 |
| bm25 + rewrite | 0.945 | 0.955 | 0.924 | 0.932 | 1.0 | 2.4 |
| dense (bge-small) | 0.930 | 0.950 | 0.878 | 0.896 | 15.3 | 22.1 |
| hybrid RRF | 0.965 | 0.975 | 0.928 | 0.940 | 19.0 | 28.5 |
| hybrid RRF + rewrite | 0.955 | 0.970 | 0.928 | 0.938 | 19.0 | 27.9 |
| **hybrid RRF + rerank@50** | **0.975** | **0.990** | **0.959** | **0.967** | 1985 | 2368 |

Index build time: BM25 0.28 s. Dense 168 s, which includes the model load and encoding all 2,000 abstracts on CPU.

### What the numbers say
- **Hybrid improves recall but not ranking.** On doc2query, RRF raises Recall@10 from 0.825 to 0.875, but MRR drops slightly (0.716 → 0.703). Fusion pulls more relevant papers into the top 10 but doesn't put them first.
- **The reranker fixes ranking.** MRR rises +0.12 over BM25 on doc2query. It is also ~130× slower than hybrid alone and ~5,000× slower than BM25, because it runs 50 transformer forward passes per query on CPU.
- **Dense alone is the weakest config here.** This is likely a property of these test sets: many queries contain a paper's unique system name (e.g. "forkgrepo"), and the doc2query questions are generated from the abstract text. Both favour exact keyword matching. It should not be read as "dense retrieval is bad" for paraphrased user questions.
- **Rule-based query rewriting had no effect or hurt slightly** (MRR change between 0 and −0.014). Abbreviation expansion adds words that match many abstracts. It stays switched off by default.
- **Caveat:** 200 queries per set, with no confidence intervals yet. Differences under ~0.02 should be treated as noise until a paired bootstrap test is added.

## Reproduce
```bash
cd projects/papertrail
make install            # plus CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
make lint test          # 50 tests, offline
make ingest             # 2,000 newest cs.CL abstracts (newest-first, so re-fetching gives a newer corpus)
make golden             # ~2 min CPU: title + doc2query golden sets (committed in eval_data/)
make eval               # ~16 min CPU, mostly the cross-encoder; writes results/retrieval.{json,md}
```
`make ingest-sample` runs the offline fixture path. Data schema: `src/papertrail/schema.py`. Decisions: `docs/adr/`.

## Not done yet
LLM answer generation and the faithfulness numbers (M5 code exists, no model wired in). FastAPI + Docker (M6). Confidence intervals. A hand-labelled query set. Scaling to 10k+ papers. Architecture diagram and failure-case analysis.
