# Next steps

_Updated by the daily build run. Read this first._

## Fixed decision (from Priyanka, 2026-10-07)
**Project #1 is locked: PaperTrail — Evaluated Retrieval-Augmented Research Assistant.**
It is already listed on her resume as "in progress", so it must be built FIRST and must match this description:
- QA over 10,000+ arXiv ML papers (start with abstracts/a subset; scale up).
- Hybrid retrieval (BM25 + dense embeddings), cross-encoder reranking, query rewriting, cited source passages.
- Evaluation harness: Recall@K, MRR, nDCG for retrieval strategies; answer faithfulness to sources.
- FastAPI service, Docker, tests, GitHub Actions CI; latency and cost per query tracked per configuration.
- Stack: Python, PyTorch, FastAPI, FAISS, BM25, Hugging Face Transformers, Docker, GitHub Actions.
- Code lives in `projects/papertrail/`.
- When results exist, write real-number resume bullets in PORTFOLIO_ROADMAP.md and flag them to Priyanka (she will update her resume).

## Done
- 2026-10-11 — Phase A(a): market research → `research/market-2026.md`.
- 2026-10-12 — Phase A(b)+(c): `research/project-ranking.md` (17 ideas ranked) and `PORTFOLIO_ROADMAP.md` (7 flagships, Oct 2026 – Sep 2027, gaps). Phase A is complete.
- 2026-10-13 — PaperTrail M1: `projects/papertrail/` skeleton, ruff+pytest (11 tests pass), CI workflow, arXiv Atom parser + Parquet IO, ADR 0001. Real data NOT fetched: sandbox blocks arxiv.org and huggingface.co (pypi works).
- 2026-10-14 — PaperTrail M2: own BM25 + metrics (Recall@K, MRR, nDCG@K) + golden-set loader, 26 tests pass, ADR 0002. Network to arxiv.org/huggingface.co still blocked; no corpus, no results.
- 2026-10-15 — PaperTrail M3: `retrieval/dense.py` (Embedder interface, FAISS flat IP index, sentence-transformers wrapper, offline hashing embedder), `retrieval/hybrid.py` (RRF + HybridRetriever), ADR 0003, 34 tests pass. Still no network to arxiv.org/huggingface.co → no corpus, no real results.

## Next run (Day 6) — PaperTrail M4: reranker + query rewriting (code) and an eval runner
- Re-test network first (curl export.arxiv.org + huggingface.co). If open: fetch ~2k abstracts, build synthetic golden set, run BM25 / dense (bge-small) / hybrid through `eval.evaluate`, record REAL numbers in the README results table.
- Write `eval/runner.py` that runs any searcher over a golden set and reports Recall@K/MRR/nDCG@K + latency p50/p95 per config (testable now with HashingEmbedder on the tiny fixture; label such runs as non-semantic).
- Implement `retrieval/rerank.py` behind a `Reranker` interface (cross-encoder wrapper + fake for tests) and `retrieval/rewrite.py` (rule-based expansion first; LLM rewrite optional).
- LEARNING_LOG: bi-encoder vs cross-encoder, why rerank only top-N, query rewriting trade-offs.
- Do not stall if the network is still blocked.

## Needs Priyanka
- **Decision (blocking real results, raised again Day 4):** the build sandbox can't reach arxiv.org/huggingface.co. Easiest fix: in your Claude environment settings, allow network access to `export.arxiv.org` and `huggingface.co` (or full access). Otherwise run `make install && python -m papertrail.cli --category cs.CL --n 2000` on your own laptop and commit `data/` metadata is gitignored — tell me and we'll pick another route.
- Start the parallel DSA track yourself (a few problems per week).
- Later: an LLM API key helps the agent project (not needed for M2–M4).
