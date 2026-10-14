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

## Next run (Day 5) — PaperTrail M3: dense retrieval + hybrid
- Re-test network first (curl arxiv + HF). If open: fetch ~2k abstracts, build a golden set (synthetic + note bias), run BM25 eval and record real numbers.
- Implement dense retrieval: sentence-transformers (pip; model weights need huggingface.co, so check reachability; if blocked, build the module behind an interface with a fake embedder in tests and mark runs TODO), FAISS flat index, RRF fusion in `retrieval/hybrid.py` with tests.
- LEARNING_LOG: embeddings/cosine, FAISS flat vs HNSW, why RRF.
- If network is still blocked and Priyanka has not acted, do not stall: finish the module + tests and tell her again.

## Needs Priyanka
- **Decision (blocking real results, raised again Day 4):** the build sandbox can't reach arxiv.org/huggingface.co. Easiest fix: in your Claude environment settings, allow network access to `export.arxiv.org` and `huggingface.co` (or full access). Otherwise run `make install && python -m papertrail.cli --category cs.CL --n 2000` on your own laptop and commit `data/` metadata is gitignored — tell me and we'll pick another route.
- Start the parallel DSA track yourself (a few problems per week).
- Later: an LLM API key helps the agent project (not needed for M2–M4).
