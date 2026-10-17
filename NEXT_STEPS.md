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

- 2026-10-16 — PaperTrail M4: reranker wrapper + cross-encoder, query rewriting, eval runner with latency p50/p95, ADR 0004, 40 tests pass. Network still blocked (re-tested).

- 2026-10-17 — PaperTrail M5: `generation/` (prompt, citation validation, lexical faithfulness, extractive generator, `answer_question`), ADR 0005, 49 tests pass. Network still blocked (curl to export.arxiv.org, huggingface.co, semanticscholar all fail).
- 2026-10-17 (evening, manual run after network opened): ingested 2,000 real cs.CL abstracts, built 2 golden sets, first real retrieval evaluation (6 configs). ADR 0006.

## Next run — PaperTrail M6: FastAPI + Docker (network is now open)
- First real results exist (2026-10-17 evening, manual run): see `projects/papertrail/README.md` Results and ADR 0006. Don't re-run the full eval unless code changes; it takes ~16 min on CPU.
- Build M6: `api/` FastAPI app (`/health`, `/search?config=bm25|hybrid|hybrid_rerank`, `/answer`), indexes loaded once at startup, pydantic response models with citations, per-stage latency and a cost/query field (0 for local models). TestClient tests using offline fakes (HashingEmbedder, OverlapReranker) so CI needs no downloads. Dockerfile + docker-compose, CI step, ADR 0007.
- Small add-on if time allows: make `eval/runner.py` keep per-query results and add a paired bootstrap CI for MRR differences (rerank vs bm25, hybrid vs bm25). Report CIs in the README.
- Later queue: rerank top_n sweep (10/20/50) for a quality-vs-latency curve; scale to 10k+ papers (multiple categories, ~15 min arXiv fetch); wire a local or API LLM for M5 generation + faithfulness; 30–50 hand-labelled queries.
- LEARNING_LOG: API design for ML services (startup loading, schemas, health checks), Docker layering.

## Needs Priyanka
- **Resume check (important):** draft bullets with real numbers are in `PORTFOLIO_ROADMAP.md` → PaperTrail → Resume bullets. Your resume says "10,000+ papers", but the measured corpus is 2,000. Either change it to 2,000 now or wait for the scale-up run.
- Start the parallel DSA track yourself (a few problems per week).
- Later: an LLM API key helps generation/faithfulness (M5) and the agent project. Optional; a local model will be tried first.
