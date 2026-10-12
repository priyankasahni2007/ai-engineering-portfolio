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

## Next run (Day 3) — start Phase B: PaperTrail M1
- Create `projects/papertrail/` skeleton: `pyproject.toml`, `src/papertrail/` layout, `tests/`, `Makefile`, `.env.example`, `.gitignore`, GitHub Actions CI at `.github/workflows/ci.yml` (lint with ruff + pytest), `docs/adr/0001-*.md` (project layout + tools).
- Arxiv ingestion: check what is reachable from the sandbox (arXiv API / Kaggle metadata / HF datasets). Pull a small subset (~2k abstracts, cs.CL/cs.LG) into a Parquet file with a documented schema; write tests on the parser. If no network access to data, say so and use a tiny fixture dataset, marking real data as TODO.
- LEARNING_LOG: explain src layout, CI, and why reproducible data ingestion matters.
- Keep each run to one component (M1 then M2 BM25 baseline + eval harness, etc. — see roadmap §1).

## Needs Priyanka
- Optional, ~10 min: skim `PORTFOLIO_ROADMAP.md` (status board + gaps) and `research/project-ranking.md`. If you want a different order of projects 2–7 or want to swap CV for another flagship, edit this file before Day 3.
- Start the parallel DSA track yourself (a few problems per week); this can't be built for you.
- Later (not yet): an LLM API key will help the agent project (project 2). Not needed for PaperTrail M1–M4.
