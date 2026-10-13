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

## Next run (Day 4) — PaperTrail M2: BM25 baseline + eval harness
- First re-test network: `curl https://export.arxiv.org/...` and HF datasets. If reachable, fetch ~2k cs.CL/cs.LG abstracts via `papertrail.cli` and record the count.
- If still blocked: build a small *realistic* corpus another way (e.g. PyPI-installable datasets, if any contain arXiv-style abstracts) or keep the fixture and mark all numbers TODO. Do not invent results.
- Implement own BM25 (tokenizer, inverted index, scoring) in `src/papertrail/retrieval/bm25.py` with unit tests; implement metrics Recall@K, MRR, nDCG@K (`eval/metrics.py`) with hand-computed test cases; define the golden-set format (query → relevant arxiv_ids) in an ADR.
- LEARNING_LOG: BM25 formula intuition, why metrics differ (Recall vs MRR vs nDCG).

## Needs Priyanka
- **Decision (blocking real results):** the build sandbox can't reach arxiv.org/huggingface.co. Easiest fix: in your Claude environment settings, allow network access to `export.arxiv.org` and `huggingface.co` (or full access). Otherwise run `make install && python -m papertrail.cli --category cs.CL --n 2000` on your own laptop and commit `data/` metadata is gitignored — tell me and we'll pick another route.
- Start the parallel DSA track yourself (a few problems per week).
- Later: an LLM API key helps the agent project (not needed for M2–M4).
