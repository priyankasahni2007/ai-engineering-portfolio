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

## Next run
- Day 1: Phase A(a) — research the 2026 AI/ML engineering hiring market → `research/market-2026.md`. Keep it to one run.
- Day 2: Phase A(b)+(c) — rank 15+ ideas, pick the 5–7 flagships (PaperTrail is #1, fixed), roadmap, create PORTFOLIO_ROADMAP.md.
- Day 3 onward: build PaperTrail, starting with the repo skeleton + data ingestion of arXiv papers.
- Priority: get a working, presentable first version (retrieval + eval + README with real numbers) as early as possible, since it is on her resume.

## Needs Priyanka
- Nothing yet.
