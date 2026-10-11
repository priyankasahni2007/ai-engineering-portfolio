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
- 2026-10-11 — Phase A(a): market research → `research/market-2026.md` (implications in §5).

## Next run (Day 2)
- Phase A(b)+(c) in one run: rank 15+ project ideas → `research/project-ranking.md`. Then pick 5–7 flagships (PaperTrail is #1, fixed), write the 6–12 month roadmap, and create `PORTFOLIO_ROADMAP.md`.
- Use `research/market-2026.md` §5 as input:
  - The agent project reuses PaperTrail as a tool.
  - Include one ranking/recsys classic-ML project with A/B or interleaving analysis.
  - Include one LLM-internals project measuring quantization × KV cache × batching on CPU.
  - Add a parallel DSA prep track (not a project).
- Keep the roadmap front-loaded: PaperTrail v1 (retrieval + eval + README with real numbers) should land before the end of November 2026.
- Day 3 onward: build PaperTrail, starting with the skeleton (pyproject, src layout, tests, Makefile, CI) and arXiv ingestion.

## Needs Priyanka
- Optional, ~10 min: skim `research/market-2026.md` §1 and §5. If you disagree with the lane priorities (Applied AI first, classic MLE second), say so in this file before Day 2 runs.
- Heads-up for later (not needed yet): the faithfulness eval and the answer generation will eventually need an LLM. We'll start with a free local model on CPU. An API key (e.g. Anthropic) is optional and will be flagged when it becomes useful.
