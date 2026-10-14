# Portfolio roadmap (master tracker)

_Last updated: 2026-10-14 (Day 4). Target: Applied AI / AI Engineer / ML Engineer internships, Summer 2027. Evidence: `research/market-2026.md`; ranking: `research/project-ranking.md`._

**Rule:** no number appears in a Results / Resume-bullet cell until it has been produced by code in this repo. Until then: `TODO`.

## Status board

| # | Project | Slug | Status | Target window |
|---|---|---|---|---|
| 1 | PaperTrail — evaluated hybrid RAG over arXiv (**locked**) | `papertrail` | In progress — M1, M2 code done; real data fetch pending network | 13 Oct – 30 Nov 2026 (v1 by end Nov) |
| 2 | Research agent on top of PaperTrail | `papertrail-agent` | Planned | Dec 2026 – mid Jan 2027 (winter break) |
| 3 | LLM inference lab (tiny transformer, KV cache, quantization, batching) | `llm-inference-lab` | Planned | mid Jan – Feb 2027 |
| 4 | Two-stage recsys / ranking with A/B-style analysis | `rec-ranking` | Planned | Mar – Apr 2027 |
| 5 | PaperTrail Web (polished AI product) | `papertrail-web` | Planned | Apr – May 2027 |
| 6 | Multimodal figure retrieval | `paper-figures-mm` | Planned | Jun – Jul 2027 |
| 7 | CV fine-tune + deploy | `cv-finetune-deploy` | Planned | Aug – Sep 2027 |

Why this order: internship applications for Summer 2027 open roughly Oct 2026 – early 2027, so the two highest-signal projects (RAG + evals, agent) come first, then the internals project (cheap, high interview value), then the classic-MLE project for big-tech lane. Product/multimodal/CV are bonus-signal and come after applications; the dates are soft and drive the "6–12 month" plan. If hiring windows demand it, projects 1–3 are the minimum viable portfolio.

## Time model
~1 hr/day of Priyanka's review; the daily run does the building. Classes limit mid-semester pace; Dec 2026 – Jan 2027 winter break = extra time (agent project and start of LLM lab). **Parallel DSA track** (not a project): ~3–4 sessions/week on LeetCode-style problems + one system-design topic/week — Google-type internship screens *require* DSA (market §1 #6).

## Biggest portfolio gaps (today)
1. **No public, finished, evaluated project yet** — repo only has research. PaperTrail v1 closes this.
2. **No evidence of production engineering** (API, Docker, CI, tests) — built into every project from the skeleton.
3. **No classic-MLE evidence** (training loop, feature pipeline, offline→online metrics) — project 4.
4. **No systems/internals depth** — project 3.
5. **DSA/system design not tracked** — parallel track above; needs Priyanka's own practice, can't be built for her.
6. **Public profile:** GitHub profile README, pinned repos, LinkedIn posts — start drafting ~2 weeks after PaperTrail has first numbers.
7. **No internship/competition/OSS signal** — consider one small open-source contribution (e.g. to a RAG/eval library) after PaperTrail v1.

---

## 1. PaperTrail — Evaluated Retrieval-Augmented Research Assistant (LOCKED)
- **Status:** In progress. M1 done 2026-10-13: src layout, ruff+pytest (11 tests passing), CI workflow, arXiv Atom parser + Parquet IO, ADR 0001. M2 done 2026-10-14: own BM25 (inverted index), Recall@K/MRR/nDCG@K, golden-set loader, ADR 0002; 26 tests passing. Real arXiv data NOT yet fetched (sandbox still blocks arxiv.org / huggingface.co); no retrieval results yet.
- **Objective:** QA over 10,000+ arXiv ML papers (start with a subset of abstracts) where every answer cites source passages, and every retrieval/generation choice is justified by a measured comparison.
- **Skills demonstrated:** information retrieval, evaluation design, backend service engineering, cost/latency awareness.
- **Tech stack (why):** Python (baseline, 71% of postings) · `rank_bm25`/own BM25 (exact-term matching: "LoRA", "KV-cache") · sentence-transformers embeddings + **FAISS** (dense recall on paraphrases; fast CPU ANN) · RRF fusion (no score calibration) · cross-encoder reranker via Hugging Face Transformers/PyTorch (precision at top-k) · query rewriting (fix vague queries; measured, may not help) · small local LLM for generation (free, CPU; API optional later) · **FastAPI** (service) · **Docker** (reproducible) · **pytest + GitHub Actions** (keep green) · arXiv metadata dump / API (free, open).
- **GitHub:** `projects/papertrail/` (may split into own repo later). **Demo:** TODO (API + short screen recording).
- **Evaluation metrics:** Recall@K, MRR, nDCG@K per retrieval config (BM25, dense, hybrid, hybrid+rerank, ±rewrite); faithfulness of answers to cited passages; p50/p95 latency and cost/query per config.
- **Results:** TODO.
- **Resume bullets:** TODO (real numbers only; flag to Priyanka when ready).
- **LinkedIn content:** launch post; "BM25 vs dense vs hybrid — what the numbers said"; "why my reranker helped/hurt"; faithfulness eval pitfalls; lessons.
- **Interview questions:** Why RRF? Recall@10 high but answers bad — why? How did you build labels? What's LLM-judge bias? Latency budget breakdown? Chunking choice?
- **Next improvements:** scale subset→10k+; HNSW vs flat tradeoff; semantic cache; judge-vs-human calibration on a small labelled set.
- **Planned milestones:** M1 skeleton+CI+ingestion → M2 BM25 baseline + eval harness + golden set → M3 dense+hybrid → M4 reranker + rewrite → M5 generation + citations + faithfulness → M6 FastAPI+Docker → M7 README with results table.

## 2. Research agent on top of PaperTrail
- **Status:** Planned.
- **Objective:** an agent that plans multi-step literature questions ("compare approaches to X across 5 papers"), calling PaperTrail as a tool.
- **Skills:** planning, tool calling, structured outputs (schemas), state/memory, retries, human-in-the-loop approval, tracing, cost caps, task-level evals.
- **Tech stack (why):** LangGraph or a small hand-rolled state machine (decide by comparing; explicit state is the interview topic) · Pydantic structured outputs · FastAPI · tracing via OpenTelemetry/simple JSONL traces · MCP **only if** PaperTrail is exposed as a tool server in a way that makes the boundary meaningful.
- **GitHub:** `projects/papertrail-agent/`. **Demo:** TODO.
- **Metrics:** task success rate on a hand-built task set, tool-call accuracy, steps/task, cost & latency per task, failure taxonomy.
- **Results / bullets:** TODO. **LinkedIn:** agent failure modes; HITL design; cost caps. **Interview Qs:** when not to use an agent; how to bound loops; how to test non-determinism.
- **Next improvements:** memory across sessions; eval of planner variants. **Needs:** probably an LLM API key for decent tool calling (CPU local models are weak at it) — will flag.

## 3. LLM inference lab
- **Status:** Planned.
- **Objective:** build a tiny GPT from scratch, add KV cache, quantize (int8/int4), and benchmark tokens/s, memory and quality (perplexity) across quantization × KV-cache × batch size on CPU.
- **Skills:** transformer internals, systems profiling, honest benchmarking.
- **Stack (why):** PyTorch (framework in most postings) · numpy for from-scratch kernels · `torch.profiler`/timeit · small open weights (e.g. a ~100M model) for the real-model comparison · pytest for numerical-equivalence tests (cache vs no cache must match).
- **GitHub:** `projects/llm-inference-lab/`. **Demo:** TODO (benchmark plots).
- **Metrics:** tokens/s, time-to-first-token, peak memory, perplexity delta vs fp32. **Results/bullets:** TODO.
- **Interview Qs:** why KV cache speeds decoding and what it costs in memory; PTQ vs QAT; why batching raises throughput but hurts latency; memory- vs compute-bound decoding.
- **Next:** speculative decoding; compare to llama.cpp numbers. **Note:** CPU-only numbers; GPU claims will not be made.

## 4. Two-stage recsys / ranking
- **Status:** Planned.
- **Objective:** candidate generation (e.g. matrix factorization / two-tower) + learning-to-rank reranker on MovieLens (free), with leakage-safe time splits and a simulated A/B + interleaving analysis.
- **Skills:** SQL/Pandas feature pipelines, offline metrics, experiment design.
- **Stack (why):** pandas + DuckDB/SQL (feature pipeline), PyTorch (two-tower), LightGBM (LTR baseline), FastAPI, Docker.
- **GitHub:** `projects/rec-ranking/`. **Metrics:** Recall@K, nDCG@K, coverage/novelty, CIs from bootstrap, simulated A/B power analysis. **Results/bullets:** TODO.
- **Interview Qs:** time-split vs random split; position bias; offline-online gap; CTR vs ranking metrics.

## 5. PaperTrail Web (polished AI product)
- **Status:** Planned (after projects 1–4). **Objective:** usable web app: search, cited answers, saved collections, usage analytics.
- **Stack (why):** Next.js/React frontend (what product teams use), FastAPI backend (reuse), Postgres (persistence), auth (only if multi-user), deployment on a free tier. **Metrics:** Lighthouse/latency, retention-style usage events (own testing), eval regression gate in CI. **Results/bullets:** TODO.

## 6. Multimodal figure retrieval
- **Status:** Planned. **Objective:** retrieve and answer questions about paper figures/tables via CLIP/SigLIP-style embeddings; compare against text-only retrieval on a hand-labelled set. **Metrics:** Recall@K on figure queries, text-vs-multimodal win rate. **Stack:** Hugging Face Transformers vision-language encoder, FAISS, reuse PaperTrail API. **Results/bullets:** TODO. **Risk:** labelling effort; CPU speed — use a small subset.

## 7. CV fine-tune + deploy
- **Status:** Planned. **Objective:** fine-tune a small pretrained CNN/ViT on a real open dataset (candidate: plant disease or industrial defect), with error analysis, ONNX export, quantized latency benchmarks, deployed API. **Metrics:** accuracy/F1 per class, calibration, latency on CPU. **Stack:** PyTorch, torchvision/timm, ONNX Runtime, FastAPI, Docker. **Results/bullets:** TODO. **Risk:** training on CPU is slow — small model, small image size, document it.

---

## Periodic tasks
- Every ~2 weeks once PaperTrail has numbers: GitHub profile README draft, repo descriptions, pinned-repo strategy, LinkedIn drafts (launch, deep-dive, results, demo, lessons), resume bullets.
- Each finished project: ADRs in `docs/adr/`, README with results table, LEARNING_LOG entry.
