# Project idea ranking — 2026-10-12

Scoring is **my judgment, 1–10 per criterion** (not measured data), informed by `research/market-2026.md`. Higher is better for every column, including **Diff** = "right-sized difficulty": hard enough to impress, doable on CPU with ~1 hr/day of Priyanka's review time (so GPU-only or multi-month ideas score low).

Columns: Hire = hiring signal · Depth = technical depth · AI = AI relevance · SWE = software-engineering depth · Orig = originality · Intv = interview discussion potential · Meas = measurable results · Diff = right-sized difficulty · Port = portfolio value · LI = LinkedIn content potential. Total is the plain sum (max 100).

Excluded as beginner (not ranked): Titanic, house prices, MNIST, basic sentiment, basic chatbot, simple PDF chatbot, weather app, generic image classifier, Streamlit wrappers, tutorial copies. The last row is shown only as a scoring floor.

| Rank | Idea | Hire | Depth | AI | SWE | Orig | Intv | Meas | Diff | Port | LI | Total /100 | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LLM inference lab: tiny transformer + KV cache + int8/int4 quantization + batching benchmarks on CPU | 8 | 10 | 9 | 8 | 8 | 10 | 10 | 8 | 9 | 9 | **89** | Premium niche (NVIDIA/vLLM), measurable on CPU |
| 2 | PaperTrail: evaluated hybrid RAG over arXiv (LOCKED #1) | 10 | 8 | 10 | 8 | 6 | 10 | 10 | 7 | 10 | 8 | **87** | RAG+evals = strongest signals (market §1 #2,#3) |
| 3 | Tool-using research agent (LangGraph-style) on top of PaperTrail with eval tasks, tracing, cost caps, HITL | 9 | 8 | 10 | 8 | 7 | 9 | 8 | 7 | 9 | 9 | **84** | Agents rising; reuses #1 |
| 4 | Two-stage recsys/ranker: candidate gen + LTR (LightGBM/PyTorch) on MovieLens/Amazon, offline metrics + simulated A/B / interleaving | 9 | 8 | 7 | 8 | 6 | 9 | 10 | 7 | 9 | 7 | **80** | TikTok/OPPO-style MLE interns |
| 5 | Multimodal paper-figure QA/retrieval (CLIP/SigLIP over arXiv figures) with Recall@K + VQA-style eval | 7 | 8 | 9 | 7 | 8 | 8 | 8 | 7 | 8 | 9 | **79** | Extends PaperTrail; bonus signal |
| 6 | LLM eval framework / LLM-as-judge calibration toolkit (judge vs human agreement) | 8 | 7 | 9 | 7 | 7 | 9 | 9 | 7 | 7 | 8 | **78** | Eval-skill signal; best folded into PaperTrail/agent |
| 7 | Vector DB from scratch (HNSW/IVF-PQ) with recall/latency tradeoff curves | 7 | 9 | 6 | 9 | 7 | 9 | 10 | 4 | 7 | 8 | **76** | Very deep but hard; overlaps FAISS usage |
| 8 | Polished AI product: PaperTrail Web (React/Next frontend, FastAPI, Postgres, auth, analytics, deploy) | 8 | 6 | 8 | 9 | 6 | 7 | 6 | 6 | 9 | 10 | **75** | Product polish; recruiters click demos |
| 9 | Semantic caching + routing gateway for LLMs (cost/latency) | 7 | 6 | 8 | 9 | 7 | 8 | 9 | 6 | 6 | 8 | **74** | Strong systems angle; fold into PaperTrail cost work |
| 10 | From-scratch BPE tokenizer + tokenizer-efficiency study across Indic languages | 6 | 7 | 7 | 6 | 8 | 8 | 8 | 8 | 6 | 8 | **72** | Good small side project / content |
| 11 | Code-repo question-answering with AST-aware chunking | 7 | 7 | 9 | 7 | 6 | 8 | 7 | 6 | 7 | 8 | **72** | RAG variant, overlaps #1 |
| 12 | Synthetic-data fine-tuning (LoRA) of a small LM for a narrow task, with before/after evals | 8 | 7 | 9 | 6 | 6 | 8 | 8 | 4 | 7 | 8 | **71** | Needs GPU; CPU-hostile |
| 13 | Speech: VAD / keyword spotting from scratch with streaming inference (Sarvam-style) | 6 | 8 | 7 | 7 | 7 | 8 | 8 | 7 | 6 | 7 | **71** | Niche, India speech startups |
| 14 | CV: fine-tune a small model on a real dataset (e.g. plant disease / defect detection) + ONNX export + latency benchmarks + deployed API | 7 | 7 | 6 | 7 | 5 | 7 | 9 | 6 | 7 | 7 | **68** | Standard but needed for CV coverage |
| 15 | Fraud detection on imbalanced tabular data with drift monitoring + calibration + cost-sensitive thresholds | 8 | 7 | 5 | 7 | 5 | 8 | 9 | 6 | 7 | 6 | **68** | Classic MLE; common elsewhere |
| 16 | Graph RAG over citation network | 6 | 7 | 9 | 6 | 7 | 7 | 6 | 5 | 6 | 7 | **66** | Hype-adjacent; weak evidence of demand |
| 17 | Time-series demand forecasting with backtesting, hierarchical reconciliation, probabilistic eval | 6 | 7 | 4 | 6 | 5 | 6 | 9 | 6 | 6 | 5 | **60** | Moderate demand |
| 18 | Basic PDF chatbot / sentiment classifier (EXCLUDED example) | 2 | 2 | 4 | 2 | 1 | 2 | 3 | 2 | 2 | 3 | **23** | Excluded: beginner |


## Reading the table
- **PaperTrail (locked #1)** scores 2nd by raw sum only because the LLM-inference idea is cheap to measure and rarely done well by students. PaperTrail stays first: it is already on Priyanka's resume, and RAG + evals are the two strongest market signals. Sequence != rank.
- **Ideas folded into flagships instead of standing alone:** LLM-judge calibration (#6) → PaperTrail faithfulness eval; semantic caching (#9) → PaperTrail cost/latency work; vector-DB-from-scratch (#7) → optional stretch/deep-dive within PaperTrail (HNSW vs flat FAISS recall/latency curve); code-repo QA (#11) and Graph RAG (#16) dropped as RAG variants that dilute the story.
- **Chosen despite a lower raw score:** the CV project (#14) is kept as flagship 6 because the brief requires CV coverage with a real train/eval/deploy loop, and it is also the lane where fine-tuning on CPU is feasible with a small model. Fraud (#15), forecasting (#17), LoRA fine-tuning (#12, GPU-hostile) and speech (#13) are backups.
- **Selected flagships (7):** PaperTrail · Research agent · LLM inference lab · Recsys/ranking · PaperTrail Web product · Multimodal figure retrieval · CV fine-tune+deploy. See `PORTFOLIO_ROADMAP.md`.
