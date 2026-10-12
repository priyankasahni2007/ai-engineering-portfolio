# Learning Log

_One dated entry per daily run. Only the important ideas, written so you can explain them in an interview._

---

## 2026-10-11 — Day 1: Reading the hiring market like an engineer

Today's output: `research/market-2026.md`. No code today. These are the concepts that came out of the research. You will be asked about them, and they shape every project we build.

### 1. Retrieval metrics: Recall@K, MRR, nDCG
**What:** Numbers that score how good a search/retrieval step is, given a set of questions with known relevant documents.
- **Recall@K:** of all relevant docs, what fraction appear in the top K results? ("Did we find it at all?")
- **MRR (Mean Reciprocal Rank):** for each query, take 1 / (rank of the first relevant result), then average. A first hit at rank 1 scores 1.0 and at rank 4 scores 0.25. ("How high is the first good result?")
- **nDCG@K:** rewards relevant docs more when they are ranked higher (log discount), normalized by the best possible ordering. It handles graded relevance (very relevant vs somewhat relevant).

**Why it matters:** In RAG the LLM can only answer from what retrieval hands it. Most "the chatbot hallucinated" bugs are actually retrieval misses. RAG appeared in ~36% of 1,000+ AI-engineer job descriptions, and "measure retrieval quality" is a recurring ask.

**Interview Qs**
- *Q: Recall@10 is high but answers are still bad. Why?* A: The right chunk is retrieved but ranked low or drowned by noise. The LLM may also ignore it ("lost in the middle"), or the chunk is too small to hold the answer. Check MRR/nDCG, add a reranker, or fix chunking.
- *Q: When would you use nDCG instead of MRR?* A: When there are several relevant docs or graded relevance. MRR only looks at the first relevant hit.
- *Q: Where do the relevance labels come from?* A: A hand-labelled golden set, existing structure (e.g. a paper's own abstract as the target for a question generated from it), or LLM-generated labels that are spot-checked by a human. Always report which one you used.

### 2. Evals are the scarce skill
**What:** An eval is a repeatable test of an AI system: a fixed dataset, a metric, and a script that produces a number you can compare across versions.
**Why:** Postings (Snowflake, Sarvam) explicitly ask for evals, and hiring-manager posts say most candidates skip them. Without evals you can't say whether a change helped. That is the difference between a demo and engineering.
**Concept: faithfulness.** Does every claim in the answer come from the retrieved sources? This is different from *correctness* (is it true?). You can measure it with NLI models or LLM-as-judge, but judges have biases (they prefer longer answers and their own outputs), so you validate the judge against a small human-labelled set.

**Interview Qs**
- *Q: How do you evaluate an LLM feature with no labelled data?* A: Build a small golden set (50–200 examples) by hand or by synthetic generation plus human review. Pick metrics per component (retrieval vs generation), run it in CI, and track regressions.
- *Q: What's wrong with LLM-as-judge?* A: Position and verbosity bias, self-preference, and cost. Mitigate by randomizing order, using rubric-based binary checks, and measuring agreement with human labels.

### 3. Hybrid search: BM25 + dense embeddings
**What:** BM25 is keyword scoring (term frequency with saturation, inverse document frequency, and length normalization). Dense retrieval embeds the query and docs as vectors and uses nearest neighbours (e.g. with FAISS). Hybrid merges both result lists, often with Reciprocal Rank Fusion (RRF).
**Why:** BM25 is good at exact terms (model names, acronyms like "LoRA", "KV-cache"). Dense retrieval is good at paraphrases. Scientific papers have both kinds of query, which is why PaperTrail uses hybrid search.

**Interview Q**
- *Q: Why RRF instead of adding the scores?* A: BM25 and cosine scores are on different, unbounded scales. RRF uses only ranks (score = Σ 1/(k + rank)), so it needs no score calibration.

### 4. Reading postings: "required" vs "preferred"
Google's Summer 2027 SWE intern posting *requires* DSA and software design and only *prefers* ML. So DSA practice runs in parallel with the portfolio. The projects make you stand out, but DSA gets you through the screen.

---

## 2026-10-12 — Day 2: choosing projects like an engineer

### 1. Scoring ideas against a rubric (and why PaperTrail still comes first)
**What:** I scored 17 project ideas on 10 criteria (hiring signal, depth, measurability, etc.) in `research/project-ranking.md`. **Why:** picking projects by gut feeling leads to a portfolio of whatever was fun. A rubric makes the tradeoffs visible and lets you explain them. **Concept:** scores are *judgments*, not data — I say so in the file. Raw rank ≠ build order: PaperTrail is built first because of timing (it's already on your resume) and signal strength (RAG + evals), not just the sum.
**Interview Qs**
- *Q: How did you decide what to build?* A: I read real postings, listed repeating skills, scored candidate projects on hiring signal and measurability, and sequenced by when internship applications open.
- *Q: Why one deep project instead of ten small ones?* A: Depth lets you show measurement and tradeoffs, which is what interviewers probe; ten tutorials show only that you can follow instructions.

### 2. Offline vs online evaluation (preview for the recsys project)
**What:** Offline = metrics on historical data (nDCG, Recall@K). Online = real users in an A/B test. **Why it matters:** a model that wins offline can lose online (position bias, feedback loops), so big-tech MLE interviews ask about the gap. Interleaving mixes two rankers' results in one list and checks which one users click, needing far less traffic than a classic A/B test.
**Interview Qs**
- *Q: Why use a time-based split for recommendations?* A: Random splits leak future behaviour into training and overstate quality.
- *Q: Why can offline metrics mislead?* A: Logged data only shows what the old system chose to display, so it is biased toward items it already showed.

### 3. Why KV cache is on the roadmap (preview)
When a transformer generates token n+1 it needs the keys and values of all earlier tokens. Without a cache you recompute them every step (quadratic work overall); with a cache you store them and compute only the new token's — trading memory for speed. That tradeoff (and quantization to shrink it) is the core of the LLM inference lab.
- *Q: Is decoding compute-bound or memory-bound?* A: Mostly memory-bound at small batch sizes: each step moves the weights and the cache through memory for little arithmetic, which is why batching helps throughput.
