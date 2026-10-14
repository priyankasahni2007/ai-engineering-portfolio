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

---

## 2026-10-13 — Day 3: PaperTrail M1 (skeleton, CI, ingestion)

### 1. Pure parsing vs. network code (testable ingestion)
**What:** `arxiv.parse_feed` turns Atom XML text into `Paper` records; `arxiv.fetch` only downloads and pages. **Why:** network calls are slow and flaky, so the logic that can be wrong (cleaning text, ids, duplicates) is a pure function tested on a saved fixture. In this sandbox arXiv is blocked, and the tests still prove the parser works. **Concept:** separate I/O from logic. Details that matter: arXiv hard-wraps text (collapse whitespace), ids carry versions (`2401.01234v2` → `2401.01234`, or two versions of one paper become two documents), paging can overlap (dedupe), and error responses come back as fake "entries" (skip them).
**Interview Qs**
- *Q: Why strip the version from the id?* A: The version is not a different paper; keeping it duplicates documents and would later let the same paper count twice as a "relevant" hit in evaluation.
- *Q: How would you test code that calls an external API?* A: Isolate parsing in a pure function, test it on recorded fixtures; mock or skip the thin network layer; add a small manual/integration check.

### 2. Why Parquet with an explicit schema
**What:** a columnar, compressed, typed file format. **Why:** later stages (BM25, embeddings) reload thousands of records often; Parquet is smaller and faster than JSON and keeps list columns (authors, categories). An explicit schema means an empty batch still has the right column types (there is a test for exactly that).
- *Q: Parquet vs CSV?* A: Typed, nested columns, compression, and column pruning; CSV loses types and can't hold lists cleanly.

### 3. `src/` layout and CI
**What:** code in `src/papertrail/`, installed with `pip install -e .`; GitHub Actions runs the same `make lint test` on Python 3.10 and 3.12. **Why:** with a flat layout, tests can import code straight from the working folder and hide packaging bugs. CI is your promise that the main branch always works; matching local and CI commands avoids "works on my machine". (Small gotcha I hit: the `pytest` command was in a different Python environment, so `python -m pytest` is the safer form — it uses the interpreter you actually installed into.)
- *Q: What does CI give a solo project?* A: A reproducible clean-environment check on every push, which also tells reviewers the repo is trustworthy.

**Honest status:** no retrieval or results yet — this milestone is plumbing. Real arXiv download is still TODO because this sandbox can't reach arxiv.org.


---

## 2026-10-14 — Day 4: PaperTrail M2 (BM25 + evaluation metrics)

### 1. BM25 (`retrieval/bm25.py`)
**What:** a keyword ranking function. For each query word it adds `idf * saturated_tf`, summed over words. **Why:** exact-term matching ("LoRA", "KV-cache") is something dense embeddings blur, so BM25 is the baseline everything else must beat. **Concept:** three ideas. (1) *IDF*: rare words carry more information than common ones. (2) *TF saturation* (`k1`): the 10th occurrence of a word adds much less than the 1st. (3) *Length normalisation* (`b`): a long document mentions words more by chance, so it is penalised. A test checks a hand-computed score, and another checks that a short document beats a long one with the same matches.
**Interview Qs**
- *Q: What do k1 and b control?* A: k1 sets how fast term frequency saturates; b sets how strongly document length is penalised (0 = none, 1 = full).
- *Q: Why does BM25 beat TF-IDF?* A: TF-IDF grows linearly with term count and ignores length; BM25 saturates and normalises.
- *Q: When does BM25 fail?* A: Vocabulary mismatch: "car" vs "automobile". That is the case for dense/hybrid retrieval.
- *Q: Why an inverted index?* A: Only documents that contain a query term are touched, not the whole corpus.

### 2. Recall@K vs MRR vs nDCG (`eval/metrics.py`)
**What:** three ways to score a ranked list against known-relevant docs. **Recall@K:** what share of the relevant docs is in the top K (good for "did the reranker/LLM get a chance to see it?"). **MRR:** 1/rank of the first relevant hit (good when one answer is enough). **nDCG@K:** rewards relevant docs more the higher they sit, via a `1/log2(rank+1)` discount, divided by the best possible score so it lies in 0..1. **Why it matters:** a retriever can have high Recall@10 but poor MRR, meaning the answer is found but buried; that is exactly what a reranker fixes.
**Interview Qs**
- *Q: Recall@10 is high but answers are bad. Why?* A: Relevant passage is retrieved but ranked low or the generator ignores it; check MRR/nDCG, reranking, and faithfulness.
- *Q: Why normalise DCG?* A: So queries with different numbers of relevant docs are comparable and can be averaged.
- *Q: Risk of synthetic golden sets?* A: Queries generated from an abstract share its words, so BM25 looks unrealistically good; report them separately from human-labelled ones (ADR 0002).

**Honest status:** no retrieval results yet. arxiv.org and huggingface.co are still blocked from the sandbox, so there is no real corpus. Everything built today is verified by unit tests only; no performance number is claimed.
