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


---

## 2026-10-15 — Day 5: PaperTrail M3 (dense retrieval + hybrid fusion)

### 1. Dense retrieval and cosine similarity (`retrieval/dense.py`)
**What:** an embedding model turns each abstract into a vector (e.g. 384 numbers); a query is embedded the same way and we return the nearest vectors. **Why:** BM25 only matches words, so "reducing memory of attention" misses a paper saying "KV-cache compression". Embeddings place similar meanings close together. **Concept:** we L2-normalise every vector, so the inner product equals cosine similarity (angle between vectors, ignoring length), and FAISS `IndexFlatIP` computes it exactly by brute force. At ~10k documents that is milliseconds; approximate indexes (HNSW/IVF) are only worth their recall loss at millions of vectors.
**Interview Qs**
- *Q: Why normalise embeddings?* A: Then inner product = cosine, so a fast inner-product index gives cosine ranking and long/short texts are comparable.
- *Q: Flat vs HNSW?* A: Flat is exact but O(N) per query; HNSW is a graph index with sub-linear search but approximate and with memory/build cost. Choose by corpus size and recall target.
- *Q: Where does dense retrieval fail?* A: Rare exact terms, names, numbers, acronyms it never saw in training; that is why we keep BM25.

### 2. Reciprocal Rank Fusion (`retrieval/hybrid.py`)
**What:** merge BM25's and dense's ranked lists with `score(d) = sum 1/(60 + rank)`. **Why:** BM25 scores (e.g. 14.2) and cosines (e.g. 0.71) are on different scales, so adding them is meaningless without tuning. RRF uses only ranks: a doc ranked high by both wins, and one list's outlier score cannot dominate. The constant 60 damps the gap between rank 1 and rank 2. A test checks the formula by hand.
**Interview Qs**
- *Q: Why RRF rather than a weighted score sum?* A: No score calibration or tuning needed and robust; weighted sums can be better when tuned on labelled data, so we expose weights and will test it.
- *Q: What does c do?* A: Larger c flattens the rank contribution so lower ranks matter more.

### 3. Designing for a blocked network: the Embedder interface
`DenseIndex` accepts anything with `encode()`. Tests use a deterministic hashing embedder (word overlap only — explicitly *not* semantic), real runs use sentence-transformers. Concept: dependency inversion; also an honesty guard — numbers from the fake embedder must never be presented as dense-retrieval results.
- *Q: Why not mock the model in tests?* A: A small real fake exercising the real FAISS path catches shape/normalisation bugs that a mock would hide.

**Honest status:** mechanics are tested (34 tests pass); no retrieval quality numbers exist yet because arxiv.org and huggingface.co are still unreachable from the sandbox.

---

## 2026-10-16 — PaperTrail M4: reranking, query rewriting, eval runner

### 1. Reranking with a cross-encoder (`retrieval/rerank.py`)
**What:** take the top-N results of a fast first stage and re-score each (query, document) pair with a slower, more accurate model. **Why:** a bi-encoder embeds the query and document *separately* and compares vectors, so it can't model fine interactions between specific query and document words. A cross-encoder feeds both into one transformer, which is more accurate but needs one forward pass per candidate, so you can't run it on 10,000 papers; you run it on the ~50 the cheap stage returned. **Concept:** "retrieve then rerank" — a recall-oriented first stage, a precision-oriented second stage. Raising N helps recall of the reranker input but increases latency linearly.
**Interview Qs**
- *Q: Bi-encoder vs cross-encoder?* A: Bi-encoder encodes texts independently (documents can be pre-indexed, fast); cross-encoder encodes the pair jointly (accurate, no pre-computation, slow).
- *Q: Why not rerank everything?* A: Cost scales with corpus size × model cost; reranking top-N bounds latency.
- *Q: Reranker can't fix which failure?* A: A relevant document missing from the top-N — so measure first-stage Recall@N.
- *Q: How would you pick N?* A: Sweep N, plot nDCG vs p95 latency, pick the knee.

### 2. Query rewriting (`retrieval/rewrite.py`)
**What:** transform the query before search; here, expanding abbreviations ("LoRA" -> "low-rank adaptation"). **Why:** BM25 only matches literal tokens, so a query "lora" finds nothing in a paper that only says "low-rank adaptation" (a test proves it). **Trade-off:** expansion raises recall but can add noise; an LLM rewriter is more flexible but adds cost, latency and nondeterminism. So it must be evaluated as an ablation, not assumed to help.
**Interview Qs**
- *Q: When can query expansion hurt?* A: Ambiguous terms drag in off-topic documents, lowering precision.
- *Q: Why wrap a retriever instead of editing it?* A: Composability and clean ablations (same interface, swap or remove stages).

### 3. Eval runner with latency percentiles (`eval/runner.py`)
**What:** runs each configuration over the golden set and reports Recall@K/MRR/nDCG plus p50/p95 latency. **Why p95:** the mean hides slow outliers; users feel the tail. Quality without latency/cost is an incomplete engineering story.
**Interview Q:** *Q: Why report p95 rather than average latency?* A: A few slow queries dominate user-perceived quality and SLAs; averages hide them.

**Honest status:** 40 tests pass, ruff clean. arxiv.org / huggingface.co are still unreachable from the sandbox (re-tested today), so there is still no real corpus, no real model, and **no retrieval numbers**. OverlapReranker / HashingEmbedder are test stand-ins, not semantic.


---

## 2026-10-17 — PaperTrail M5: grounded generation, citations, faithfulness

### 1. Grounded generation with numbered passages (`generation/prompt.py`)
**What:** retrieved passages are numbered `[1]..[k]` in the prompt and the model is told to answer only from them and cite `[n]` after each claim. **Why:** retrieval alone returns documents; a research assistant must answer, and users must be able to check the answer. Numbered citations give a machine-checkable link from claim to source. **Concept:** RAG = retrieve, then condition generation on the evidence; "abstain if not found" reduces made-up answers.

### 2. Citation validity vs faithfulness (`generation/citations.py`)
**What:** `validate_citations` checks every `[n]` refers to a passage that was actually retrieved and counts uncited sentences. `faithfulness` is the share of sentences whose content words are mostly (>=60%) found in the passages they cite. **Why two metrics:** a model can cite `[1]` correctly-formatted yet say something passage 1 never said. Tests show a *true* sentence with the *wrong* citation scores 0. **Honest limit:** word overlap rewards copying and misses paraphrase/negation ("X does not help" overlaps "X helps"). It's a cheap CI-safe proxy, to be replaced by NLI or an LLM judge calibrated on human labels.
**Bug found by tests today:** my extractive generator wrote `sentence. [1]`; the splitter saw `[1]` as its own uncited sentence and faithfulness came out 0.5. Fix: cite before the full stop. Lesson: define output conventions once, in the prompt, parser and tests.
**Interview Qs**
- *Q: Recall@10 is high but answers are bad. Why?* A: Retrieval found the evidence but generation ignored or distorted it, the context was too long/noisy, or the passage needed was ranked low. Measure retrieval and faithfulness separately.
- *Q: How do you evaluate faithfulness?* A: Split the answer into claims, check each against the cited evidence with NLI or an LLM judge; validate the judge on a small human-labelled set because judges have biases (verbosity, self-preference).
- *Q: Why not just trust the model's citations?* A: Models invent citation numbers and attach real ones to unsupported claims; validate the index and check the support.
- *Q: What's the weakness of token-overlap metrics?* A: They can't see meaning: paraphrases score low, contradictions with the same words score high.

**Honest status:** 49 tests pass, ruff clean. Network (arxiv.org / huggingface.co / semanticscholar) still unreachable, so there is no real corpus and no real LLM; the extractive generator is a test stand-in. No results yet.

---

## 2026-10-17 (evening) — PaperTrail's first real numbers: what they teach

Network access was opened, so everything M2–M5 built was finally run on real data: 2,000 cs.CL abstracts and two 200-query golden sets. Results are in `projects/papertrail/README.md`. The concepts below are the ones to be able to explain.

### 1. Synthetic golden sets (doc2query) and their bias
**What:** We had no human-labelled queries, so we generated them. A T5 model trained on MS MARCO reads each abstract and writes a question a user might ask. That abstract becomes the one "relevant" answer.
**Why:** It is the cheapest way to get hundreds of labelled queries. Real research systems do the same (InPars, GPL).
**The catch:** the question is written *from the abstract's own words*, so it shares vocabulary with it. That favours keyword search (BM25). The labels are also incomplete: only one paper counts as relevant, even if three others also answer the question. Scores are therefore **lower bounds**, and vague queries are punished.
**Beam search vs sampling:** beam search picks the most *probable* output, and for this model that collapsed to bland "what is X" questions. Sampling (top-k/top-p) gives more varied, realistic queries. With a fixed seed it is still reproducible.

**Interview Qs**
- *Q: Your eval set is synthetic. Why should I trust the numbers?* A: Trust them for *comparing configs on the same set*, not as absolute quality. I used two query types with different biases (titles and doc2query), reported the biases, and the conclusions agree on both. The next step is a small hand-labelled set to check them.
- *Q: What is a false negative in IR evaluation?* A: A relevant document that isn't labelled relevant. Retrieving it counts as a miss, so metrics are biased downward. That bias hits systems that find *different* relevant documents from the ones the labels came from.

### 2. Why hybrid helped recall but not MRR
RRF adds 1/(60+rank) from each retriever. A paper ranked 3rd by BM25 and 40th by dense can lose to one ranked 8th by both. Fusion is good at getting the right paper *somewhere* in the top 10 (Recall@10 0.825 → 0.875). It is bad at putting it *first* (MRR 0.716 → 0.703), because neither list's confidence is used, only ranks.

- *Q: How would you make hybrid improve MRR too?* A: Tune RRF weights or `c` on a dev set, use score-based fusion with normalisation, or add a reranker (which is what worked here).

### 3. Cross-encoder vs bi-encoder (the big result)
- A **bi-encoder** (bge-small) embeds the query and each document *separately*. Documents are embedded once and offline, so search is a fast vector lookup (~15 ms).
- A **cross-encoder** reads the query and document *together*, so every query word can attend to every document word. It is much more accurate but has to run once per (query, doc) pair at query time: 50 pairs ≈ 2 s on CPU.
- Hence the **two-stage design**: a cheap retriever narrows 2,000 papers to 50, then the expensive reranker orders those 50. MRR went 0.716 → 0.838.

**Interview Qs**
- *Q: Why not use the cross-encoder on all 2,000 docs?* A: Cost grows linearly with corpus size: 2,000 forward passes ≈ 80 s per query here. Two-stage keeps the cost fixed at top_n pairs.
- *Q: 2 s per query is too slow. What would you do?* A: Rerank fewer candidates (top 20), use a smaller or distilled reranker, run it with ONNX/int8 quantization, batch on a GPU, or cache frequent queries. Then measure the quality-vs-latency curve instead of guessing.

### 4. "No improvement" is a result
Query rewriting (expanding "LLM" → "large language model") didn't help and slightly hurt hybrid MRR (−0.014). Reporting that honestly, and switching the feature off, shows more engineering judgement than claiming every component helps.
