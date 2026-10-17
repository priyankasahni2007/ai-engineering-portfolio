# ADR 0006: First real-corpus evaluation (synthetic golden sets, abstract-only index)

Status: accepted (2026-10-17)

## Context
Network access to export.arxiv.org and huggingface.co became available on 2026-10-17. Before this, M2–M5 were tested only on fixtures.

## Decisions
1. **Corpus:** 2,000 newest cs.CL abstracts (published 2026-09-29 to 2026-10-15). The arXiv API returns newest first, so a later fetch gives a different corpus. The exact ID list is committed in `eval_data/corpus_ids.txt`.
2. **Index abstracts only, not titles.** Otherwise the title golden set is trivial, because the query would appear verbatim in the document.
3. **Two synthetic golden sets, 200 queries each, seed 42, one relevant paper per query:**
   - `titles`: the title is the query (known-item search, real author phrasing).
   - `doc2query`: a question generated from the abstract by `BeIR/query-gen-msmarco-t5-base-v1` with top-k/top-p sampling. Beam search was tried first and rejected: all 8 inspected queries had the form "what is X". With sampling, 113 of 200 queries still start with "what is", but more of them are specific.
4. **No hand labels yet.** Each query has exactly one labelled relevant paper. Other papers that are also relevant count as misses, so all scores are **lower bounds**, and vague queries are penalised hardest.
5. **Models:** `BAAI/bge-small-en-v1.5` embeddings (no query instruction prefix) and `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker over the hybrid top 50. Both are small enough for CPU.

## Consequences
- Results are in `results/retrieval.{json,md}` and the README.
- Known biases: doc2query queries are generated from the abstract, which favours lexical overlap and so favours BM25. Many queries contain a paper's unique system name ("forkgrepo"), which also favours BM25. Dense retrieval losing to BM25 here should not be generalised to paraphrased user questions.
- Not yet done: confidence intervals or a paired significance test (the runner does not keep per-query results), a hand-labelled set, and a scale-up to 10k+ papers.
