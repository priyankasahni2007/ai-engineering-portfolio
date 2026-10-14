# ADR 0002 — Own BM25, binary-relevance metrics, golden-set format

**Status:** accepted (2026-10-14)

- **Own BM25 instead of `rank_bm25`:** ~50 lines, lets us unit-test every term against hand-computed values, and is a better interview story. It uses the non-negative Lucene idf. If speed becomes a problem at 10k+ docs, swap in a library and keep our tests as the contract.
- **Tokenizer keeps hyphen/dot-joined terms** ("kv-cache", "bert-base") because ML jargon matters; a tiny stoplist, no stemming yet (to be tried and measured, not assumed).
- **Binary relevance** (relevant or not) for Recall@K, MRR, nDCG@K — simple to label; graded labels are a possible later extension.
- **Golden set:** JSONL, `{"qid","query","relevant":[arxiv_id,...]}`. Label sources, in order of trust: (1) hand-labelled by Priyanka on a sample, (2) synthetic queries generated from a paper's abstract with that paper as the relevant doc (cheap but biased towards lexical overlap — BM25 will look better than it is; this must be stated in the README). Both kinds are reported separately.
