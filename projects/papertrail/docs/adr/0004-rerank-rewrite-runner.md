# ADR 0004: Reranker and rewriter as wrappers with the same `search` interface; latency-aware eval runner

Status: accepted (2026-10-16)

## Decisions
1. **Wrappers, not special cases.** `RerankedRetriever` and `RewritingRetriever` take any searcher and expose `search(query, k)`. Any pipeline (BM25 -> rewrite -> hybrid -> rerank) is composed, and the eval runner treats every configuration identically, so each stage's gain is measured by an ablation rather than assumed.
2. **Rerank only top-N (default 50).** A cross-encoder costs one forward pass per candidate; N bounds p95 latency. N is a tunable the runner can sweep (quality vs latency).
3. **Rule-based rewriting first.** Deterministic abbreviation expansion fixes a real BM25 failure (vocabulary mismatch, tested). LLM rewriting needs an API key and adds cost/latency; it can plug in as any `rewrite(str) -> str` callable and be compared in the same runner.
4. **Runner reports p50/p95 wall-clock latency per query**, since the project's claim is quality *and* latency/cost per configuration.

## Consequences
`OverlapReranker` and `HashingEmbedder` are mechanics-only stand-ins. No reranker/rewrite quality number may be reported until a real cross-encoder runs on a real corpus.
