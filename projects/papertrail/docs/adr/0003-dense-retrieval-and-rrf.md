# ADR 0003: Dense retrieval behind an interface; exact FAISS index; RRF fusion

Status: accepted (2026-10-15)

## Decisions
1. **`Embedder` protocol.** `DenseIndex` depends on `encode(texts) -> float32 array`, not on a model. Real model: `SentenceTransformerEmbedder` (default `BAAI/bge-small-en-v1.5`, optional extra `embed`). Offline/test model: `HashingEmbedder` (word-overlap only, **not semantic**). Reason: huggingface.co is unreachable from the build sandbox and tests must not download weights.
2. **`IndexFlatIP` on normalised vectors** = exact cosine search. At ~10k docs x 384 dims a query is a ~4M-float matmul; approximate indexes would trade recall for speed we do not need. Revisit (HNSW) only if the corpus grows past ~1M.
3. **Reciprocal Rank Fusion (c=60)** instead of score blending: BM25 scores and cosines live on different scales; RRF uses ranks only and needs no tuning. Weights are exposed to test alternatives in the evaluation harness.

## Consequences
No dense/hybrid number may be reported until a real embedder has been run on a real corpus. Tests only prove mechanics.
