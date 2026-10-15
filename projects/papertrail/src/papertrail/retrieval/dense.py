"""Dense retrieval: an Embedder interface plus an exact FAISS inner-product index.

Vectors are L2-normalised, so inner product == cosine similarity.
`IndexFlatIP` is exact (brute force); at ~10k abstracts it is fast enough that approximate
indexes (HNSW/IVF) would only add recall loss. See ADR 0003.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from papertrail.retrieval.bm25 import tokenize


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array (len(texts), dim)."""


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. NOT semantic.

    Exists so the dense path can be tested and run offline (no model download). It captures
    word overlap only, so never report its numbers as "dense retrieval" results.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                sign = 1.0 if (h >> 32) & 1 else -1.0  # signed hashing reduces collision bias
                out[row, h % self.dim] += sign
        return out


class SentenceTransformerEmbedder:
    """Wraps a Hugging Face sentence-transformers model (weights download on first use)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True),
            dtype=np.float32,
        )


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)  # zero vectors stay zero instead of NaN


class DenseIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.doc_ids: list[str] = []
        self.index = None

    def build(self, doc_ids: list[str], texts: list[str]) -> DenseIndex:
        import faiss

        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.index = faiss.IndexFlatIP(self.embedder.dim)
        if texts:
            self.index.add(_normalize(self.embedder.encode(texts)))
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        q = _normalize(self.embedder.encode([query]))
        scores, idx = self.index.search(q, min(k, len(self.doc_ids)))
        return [
            (self.doc_ids[i], float(s)) for i, s in zip(idx[0], scores[0], strict=True) if i >= 0
        ]
