import numpy as np
import pytest

from papertrail.retrieval.bm25 import BM25Index
from papertrail.retrieval.dense import DenseIndex, HashingEmbedder
from papertrail.retrieval.hybrid import HybridRetriever, rrf_fuse

DOCS = {
    "a": "low-rank adaptation lora fine-tuning of large language models",
    "b": "kv-cache memory optimisation for transformer inference",
    "c": "convolutional networks for image classification",
}


def test_hashing_embedder_deterministic_and_shape():
    e = HashingEmbedder(dim=64)
    x, y = e.encode(["lora tuning"]), e.encode(["lora tuning"])
    assert x.shape == (1, 64) and x.dtype == np.float32
    assert np.array_equal(x, y)


def test_dense_search_finds_overlap_and_scores_are_cosine():
    idx = DenseIndex(HashingEmbedder(512)).build(list(DOCS), list(DOCS.values()))
    top = idx.search("kv-cache transformer inference", k=3)
    assert top[0][0] == "b"
    assert all(-1.0001 <= s <= 1.0001 for _, s in top)
    # identical text -> cosine 1
    assert idx.search(DOCS["a"], k=1)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_empty_and_k_larger_than_corpus():
    assert DenseIndex(HashingEmbedder()).build([], []).search("x") == []
    idx = DenseIndex(HashingEmbedder()).build(["a"], [DOCS["a"]])
    assert len(idx.search("lora", k=50)) == 1


def test_dense_length_mismatch():
    with pytest.raises(ValueError):
        DenseIndex(HashingEmbedder()).build(["a"], [])


def test_rrf_hand_computed():
    fused = dict(rrf_fuse([["x", "y"], ["y", "z"]], c=60))
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)
    assert rrf_fuse([["x", "y"], ["y", "z"]])[0][0] == "y"  # agreement wins


def test_rrf_weights_and_validation():
    top = rrf_fuse([["x"], ["y"]], weights=[1.0, 2.0])[0][0]
    assert top == "y"
    with pytest.raises(ValueError):
        rrf_fuse([["x"]], weights=[1.0, 1.0])


def test_rrf_deterministic_ties():
    assert [d for d, _ in rrf_fuse([["x"], ["y"]])] == ["x", "y"]


def test_hybrid_combines_both_retrievers():
    ids, texts = list(DOCS), list(DOCS.values())
    h = HybridRetriever(
        BM25Index().build(ids, texts), DenseIndex(HashingEmbedder(512)).build(ids, texts)
    )
    out = h.search("lora fine-tuning", k=2)
    assert out[0][0] == "a" and len(out) == 2
