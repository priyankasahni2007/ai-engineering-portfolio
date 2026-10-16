import pytest

from papertrail.eval.runner import compare, run_config, to_markdown
from papertrail.retrieval.bm25 import BM25Index
from papertrail.retrieval.rerank import OverlapReranker, RerankedRetriever
from papertrail.retrieval.rewrite import RewritingRetriever, expand_query

DOCS = {
    "a": "low-rank adaptation of large language model weights",
    "b": "kv cache for transformer inference",
    "c": "image classification with convolutional neural network",
}


class Fixed:
    def __init__(self, order):
        self.order = order

    def search(self, q, k=10):
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(self.order[:k])]


def test_expand_query_adds_long_forms_only_for_known_terms():
    assert expand_query("LoRA tuning") == "LoRA tuning low-rank adaptation"
    assert expand_query("plain words") == "plain words"
    assert expand_query("LLM and CNN").endswith("large language model convolutional neural network")


def test_rewriting_fixes_vocabulary_mismatch_for_bm25():
    bm25 = BM25Index().build(list(DOCS), list(DOCS.values()))
    assert bm25.search("lora", 3) == []  # "lora" never appears in the text
    assert RewritingRetriever(bm25).search("lora", 3)[0][0] == "a"


def test_reranker_reorders_first_stage_and_respects_k():
    rr = RerankedRetriever(Fixed(["c", "b", "a"]), OverlapReranker(), DOCS, top_n=3)
    assert rr.search("low-rank adaptation", 3)[0][0] == "a"
    assert len(rr.search("low-rank adaptation", 2)) == 2


def test_reranker_only_sees_top_n():
    rr = RerankedRetriever(Fixed(["c", "b", "a"]), OverlapReranker(), DOCS, top_n=2)
    assert {d for d, _ in rr.search("low-rank adaptation", 2)} == {"c", "b"}


def test_run_config_metrics_and_latency():
    qs, qrels = {"q1": "x"}, {"q1": {"a"}}
    res = run_config(Fixed(["b", "a"]), qs, qrels, k=2, ks=(1, 2))
    assert res["recall@1"] == 0 and res["recall@2"] == 1 and res["mrr"] == pytest.approx(0.5)
    assert res["latency_p95_ms"] >= res["latency_p50_ms"] >= 0


def test_compare_and_markdown():
    qs, qrels = {"q1": "x"}, {"q1": {"a"}}
    out = compare({"good": Fixed(["a"]), "bad": Fixed(["b"])}, qs, qrels, k=10)
    md = to_markdown(out)
    assert out["good"]["mrr"] == 1 and out["bad"]["mrr"] == 0
    assert "| good |" in md and "| bad |" in md
