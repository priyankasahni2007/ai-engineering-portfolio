import pytest

from papertrail.eval.golden import load_golden, run_retriever
from papertrail.eval.metrics import evaluate
from papertrail.retrieval.bm25 import BM25Index


def test_load_and_end_to_end(tmp_path):
    f = tmp_path / "g.jsonl"
    f.write_text(
        '{"qid":"q1","query":"sparse retrieval","relevant":["a"]}\n\n'
        '{"qid":"q2","query":"dense embeddings","relevant":["b"]}\n'
    )
    queries, qrels = load_golden(f)
    idx = BM25Index().build(["a", "b"], ["sparse retrieval methods", "dense embeddings models"])
    out = evaluate(run_retriever(idx.search, queries, k=5), qrels, ks=(1,))
    assert out["recall@1"] == 1.0 and out["mrr"] == 1.0


@pytest.mark.parametrize(
    "body",
    [
        '{"qid":"q","query":"x","relevant":["a"]}\n{"qid":"q","query":"y","relevant":["a"]}',
        '{"qid":"q","query":"x","relevant":[]}',
    ],
)
def test_rejects_bad_golden(tmp_path, body):
    f = tmp_path / "g.jsonl"
    f.write_text(body)
    with pytest.raises(ValueError):
        load_golden(f)
