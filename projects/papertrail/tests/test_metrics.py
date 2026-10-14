import math

import pytest

from papertrail.eval.metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = ["d1", "d2", "d3", "d4"]


def test_recall():
    assert recall_at_k(RANKED, {"d2", "d9"}, 2) == 0.5
    assert recall_at_k(RANKED, {"d4"}, 3) == 0.0
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_mrr():
    assert reciprocal_rank(RANKED, {"d3", "d4"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"zz"}) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 2 and 4: DCG = 1/log2(3) + 1/log2(5); ideal (2 rel) = 1 + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, {"d2", "d4"}, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["d1"], {"d1"}, 5) == 1.0
    assert ndcg_at_k(RANKED, {"d4"}, 3) == 0.0


def test_ndcg_perfect_ranking_is_one_even_with_more_relevant_than_k():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c", "d"}, 2) == pytest.approx(1.0)


def test_evaluate_means_and_missing_query():
    qrels = {"q1": {"d1"}, "q2": {"d9"}}
    out = evaluate({"q1": RANKED}, qrels, ks=(2,))
    assert out["recall@2"] == 0.5 and out["mrr"] == 0.5 and out["ndcg@2"] == 0.5
    with pytest.raises(ValueError):
        evaluate({}, {})
