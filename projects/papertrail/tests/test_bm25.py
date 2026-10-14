import math

from papertrail.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_keeps_hyphenated_terms_and_drops_stopwords():
    assert tokenize("The KV-cache is fast, and LoRA works.") == [
        "kv-cache",
        "fast",
        "lora",
        "works",
    ]


def test_idf_rare_term_higher_and_matches_formula():
    idx = BM25Index().build(["a", "b", "c"], ["rare common", "common", "common"])
    assert idx.idf("rare") > idx.idf("common") > 0
    assert math.isclose(idx.idf("rare"), math.log(1 + (3 - 1 + 0.5) / (1 + 0.5)))


def test_hand_computed_score():
    # One doc, one term, tf=2, len=2 == avgdl -> norm=1. N=1, df=1.
    idx = BM25Index(k1=1.5, b=0.75).build(["a"], ["x x"])
    expected = math.log(1 + 0.5 / 1.5) * 2 * 2.5 / (2 + 1.5)
    assert math.isclose(idx.search("x")[0][1], expected)


def test_ranking_prefers_matching_and_shorter_doc():
    idx = BM25Index().build(
        ["long", "short", "none"],
        ["sparse retrieval " + "filler " * 40, "sparse retrieval", "dense embeddings"],
    )
    res = idx.search("sparse retrieval")
    assert [d for d, _ in res] == ["short", "long"]  # length normalisation; "none" not returned


def test_repeated_query_terms_do_not_double_count():
    idx = BM25Index().build(["a", "b"], ["x y", "y y"])
    assert idx.search("x") == idx.search("x x x")


def test_unknown_terms_and_empty_index():
    assert BM25Index().build(["a"], ["hello"]).search("zzz") == []
    assert BM25Index().build([], []).search("x") == []


def test_k_limits_results():
    idx = BM25Index().build(list("abc"), ["t", "t", "t"])
    assert len(idx.search("t", k=2)) == 2
