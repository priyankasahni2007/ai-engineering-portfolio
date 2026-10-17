from papertrail.generation.citations import (
    extract_citations,
    faithfulness,
    sentence_support,
    split_sentences,
    validate_citations,
)
from papertrail.generation.generator import ExtractiveGenerator, answer_question
from papertrail.generation.prompt import Passage, build_prompt, make_passages
from papertrail.retrieval.bm25 import BM25Index

TEXTS = {
    "a": "LoRA adds low-rank adapters to frozen weights. It cuts trainable parameters a lot.",
    "b": "A KV cache stores past keys and values. It speeds up autoregressive decoding.",
}
P = [Passage(1, "a", TEXTS["a"]), Passage(2, "b", TEXTS["b"])]


def test_make_passages_numbers_from_one_and_skips_missing():
    ps = make_passages([("a", 1.0), ("zzz", 0.9), ("b", 0.5)], TEXTS)
    assert [(p.n, p.doc_id) for p in ps] == [(1, "a"), (2, "b")]


def test_prompt_contains_numbered_passages_and_question():
    pr = build_prompt("What is LoRA?", P)
    assert "[1] LoRA adds" in pr and "[2] A KV cache" in pr and "Question: What is LoRA?" in pr


def test_extract_and_split():
    assert extract_citations("x [1][3]. y [2].") == [1, 3, 2]
    assert split_sentences("First one [1]. Second one [2].") == [
        "First one [1].",
        "Second one [2].",
    ]


def test_validate_flags_hallucinated_citation_and_uncited_sentence():
    r = validate_citations("LoRA is small [1]. It is great [7]. No cite here.", P)
    assert r.valid == [1] and r.invalid == [7]
    assert r.uncited_sentences == 1 and r.n_sentences == 3
    assert r.validity_rate == 0.5


def test_validity_rate_no_citations_is_one():
    assert validate_citations("nothing", P).validity_rate == 1.0


def test_sentence_support_fraction():
    assert sentence_support("LoRA adds low-rank adapters", TEXTS["a"]) == 1.0
    assert sentence_support("quantum banana", TEXTS["a"]) == 0.0


def test_faithfulness_separates_supported_from_invented():
    good = "LoRA adds low-rank adapters to frozen weights [1]."
    bad = "LoRA was invented by aliens in 1850 [1]."
    wrong_cite = "A KV cache stores past keys and values [1]."  # true, but passage 1 lacks it
    no_cite = "LoRA adds low-rank adapters to frozen weights."
    assert faithfulness(good, P) == 1.0
    assert faithfulness(bad, P) == 0.0
    assert faithfulness(wrong_cite, P) == 0.0
    assert faithfulness(no_cite, P) == 0.0
    assert faithfulness(good + " " + bad, P) == 0.5
    assert faithfulness("", P) == 0.0


def test_extractive_generator_end_to_end_is_faithful_with_valid_citations():
    idx = BM25Index().build(list(TEXTS), list(TEXTS.values()))
    a = answer_question("how does a kv cache speed decoding", idx, TEXTS, ExtractiveGenerator())
    assert a.passages[0].doc_id == "b"
    assert a.citations.invalid == [] and a.citations.uncited_sentences == 0
    assert a.faithfulness == 1.0
    assert a.text.endswith("[1].")


def test_extractive_generator_abstains_without_passages():
    assert "cannot answer" in ExtractiveGenerator().generate("q", [])


def test_citation_placed_after_period_is_flagged_as_uncited_sentence():
    # Convention: cite BEFORE the full stop ("... [1]."). "... . [1]" splits into a stray sentence.
    r = validate_citations("A KV cache stores keys. [2]", P)
    assert r.n_sentences == 2 and r.uncited_sentences == 1
