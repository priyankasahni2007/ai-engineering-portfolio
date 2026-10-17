"""Generator interface plus an offline extractive generator used for tests and plumbing.

Real LLM wrappers (Anthropic API / local HF model) implement the same `generate(prompt)`; they
need an API key or model weights and are not exercised here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from papertrail.generation.citations import (
    CitationReport,
    faithfulness,
    split_sentences,
    validate_citations,
)
from papertrail.generation.prompt import Passage, build_prompt
from papertrail.retrieval.bm25 import tokenize


class Generator(Protocol):
    def generate(self, question: str, passages: list[Passage]) -> str: ...


class ExtractiveGenerator:
    """Picks the sentence most overlapping the question from each top passage and cites it.

    Faithful by construction (it only copies), so it is a floor/sanity check, not a quality bar.
    """

    def __init__(self, max_passages: int = 2):
        self.max_passages = max_passages

    def generate(self, question: str, passages: list[Passage]) -> str:
        q = set(tokenize(question))
        out = []
        for p in passages[: self.max_passages]:
            sents = split_sentences(p.text) or [p.text]
            best = max(sents, key=lambda s: len(q & set(tokenize(s))))
            clean = " ".join(best.split()).rstrip(".")
            out.append(f"{clean} [{p.n}].")
        return " ".join(out) if out else "I cannot answer from the provided sources."


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    passages: list[Passage]
    citations: CitationReport
    faithfulness: float


def answer_question(
    question: str,
    retriever,
    texts: dict[str, str],
    generator: Generator,
    k: int = 5,
) -> Answer:
    """retrieve -> number passages -> generate -> validate citations -> score faithfulness."""
    from papertrail.generation.prompt import make_passages

    passages = make_passages(retriever.search(question, k), texts)
    text = generator.generate(question, passages)
    return Answer(
        question, text, passages, validate_citations(text, passages), faithfulness(text, passages)
    )


__all__ = ["Generator", "ExtractiveGenerator", "Answer", "answer_question", "build_prompt"]
