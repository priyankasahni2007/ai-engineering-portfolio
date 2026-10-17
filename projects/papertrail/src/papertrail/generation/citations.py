"""Citation parsing/validation and a lexical faithfulness metric.

Two different failure modes are checked separately:
  * citation validity  - does [n] point at a passage that was actually retrieved?
  * faithfulness       - is each sentence supported by the passages it cites?
A model can cite perfectly valid numbers and still say things the passages never said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from papertrail.generation.prompt import Passage
from papertrail.retrieval.bm25 import tokenize

_CITE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def extract_citations(text: str) -> list[int]:
    return [int(m) for m in _CITE.findall(text)]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def strip_citations(text: str) -> str:
    return _CITE.sub("", text).strip()


@dataclass(frozen=True)
class CitationReport:
    valid: list[int]
    invalid: list[int]  # cited numbers that match no retrieved passage (hallucinated citations)
    uncited_sentences: int  # sentences with no citation at all
    n_sentences: int

    @property
    def validity_rate(self) -> float:
        total = len(self.valid) + len(self.invalid)
        return 1.0 if total == 0 else len(self.valid) / total


def validate_citations(answer: str, passages: list[Passage]) -> CitationReport:
    allowed = {p.n for p in passages}
    cites = extract_citations(answer)
    sents = split_sentences(answer)
    return CitationReport(
        valid=[c for c in cites if c in allowed],
        invalid=[c for c in cites if c not in allowed],
        uncited_sentences=sum(1 for s in sents if not extract_citations(s)),
        n_sentences=len(sents),
    )


def sentence_support(sentence: str, passage_text: str) -> float:
    """Fraction of the sentence's content tokens that appear in the passage (lexical proxy)."""
    toks = set(tokenize(strip_citations(sentence)))
    if not toks:
        return 1.0  # nothing checkable
    return len(toks & set(tokenize(passage_text))) / len(toks)


def faithfulness(answer: str, passages: list[Passage], threshold: float = 0.6) -> float:
    """Share of sentences supported (>= threshold) by the passages THEY cite.

    A sentence with no valid citation counts as unsupported. Lexical overlap is a cheap, weak
    proxy: it rewards copying and misses paraphrase. NLI / LLM-judge replace it later.
    """
    sents = split_sentences(answer)
    if not sents:
        return 0.0
    by_n = {p.n: p.text for p in passages}
    ok = 0
    for s in sents:
        cited = [by_n[c] for c in extract_citations(s) if c in by_n]
        if cited and max(sentence_support(s, t) for t in cited) >= threshold:
            ok += 1
    return ok / len(sents)
