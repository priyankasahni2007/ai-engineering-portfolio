"""Prompt construction: number the retrieved passages so the model can cite them as [n]."""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM = (
    "Answer the question using ONLY the numbered passages. After every claim, cite the passage "
    "number in square brackets, e.g. [1] or [2][3]. If the passages do not contain the answer, "
    "say you cannot answer from the provided sources."
)


@dataclass(frozen=True)
class Passage:
    n: int  # 1-based citation number shown to the model
    doc_id: str
    text: str


def make_passages(hits: list[tuple[str, float]], texts: dict[str, str]) -> list[Passage]:
    """Turn ranked (doc_id, score) hits into numbered passages, skipping ids without text."""
    kept = [d for d, _ in hits if d in texts]
    return [Passage(i, d, texts[d]) for i, d in enumerate(kept, start=1)]


def build_prompt(question: str, passages: list[Passage]) -> str:
    body = "\n".join(f"[{p.n}] {p.text}" for p in passages)
    return f"{SYSTEM}\n\nPassages:\n{body}\n\nQuestion: {question}\nAnswer:"
