"""Canonical record for one paper. Every later stage (BM25, embeddings, eval) reads this."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pyarrow as pa


@dataclass(frozen=True)
class Paper:
    arxiv_id: str  # versionless, e.g. "2401.01234" or "cs/0501001"
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]  # first one is the primary category
    published: str  # ISO date of v1, YYYY-MM-DD
    updated: str  # ISO date of latest version, YYYY-MM-DD
    url: str

    def to_dict(self) -> dict:
        return asdict(self)


ARROW_SCHEMA = pa.schema(
    [
        ("arxiv_id", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("authors", pa.list_(pa.string())),
        ("categories", pa.list_(pa.string())),
        ("published", pa.string()),
        ("updated", pa.string()),
        ("url", pa.string()),
    ]
)
