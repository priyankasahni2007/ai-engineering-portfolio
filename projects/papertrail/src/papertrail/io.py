"""Parquet read/write with an explicit schema (so empty/odd batches can't change column types)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import ARROW_SCHEMA, Paper


def write_papers(papers: list[Paper], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([p.to_dict() for p in papers], schema=ARROW_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    return path


def read_papers(path: str | Path) -> list[Paper]:
    return [Paper(**row) for row in pq.read_table(path).to_pylist()]
