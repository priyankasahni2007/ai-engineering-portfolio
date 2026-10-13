"""Ingest arXiv abstracts into Parquet: `python -m papertrail.cli --category cs.CL --n 1000`."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import arxiv
from .io import write_papers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default="cs.CL")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/papers.parquet")
    ap.add_argument("--from-file", help="parse a saved Atom XML file instead of calling arXiv")
    args = ap.parse_args(argv)

    if args.from_file:
        papers = arxiv.parse_feed(Path(args.from_file).read_text(encoding="utf-8"))
    else:
        papers = list(
            arxiv.fetch(args.category, args.n, contact=os.environ.get("PAPERTRAIL_CONTACT", ""))
        )
    papers = arxiv.dedupe(papers)
    write_papers(papers, args.out)
    print(f"wrote {len(papers)} papers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
