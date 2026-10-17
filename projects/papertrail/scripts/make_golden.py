"""Build two synthetic golden sets from an ingested corpus (one relevant paper per query).

- titles:    query = paper title, relevant = that paper. Titles are NOT indexed (abstract-only
             corpus), so this is known-item search with real author-written phrasing.
- doc2query: query = question generated from the abstract by a T5 query-generation model
             (BeIR/query-gen-msmarco-t5-base-v1). Generated from the abstract, so it is biased
             towards lexical overlap with it; that bias is reported, not hidden.

Usage: python scripts/make_golden.py data/papers.parquet eval_data --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from papertrail.io import read_papers

QGEN_MODEL = "BeIR/query-gen-msmarco-t5-base-v1"


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    papers = read_papers(args.corpus)
    sample = random.Random(args.seed).sample(papers, args.n)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        [
            {"qid": f"t{i}", "query": p.title, "relevant": [p.arxiv_id]}
            for i, p in enumerate(sample)
        ],
        out / "golden_titles.jsonl",
    )

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(args.seed)  # makes sampling reproducible on the same torch build
    tok = AutoTokenizer.from_pretrained(QGEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QGEN_MODEL).eval()
    rows = []
    for i in range(0, len(sample), 16):
        batch = sample[i : i + 16]
        enc = tok(
            [p.abstract for p in batch],
            truncation=True,
            max_length=384,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            # Sampling, as the model card recommends: beam search gave vague "what is X" queries.
            gen = model.generate(**enc, max_length=48, do_sample=True, top_k=10, top_p=0.95)
        for p, g in zip(batch, gen, strict=True):
            rows.append(
                {
                    "qid": f"q{len(rows)}",
                    "query": tok.decode(g, skip_special_tokens=True),
                    "relevant": [p.arxiv_id],
                }
            )
    write_jsonl(rows, out / "golden_doc2query.jsonl")
    print(f"wrote {len(sample)} title queries and {len(rows)} doc2query queries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
