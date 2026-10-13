# PaperTrail

Evaluated hybrid retrieval-augmented research assistant over arXiv ML papers.
**Status: M1 (skeleton + ingestion). README sections (architecture, results, failure cases, etc.) are filled in as milestones land — no results exist yet.**

## Quickstart
```bash
cd projects/papertrail
make install
make lint test
make ingest-sample                                   # offline, uses a synthetic fixture
python -m papertrail.cli --category cs.CL --n 2000 --out data/papers.parquet   # real data, needs network
```
Data schema: see `src/papertrail/schema.py`. Decisions: `docs/adr/`.
