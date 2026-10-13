# ADR 0001: Project layout, tooling, and ingestion source

Date: 2026-10-13 · Status: accepted

## Decisions
1. **`src/` layout + `pyproject.toml`.** Tests import the *installed* package, so packaging mistakes surface locally and in CI instead of after release.
2. **ruff (lint + format) and pytest.** One fast tool for lint/format/import order; CI runs exactly `make lint test`, the same commands a contributor runs.
3. **Ingestion via the arXiv Atom API, parsed with stdlib `xml.etree`, stored as Parquet (pyarrow).** The parser is a pure function over XML text, so it is tested offline on a fixture; network code is separate and thin. Parquet gives typed columns, compression (zstd), and fast reload for BM25/embedding stages. Explicit Arrow schema prevents type drift on empty batches.
4. **Polite fetching:** >=3 s between requests, User-Agent with contact, exponential-backoff retries (arXiv usage policy).
5. **Dependencies kept minimal at M1** (only pyarrow); heavy ML deps are added in the milestone that needs them.

## Alternatives considered
- Kaggle/HF arXiv metadata dump: bigger and faster to get 10k+ papers, but needs those hosts/accounts. Kept as the fallback if API paging is too slow for 10k+ (see NEXT_STEPS).
- pandas for storage: heavier dependency, no need for it yet.

## Known limitation (2026-10-13)
The remote build sandbox cannot reach arxiv.org / huggingface.co (egress policy), so the real fetch has not been run here; only the parser/IO are tested against a synthetic fixture.
