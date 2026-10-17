.PHONY: install lint test ingest-sample ingest golden eval
install:
	python -m pip install -e ".[dev]"
lint:
	python -m ruff check src tests
	python -m ruff format --check src tests
test:
	python -m pytest -q
# Offline smoke run on the bundled fixture (no network needed)
ingest-sample:
	python -m papertrail.cli --from-file tests/fixtures/sample_arxiv.xml --out data/sample.parquet
# Real data + evaluation (needs network to export.arxiv.org and huggingface.co)
ingest:
	python -m papertrail.cli --category cs.CL --n 2000 --out data/papers.parquet
golden:
	python scripts/make_golden.py data/papers.parquet eval_data --n 200 --seed 42
eval:
	python scripts/run_eval.py data/papers.parquet eval_data results
