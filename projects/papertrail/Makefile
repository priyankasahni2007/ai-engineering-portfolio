.PHONY: install lint test ingest-sample
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
