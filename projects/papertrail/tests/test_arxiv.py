from pathlib import Path

import pytest

from papertrail import arxiv
from papertrail.io import read_papers, write_papers

FIXTURE = Path(__file__).parent / "fixtures" / "sample_arxiv.xml"


@pytest.fixture
def papers():
    return arxiv.parse_feed(FIXTURE.read_text(encoding="utf-8"))


def test_skips_error_and_empty_abstract_entries(papers):
    assert [p.arxiv_id for p in papers] == ["0000.00001", "cs/0000001", "0000.00001"]


def test_whitespace_collapsed(papers):
    assert papers[0].title == "A Fictional Study of Sparse Retrieval"
    assert "\n" not in papers[0].abstract
    assert papers[0].abstract.startswith("We study term-weighting for retrieval")


def test_primary_category_first(papers):
    assert papers[0].categories == ["cs.CL", "cs.LG"]


def test_dates_and_authors(papers):
    assert papers[0].published == "2024-01-15"
    assert papers[0].updated == "2024-02-01"
    assert papers[0].authors == ["Ada Example", "Bo Sample"]


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://arxiv.org/abs/2401.01234v12", "2401.01234"),
        ("http://arxiv.org/abs/cs/0501001v1", "cs/0501001"),
        ("http://arxiv.org/abs/2401.01234", "2401.01234"),
    ],
)
def test_id_normalisation(url, expected):
    assert arxiv.arxiv_id_from_url(url) == expected


def test_dedupe_keeps_first(papers):
    out = arxiv.dedupe(papers)
    assert [p.arxiv_id for p in out] == ["0000.00001", "cs/0000001"]
    assert out[0].title.endswith("Sparse Retrieval")  # first (v2) wins, not the later duplicate


def test_parquet_roundtrip(tmp_path, papers):
    out = arxiv.dedupe(papers)
    path = write_papers(out, tmp_path / "p.parquet")
    assert read_papers(path) == out


def test_empty_write_keeps_schema(tmp_path):
    path = write_papers([], tmp_path / "e.parquet")
    assert read_papers(path) == []
