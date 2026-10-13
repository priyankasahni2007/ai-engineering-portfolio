from pathlib import Path

from papertrail.cli import main
from papertrail.io import read_papers

FIXTURE = Path(__file__).parent / "fixtures" / "sample_arxiv.xml"


def test_cli_from_file(tmp_path):
    out = tmp_path / "x.parquet"
    assert main(["--from-file", str(FIXTURE), "--out", str(out)]) == 0
    assert len(read_papers(out)) == 2
