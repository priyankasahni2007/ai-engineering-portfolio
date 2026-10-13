"""arXiv Atom API: parsing (pure, tested offline) and polite fetching (network)."""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from .schema import Paper

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
_VERSION = re.compile(r"v\d+$")
_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    """Atom text contains hard-wrapped lines; collapse all whitespace runs."""
    return _WS.sub(" ", text or "").strip()


def arxiv_id_from_url(url: str) -> str:
    """'http://arxiv.org/abs/2401.01234v2' -> '2401.01234' (old style 'cs/0501001v1' kept)."""
    tail = url.split("/abs/", 1)[-1]
    return _VERSION.sub("", tail.strip())


def parse_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed. Skips entries lacking an id, title, or abstract."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", NS):
        raw_id = _clean(entry.findtext("a:id", namespaces=NS))
        title = _clean(entry.findtext("a:title", namespaces=NS))
        abstract = _clean(entry.findtext("a:summary", namespaces=NS))
        if not (raw_id and title and abstract) or "/api/errors" in raw_id:
            continue  # arXiv returns an error "entry" for bad queries
        cats = [c.get("term", "") for c in entry.findall("a:category", NS) if c.get("term")]
        primary = entry.find("ax:primary_category", NS)
        if primary is not None and primary.get("term") in cats:
            cats.remove(primary.get("term"))
            cats.insert(0, primary.get("term"))
        papers.append(
            Paper(
                arxiv_id=arxiv_id_from_url(raw_id),
                title=title,
                abstract=abstract,
                authors=[
                    _clean(a.findtext("a:name", namespaces=NS))
                    for a in entry.findall("a:author", NS)
                ],
                categories=cats,
                published=_clean(entry.findtext("a:published", namespaces=NS))[:10],
                updated=_clean(entry.findtext("a:updated", namespaces=NS))[:10],
                url=f"https://arxiv.org/abs/{arxiv_id_from_url(raw_id)}",
            )
        )
    return papers


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Keep the first occurrence of each arxiv_id (paging can overlap while new papers arrive)."""
    seen: set[str] = set()
    out = []
    for p in papers:
        if p.arxiv_id not in seen:
            seen.add(p.arxiv_id)
            out.append(p)
    return out


def fetch(
    category: str,
    n: int,
    page_size: int = 100,
    delay_s: float = 3.0,
    contact: str = "",
    retries: int = 3,
) -> Iterator[Paper]:
    """Page through one category, newest first. arXiv asks for >=3s between calls."""
    ua = f"papertrail/0.1 (portfolio project; {contact})"
    for start in range(0, n, page_size):
        qs = urllib.parse.urlencode(
            {
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": min(page_size, n - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        for attempt in range(retries):
            try:
                req = urllib.request.Request(f"{API_URL}?{qs}", headers={"User-Agent": ua})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    page = parse_feed(resp.read().decode("utf-8"))
                break
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay_s * 2**attempt)  # exponential backoff
        if not page:
            return
        yield from page
        time.sleep(delay_s)
