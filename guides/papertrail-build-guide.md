# PaperTrail: build it yourself, step by step (Mac)

You'll build an evaluated research assistant over arXiv papers: search with three methods, measure which works best, then answer questions with citations and check that the answers stick to their sources.

**How to use this guide:** every step says **Where** (which app or file), **Do / Paste** (exactly what to run or type), and **Why** (what it does and why it's there). **Type the code yourself instead of copy-pasting the bigger files**, at least the core ones (BM25, metrics, RRF). That's how it ends up in your head and not just on GitHub. Commit at the end of every phase.

**Time:** about 6–8 hours, plus downloads. The slow parts are installing libraries and running the reranker evaluation (~10 min).

Every file here was run and tested before you got it: 18 tests pass, and the full pipeline ran end to end. The one exception is the Ollama call itself (Phase 10), which couldn't run in my test sandbox. I tested that step with a similar small model instead.

**Your numbers will differ from mine.** arXiv returns the newest papers, so you'll get a different 2,000. That's expected. Report *your* numbers, never mine.

---

## The project in one page

**Problem.** Hundreds of ML papers appear on arXiv every day. Keyword search misses papers that use different words, and chatbots invent facts without saying where they came from.

**What you build.**
```
arXiv API ─► 2,000 abstracts ─► BM25 index ───────┐
                              └► embedding index ─┴► RRF fusion ─► cross-encoder rerank ─► top 5 papers
                                                                                              │
question ─────────────────────────────────────────────────────────────────────────────────────┤
                                                                                              ▼
                                               local LLM writes the answer with [1][2] citations
                                                                                              ▼
                                          faithfulness check (NLI) + metrics + latency + tokens
```

**What makes it portfolio-worthy:** you don't just build search, you **measure** it. Recall@K, MRR and nDCG for 5 search setups, a confidence interval on the differences, latency per query, and faithfulness of the answers. That's exactly the "evals" skill job postings ask for.

| Tech | Why it's in the project |
|---|---|
| Python 3.12 | The language of AI engineering |
| requests + feedparser | Download from the arXiv API and read its XML |
| pandas + Parquet | Handle tables, and save list columns (authors) safely |
| BM25 (written by you) | Exact keyword matching. You'll be able to explain every line |
| sentence-transformers + FAISS | Meaning-based search (embeddings) and fast vector lookup |
| RRF | Combines two ranked lists without mixing their incompatible scores |
| Cross-encoder | Slower but more accurate re-ordering of the top results |
| Ollama + Llama 3.2 3B | Free LLM that runs on your Mac. No API key, no cost per query |
| NLI model | Checks whether a paper actually supports each answer sentence |
| FastAPI | Turns the code into a web service, with free interactive docs |
| pytest + GitHub Actions | Tests run automatically on every push |
| Docker | Anyone can run the project with one command |

**Final folder layout:**
```
papertrail/
├── src/papertrail/      fetch.py bm25.py metrics.py dense.py hybrid.py rerank.py
│                        pipeline.py generate.py faithfulness.py api.py __init__.py
├── scripts/             make_golden.py evaluate.py evaluate_answers.py
├── tests/               test_bm25.py test_metrics.py test_hybrid.py test_generate.py test_api.py
├── data/                (downloaded, never committed)
├── eval/                golden test sets (committed)
├── results/             your numbers (committed)
├── .github/workflows/ci.yml   Dockerfile   docker-compose.yml   .dockerignore
└── requirements.txt  requirements-ml.txt  pyproject.toml  .gitignore  README.md
```

---

# Phase 0: Setup (≈30 min)

### Step 0.1: Install the tools
**Where:** the Terminal app (press ⌘+Space, type "Terminal").
**Do:** if you don't have Homebrew yet (check with `brew --version`), install it with the command on https://brew.sh. Then run:
```bash
brew install python@3.12 git
python3.12 --version
```
**Why:** macOS ships an old Python (3.9). Some of our libraries (pandas 3) need 3.11 or newer, so we install 3.12. Homebrew is the standard Mac package manager. Also install **VS Code** (code.visualstudio.com) if you don't have it.

### Step 0.2: Create the repo
**Where:** github.com → **New repository**.
**Do:** name it `papertrail`, make it **Public**, tick **Add a README**, and leave .gitignore as None (we'll write our own). Then in Terminal:
```bash
cd ~                                   # or wherever you keep projects
git clone https://github.com/<your-username>/papertrail.git
cd papertrail
code .                                 # opens VS Code (if "code" isn't found: VS Code → ⌘⇧P → "Install 'code' command in PATH")
```
**Why:** it's a standalone repo, so it can be pinned on your GitHub profile and linked from your resume.

### Step 0.3: Make the folders
**Where:** Terminal, inside `papertrail/`.
```bash
mkdir -p src/papertrail scripts tests data results eval .github/workflows
touch src/papertrail/__init__.py
```
**Why:** this is the **src layout**. The code lives in a real package (`papertrail`) that you install, so imports work identically in scripts, tests, the API and Docker. The empty `__init__.py` marks the folder as a Python package.

### Step 0.4: `.gitignore`
**Where:** VS Code → new file `.gitignore` in the project root.
**Paste:**
```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
data/
```
**Why:** the virtual environment (hundreds of MB) and `data/` (downloaded papers and embeddings) must never go to GitHub. They can be regenerated, and they're big.

### Step 0.5: `requirements.txt` (light) and `requirements-ml.txt` (heavy)
**Where:** two new files in the root.
**Paste into `requirements.txt`:**
```
requests==2.34.2
feedparser==6.0.14
pandas==3.1.0
pyarrow==26.0.0
numpy==2.5.4
fastapi==0.162.0
uvicorn==0.54.0
pytest==9.1.1
httpx==0.28.1
```
**Paste into `requirements-ml.txt`:**
```
torch==2.14.1
sentence-transformers==6.2.0
faiss-cpu==1.15.1
transformers==5.21.0
sentencepiece==0.2.2
protobuf==7.36.2
```
**Why:** versions are pinned (`==`) so the project behaves the same on every machine. The split is deliberate. The heavy ML libraries (PyTorch is about 1 GB) are only needed for embeddings and models. Tests and CI only need the light file, so CI runs in seconds instead of minutes. Interviewers like this kind of detail.

### Step 0.6: `pyproject.toml`
**Where:** new file in the root.
**Paste:**
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "papertrail"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
**Why:** this makes `papertrail` an installable package (it's found under `src/`), and tells pytest where the tests are.

### Step 0.7: Virtual environment + install
**Where:** Terminal, in the project root.
```bash
python3.12 -m venv .venv
source .venv/bin/activate              # your prompt now starts with (.venv); run this in every new terminal
pip install --upgrade pip
pip install -r requirements.txt -r requirements-ml.txt
pip install -e .
```
**Why:** `.venv` is a private Python just for this project, so other projects can't break it. `pip install -e .` installs your package in "editable" mode: edits to `src/papertrail/*.py` take effect immediately, with no reinstall.

✅ **Checkpoint:** `python -c "import papertrail, torch, faiss; print('ok')"` prints `ok`.
```bash
git add . && git commit -m "Project setup: src layout, pinned requirements" && git push
```

---

# Phase 1: Download the papers (≈20 min)

### Step 1.1: Look at the raw data first
**Where:** your browser.
**Do:** open `https://export.arxiv.org/api/query?search_query=cat:cs.CL&max_results=2`
**Why:** always look at raw data before writing code that parses it. Find `<entry>`, `<id>`, `<title>`, `<summary>` (the abstract), `<author>` and `<published>`. `cs.CL` is the arXiv category for computational linguistics (NLP). Notice the id ends in `v1` or `v2`: that's the paper's version.

### Step 1.2: `src/papertrail/fetch.py`
**Where:** new file `src/papertrail/fetch.py`. Type it yourself.
```python
"""Download arXiv paper metadata and save it as a Parquet file.

Run:  python -m papertrail.fetch --category cs.CL --n 2000
"""

import argparse
import re
import time

import feedparser
import pandas as pd
import requests

API_URL = "https://export.arxiv.org/api/query"
VERSION_SUFFIX = re.compile(r"v\d+$")  # matches the "v2" at the END of "2410.01234v2"


def clean_id(url: str) -> str:
    """'http://arxiv.org/abs/2410.01234v2' -> '2410.01234'."""
    return VERSION_SUFFIX.sub("", url.split("/abs/")[-1])


def fetch_page(category: str, start: int, page_size: int) -> list[dict]:
    """Get one page of papers from arXiv, newest first."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": page_size,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    return [
        {
            "arxiv_id": clean_id(e.id),
            "title": " ".join(e.title.split()),
            "abstract": " ".join(e.summary.split()),
            "authors": [a.name for a in e.authors],
            "published": e.published[:10],
        }
        for e in feed.entries
    ]


def fetch(category: str, n: int, page_size: int = 100) -> list[dict]:
    papers = []
    for start in range(0, n, page_size):
        for attempt in range(3):  # arXiv sometimes times out: retry with growing waits
            try:
                page = fetch_page(category, start, min(page_size, n - start))
                break
            except requests.RequestException as err:
                print(f"  attempt {attempt + 1} failed: {err}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"arXiv failed 3 times at start={start}")
        print(f"fetched {start + len(page)} papers")
        papers += page
        time.sleep(3)  # arXiv's rule: at least 3 seconds between requests
    return papers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="cs.CL")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", default="data/papers.parquet")
    args = parser.parse_args()

    df = pd.DataFrame(fetch(args.category, args.n))
    df = df.drop_duplicates("arxiv_id")  # paging can overlap when new papers arrive
    df = df[df["abstract"].str.len() > 0]
    df.to_parquet(args.out, index=False)
    print(f"saved {len(df)} papers to {args.out}")


if __name__ == "__main__":
    main()
```
**Why, line by line:**
- **Paging (`start`, `page_size=100`):** the API won't return 2,000 results at once, so you ask for pages.
- **`time.sleep(3)`:** arXiv's terms ask for 3 seconds between requests. Ignore it and you get blocked.
- **Retry loop:** real APIs fail sometimes (timeouts, HTTP 429 "too many requests"). When I tested this, arXiv returned a 429 once and the retry saved the run. The `for ... else` runs the `else` only if the loop never hit `break`, meaning all 3 attempts failed.
- **`clean_id` with a regex:** removes `v2` *only at the end* (`$`). The same paper must have one ID across versions, or you'd count duplicates and break evaluation later. A naive `.split("v")` would break on old IDs like `solv-int/9901001`.
- **`drop_duplicates`:** if new papers are published while you're paging, results shift and the same paper can appear on two pages.
- **Parquet, not CSV:** `authors` is a *list*. CSV would flatten it into a string. Parquet keeps types and is smaller and faster.

### Step 1.3: Run it
**Where:** Terminal (venv active).
```bash
python -m papertrail.fetch --category cs.CL --n 2000
```
**Why `python -m`:** it runs the module from your installed package, so imports resolve the same way everywhere. It takes about 2 minutes (20 pages × 3 s, plus download time).

### Step 1.4: Check the data (write this yourself)
**Where:** Terminal, `python` (interactive).
**Do:** load `pd.read_parquet("data/papers.parquet")` and print:
1. The number of rows.
2. `df["arxiv_id"].duplicated().sum()`, which should be 0.
3. `df["abstract"].str.split().str.len().describe()`, the abstract length in words.
4. `df["published"].min(), df["published"].max()`.
5. `df.sample(1, random_state=0).iloc[0].to_dict()`.

**Why:** you'll quote these numbers in your README ("2,000 abstracts, avg N words, published X–Y"). Knowing your data is the first thing interviewers probe.

✅ **Checkpoint:** 2,000 papers, 0 duplicates.
```bash
git add . && git commit -m "arXiv fetcher with paging, retries, rate limit" && git push
```
Check on GitHub that `data/` was **not** uploaded.

---

# Phase 2: BM25 from scratch (≈45 min) ★ core interview topic

### Step 2.1: Understand it before you type it
BM25 gives each document a score for a query by summing, over the query's words:

`score += IDF(word) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × doc_len / avg_len))`

- **tf** (term frequency): how often the word appears in this document. More is better, but with **saturation**: the 10th occurrence adds much less than the 1st. `k1` controls how fast that happens.
- **IDF** (inverse document frequency): rare words matter more. "transformer" appears in half the papers and says little, while "Ewe" appears in one and says a lot.
- **Length normalisation (`b`):** long documents contain more words just by being long, so they're penalised a bit.

### Step 2.2: `src/papertrail/bm25.py`
**Where:** new file. Type it.
```python
"""BM25 keyword search, written from scratch."""

import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "we", "our",
    "with", "by", "this", "that", "as", "at", "be", "from", "it", "its", "which", "what",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop very common words."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b  # how much long documents are penalised

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25":
        self.doc_ids = list(doc_ids)
        self.doc_len = []
        self.postings = defaultdict(list)  # word -> [(doc_index, count_in_doc), ...]
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            for word, count in Counter(tokens).items():
                self.postings[word].append((i, count))
        self.n_docs = len(texts)
        self.avg_len = sum(self.doc_len) / max(self.n_docs, 1)
        return self

    def idf(self, word: str) -> float:
        """Rare words score high, common words score low."""
        n = len(self.postings.get(word, []))
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = defaultdict(float)
        for word in set(tokenize(query)):
            idf = self.idf(word)
            for i, tf in self.postings.get(word, []):
                length_norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        best = sorted(scores.items(), key=lambda item: -item[1])[:k]
        return [(self.doc_ids[i], score) for i, score in best]
```
**Why:**
- **Inverted index (`postings`):** for each word, the list of documents containing it. At query time you only touch documents that share a word with the query, not all 2,000. This is how every search engine works.
- **`set(tokenize(query))`:** a word repeated in the query counts once.
- **Stopwords:** "the", "of" and so on appear everywhere and carry no meaning.

### Step 2.3: `tests/test_bm25.py`
**Where:** new file.
```python
from papertrail.bm25 import BM25, tokenize

DOCS = {
    "a": "LoRA is a low-rank adaptation method for fine-tuning large language models",
    "b": "We study speech recognition for low-resource languages",
    "c": "Retrieval augmented generation grounds language models in documents",
}


def make_index():
    return BM25().fit(list(DOCS), list(DOCS.values()))


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The LoRA of Models") == ["lora", "models"]


def test_exact_keyword_ranks_first():
    assert make_index().search("LoRA fine-tuning", k=3)[0][0] == "a"


def test_rare_word_has_higher_idf_than_common_word():
    index = make_index()
    assert index.idf("speech") > index.idf("language")  # 'language' is in 2 docs


def test_unknown_words_return_nothing():
    assert make_index().search("quantum chromodynamics") == []
```
**Why:** each test pins one property you just learned: exact keywords win, rare words have higher IDF, and unknown words return nothing. If you ever "optimise" BM25 and break it, these fail.

### Step 2.4: Run the tests
```bash
pytest -q
```
✅ **Checkpoint:** `4 passed`. Then try it on real data in `python`:
```python
import pandas as pd; from papertrail.bm25 import BM25
df = pd.read_parquet("data/papers.parquet")
bm = BM25().fit(df.arxiv_id.tolist(), df.abstract.tolist())
for d, s in bm.search("speech recognition for low-resource languages", 3): print(round(s, 2), df.set_index("arxiv_id").loc[d, "title"])
```
```bash
git add . && git commit -m "BM25 from scratch with inverted index + tests" && git push
```

**Interview Qs:** *Why does BM25 saturate term frequency?* Without saturation, a document that repeats a word 50 times would beat one that uses it twice but is actually about the topic. *What does `b` do?* It controls how strongly long documents are penalised: `b=0` means no length penalty, `b=1` means full normalisation.

---

# Phase 3: Evaluation metrics (≈30 min) ★ core interview topic

### Step 3.1: `src/papertrail/metrics.py`
**Where:** new file. Type it.
```python
"""Retrieval metrics. `ranked` = doc ids in the order the search returned them."""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """What fraction of the relevant docs appear in the top k?"""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant doc (0 if none found)."""
    for position, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Relevant docs count more the higher they are ranked, scaled so perfect = 1."""
    dcg = sum(1 / math.log2(pos + 1) for pos, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def evaluate(runs: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    """Average each metric over all queries. runs/qrels are keyed by query id."""
    rows = [
        {
            "recall@5": recall_at_k(runs[q], qrels[q], 5),
            "recall@10": recall_at_k(runs[q], qrels[q], 10),
            "mrr": reciprocal_rank(runs[q], qrels[q]),
            "ndcg@10": ndcg_at_k(runs[q], qrels[q], 10),
        }
        for q in qrels
    ]
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}
```
**Why each metric:**
- **Recall@K:** did we find the right paper *somewhere* in the top K? If retrieval misses it, the LLM can never answer correctly.
- **MRR (mean reciprocal rank):** *how high* is the first right paper? Rank 1 scores 1.0, rank 2 scores 0.5, rank 4 scores 0.25. Users and LLMs pay most attention to the top results.
- **nDCG@10:** like MRR but works with several relevant documents, rewarding each one more the higher it's ranked. The `log2` is the "discount" for lower positions.

### Step 3.2: `tests/test_metrics.py`
```python
import pytest

from papertrail.metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall():
    assert recall_at_k(["x", "a", "y"], {"a", "b"}, k=2) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_perfect_and_worse():
    assert ndcg_at_k(["a", "x"], {"a"}, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}, k=10) < 1.0


def test_evaluate_averages_over_queries():
    runs = {"q1": ["a"], "q2": ["x", "b"]}
    qrels = {"q1": {"a"}, "q2": {"b"}}
    assert evaluate(runs, qrels)["mrr"] == pytest.approx((1 + 0.5) / 2)
```
✅ **Checkpoint:** `pytest -q` shows 8 passed.
```bash
git add . && git commit -m "Retrieval metrics: Recall@K, MRR, nDCG + tests" && git push
```

---

# Phase 4: A test set with known answers (≈20 min, mostly waiting)

### Step 4.1: Understand the trick
To score search, you need questions where you *know* the right paper. Labelling hundreds by hand takes days, so we generate two sets:
- **titles:** the paper's title is the question, and the answer is that paper. Titles are *not* in our index (we index abstracts only), so this isn't trivial.
- **doc2query:** a T5 model trained on real search queries reads an abstract and writes a question about it. Researchers use this approach too (InPars, GPL).

**Know the biases.** The questions come from the abstract's own words, which favours keyword search. Each question has only *one* paper marked correct, even if others also answer it, so all scores are **lower bounds**.

### Step 4.2: `scripts/make_golden.py`
```python
"""Make test questions with known answers ("golden sets") from the papers.

titles:    the paper's title is the question; the answer is that paper.
doc2query: a T5 model reads an abstract and writes a question a user might ask about it.

Run:  python scripts/make_golden.py
"""

import json
import random

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

QGEN_MODEL = "BeIR/query-gen-msmarco-t5-base-v1"
N_QUERIES = 200
SEED = 42


def save(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    papers = pd.read_parquet("data/papers.parquet").to_dict("records")
    sample = random.Random(SEED).sample(papers, N_QUERIES)

    titles = [
        {"qid": f"t{i}", "query": p["title"], "relevant": [p["arxiv_id"]]}
        for i, p in enumerate(sample)
    ]
    save(titles, "eval/golden_titles.jsonl")

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(QGEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QGEN_MODEL)
    questions = []
    for start in range(0, len(sample), 16):
        batch = sample[start : start + 16]
        inputs = tokenizer(
            [p["abstract"] for p in batch],
            truncation=True, max_length=384, padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=48, do_sample=True, top_k=10, top_p=0.95
            )
        for p, out in zip(batch, outputs):
            questions.append({
                "qid": f"q{len(questions)}",
                "query": tokenizer.decode(out, skip_special_tokens=True),
                "relevant": [p["arxiv_id"]],
            })
        print(f"generated {len(questions)} questions")
    save(questions, "eval/golden_doc2query.jsonl")


if __name__ == "__main__":
    main()
```
**Why:**
- **`SEED = 42`:** the same random sample and the same generated questions every run, so the results are reproducible.
- **Sampling (`do_sample, top_k, top_p`) instead of beam search:** when I tried beam search, every question came out as a bland "what is X". Sampling gives varied, realistic queries.

### Step 4.3: Run it and read the questions
```bash
python scripts/make_golden.py
head -20 eval/golden_doc2query.jsonl
```
It takes 2–5 min, and downloads the T5 model (~900 MB) the first time.
**Do:** read 20 questions. Note one that's clearly good and one that's vague ("what is benchmark score"). Vague questions match many papers but only one counts as correct. You'll mention this under "Failure cases" in your README.

The script saves them in `eval/`, not `data/`, on purpose. **The test sets get committed**, so anyone can reproduce your numbers on the same questions.
```bash
git add . && git commit -m "Synthetic golden sets: titles + doc2query (200 each, seed 42)" && git push
```

---

# Phase 5: Your first real result, the BM25 baseline (≈20 min)

### Step 5.1: `src/papertrail/pipeline.py` (version 1, BM25 only)
```python
"""Build every search configuration from the saved papers. Shared by evaluation and the API."""

import pandas as pd

from papertrail.bm25 import BM25


def load_papers(path: str = "data/papers.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def build_searchers(papers: pd.DataFrame, with_models: bool = True) -> dict:
    ids, texts = papers["arxiv_id"].tolist(), papers["abstract"].tolist()
    searchers = {"bm25": BM25().fit(ids, texts)}
    # Later phases add dense, hybrid and reranked searchers here.
    return searchers
```
**Why:** one place builds every search setup. The evaluation script and the API both use it, so they can never disagree about how search is configured.

### Step 5.2: `scripts/evaluate.py`
```python
"""Score every search configuration on a golden set: quality, speed, and a confidence interval.

Run:  python scripts/evaluate.py data/golden_doc2query.jsonl
"""

import json
import sys
import time

import numpy as np

from papertrail.metrics import evaluate, reciprocal_rank
from papertrail.pipeline import build_searchers, load_papers


def load_golden(path: str):
    rows = [json.loads(line) for line in open(path)]
    return {r["qid"]: r["query"] for r in rows}, {r["qid"]: set(r["relevant"]) for r in rows}


def bootstrap_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 0):
    """95% interval for mean(b - a), resampling the same queries for both systems."""
    diffs = np.array(b) - np.array(a)
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, len(diffs)).mean() for _ in range(n)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main() -> None:
    golden_path = sys.argv[1]
    queries, qrels = load_golden(golden_path)
    searchers = build_searchers(load_papers())

    results, per_query_rr = {}, {}
    for name, searcher in searchers.items():
        runs, latencies = {}, []
        for qid, query in queries.items():
            start = time.perf_counter()
            runs[qid] = [d for d, _ in searcher.search(query, k=10)]
            latencies.append((time.perf_counter() - start) * 1000)
        results[name] = evaluate(runs, qrels)
        results[name]["p50_ms"] = float(np.percentile(latencies, 50))
        results[name]["p95_ms"] = float(np.percentile(latencies, 95))
        per_query_rr[name] = [reciprocal_rank(runs[q], qrels[q]) for q in qrels]
        print(name, {m: round(v, 3) for m, v in results[name].items()}, flush=True)

    cols = ["recall@5", "recall@10", "mrr", "ndcg@10", "p50_ms", "p95_ms"]
    lines = [f"Golden set: {golden_path} ({len(queries)} queries)", "",
             "| config | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    lines += ["", "MRR difference vs bm25 (95% bootstrap CI):"]
    for name in searchers:
        if name != "bm25":
            low, high = bootstrap_ci(per_query_rr["bm25"], per_query_rr[name])
            lines.append(f"- {name}: [{low:+.3f}, {high:+.3f}]")
    report = "\n".join(lines)
    print("\n" + report)

    tag = golden_path.split("golden_")[-1].replace(".jsonl", "")
    with open(f"results/retrieval_{tag}.md", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
```
**Why:**
- **Latency `p50` / `p95`:** p50 is the typical query time. p95 is the slow tail, the time 1 in 20 users waits. Production teams care about p95.
- **Bootstrap confidence interval:** with 200 questions, an MRR difference of 0.01 could be luck. The bootstrap resamples the questions 2,000 times and sees how much the difference moves. If the 95% interval doesn't include 0, the difference is real. This puts you ahead of most student projects.

### Step 5.3: Run the baseline
```bash
python scripts/evaluate.py eval/golden_doc2query.jsonl
python scripts/evaluate.py eval/golden_titles.jsonl
```
**Why a baseline first:** every fancier method has to beat BM25 to justify its cost. Write your BM25 MRR and Recall@10 in a notes file.
```bash
git add . && git commit -m "Evaluation script + BM25 baseline results" && git push
```

---

# Phase 6: Embedding search (≈30 min) ★

### Step 6.1: Understand it
An **embedding model** turns text into a vector (384 numbers here), so that texts with similar *meaning* get nearby vectors. It's trained so that a question and the passage answering it end up close. **Cosine similarity** measures closeness. **FAISS** is a library for fast vector search. `IndexFlatIP` is exact (it compares against all vectors), which is fine for 2,000 papers. At millions of papers you'd switch to an approximate index (HNSW/IVF).

### Step 6.2: `src/papertrail/dense.py`
```python
"""Dense (embedding) search: similar meaning, even with different words."""

from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to see this prefix in front of short search queries.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseIndex:
    def __init__(self, model_name: str = MODEL, query_prefix: str = QUERY_PREFIX):
        from sentence_transformers import SentenceTransformer  # heavy import: only when used

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix

    def fit(self, doc_ids: list[str], texts: list[str], cache: str | None = None) -> "DenseIndex":
        import faiss

        self.doc_ids = list(doc_ids)
        if cache and Path(cache).exists():
            vectors = np.load(cache)  # embedding 2,000 abstracts is slow; do it once
        else:
            vectors = self.model.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
            )
            if cache:
                np.save(cache, vectors)
        # Inner product of unit-length vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.asarray(vectors, dtype="float32"))
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(self.doc_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]
```
**Why:**
- **`normalize_embeddings=True` + `IndexFlatIP`:** the inner product of unit-length vectors *is* cosine similarity.
- **Embedding cache (`data/embeddings.npy`):** encoding 2,000 abstracts takes a few minutes, and you'll run the evaluation many times. Delete the file if you change the model or the corpus, otherwise it serves stale vectors.
- **`QUERY_PREFIX`:** bge models were trained with this instruction in front of queries, and the model card recommends it.
- **Import inside `__init__`:** importing torch is slow. Code that only needs BM25 (tests, CI) never pays for it.

---

# Phase 7: Hybrid search with RRF (≈30 min) ★

### Step 7.1: `src/papertrail/hybrid.py`
```python
"""Hybrid search: merge keyword (BM25) and dense results with Reciprocal Rank Fusion."""


def rrf(rankings: list[list[str]], c: int = 60) -> list[tuple[str, float]]:
    """Each list gives a doc 1/(c + rank). Uses only ranks, so the two score scales never mix."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


class HybridSearch:
    def __init__(self, keyword, dense, candidates: int = 100):
        self.keyword, self.dense, self.candidates = keyword, dense, candidates

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        lists = [[d for d, _ in s.search(query, self.candidates)] for s in (self.keyword, self.dense)]
        return rrf(lists)[:k]
```
**Why RRF:** BM25 scores (e.g. 12.7) and cosine scores (e.g. 0.83) are on completely different scales, so adding them is meaningless. RRF ignores scores and uses only **ranks**: each list gives a document `1/(60 + rank)`, and a paper that's high in *both* lists wins. The `60` keeps rank 1 from completely dominating rank 2.

### Step 7.2: `tests/test_hybrid.py`
```python
from papertrail.hybrid import HybridSearch, rrf


def test_doc_ranked_well_by_both_lists_wins():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert fused[-1][0] in {"c", "d"}


def test_doc_in_only_one_list_still_appears():
    assert "d" in [d for d, _ in rrf([["a"], ["d"]])]


class FakeSearch:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=10):
        return [(d, 1.0) for d in self.ranking[:k]]


def test_hybrid_search_combines_both():
    hybrid = HybridSearch(FakeSearch(["a", "b"]), FakeSearch(["c", "a"]))
    assert hybrid.search("anything", k=1)[0][0] == "a"
```
**Why the `FakeSearch`:** it tests the fusion logic without loading any model. Tests should be fast and test one thing.

✅ `pytest -q` shows 11 passed.

---

# Phase 8: Reranker (≈20 min) ★

### Step 8.1: Understand bi-encoder vs cross-encoder
- **Bi-encoder** (Phase 6): the query and document are embedded **separately**. Documents are embedded once, ahead of time, so search is fast (~10 ms).
- **Cross-encoder:** reads the query **and** document **together**, so every query word can look at every document word. It's much more accurate, but it must run once per (query, document) pair *at query time*.
- **So, two stages:** a cheap search narrows 2,000 papers down to 20 or 50, then the expensive model orders only those. That's the standard production design.

### Step 8.2: `src/papertrail/rerank.py`
```python
"""Second stage: a cross-encoder re-scores the top candidates by reading query + doc together."""

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankedSearch:
    def __init__(self, first_stage, texts: dict[str, str], top_n: int = 50, model_name=MODEL):
        from sentence_transformers import CrossEncoder

        self.first_stage, self.texts, self.top_n = first_stage, texts, top_n
        self.model = CrossEncoder(model_name)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        candidates = [d for d, _ in self.first_stage.search(query, self.top_n)]
        scores = self.model.predict([(query, self.texts[d]) for d in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
        return [(d, float(s)) for d, s in ranked[:k]]
```

### Step 8.3: Update `src/papertrail/pipeline.py` to the final version
**Where:** replace the whole file.
```python
"""Build every search configuration from the saved papers. Shared by evaluation and the API."""

import pandas as pd

from papertrail.bm25 import BM25


def load_papers(path: str = "data/papers.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def build_searchers(papers: pd.DataFrame, with_models: bool = True) -> dict:
    ids, texts = papers["arxiv_id"].tolist(), papers["abstract"].tolist()
    searchers = {"bm25": BM25().fit(ids, texts)}
    if with_models:  # False = keyword search only: fast, no downloads (used in tests)
        from papertrail.dense import DenseIndex
        from papertrail.hybrid import HybridSearch
        from papertrail.rerank import RerankedSearch

        searchers["dense"] = DenseIndex().fit(ids, texts, cache="data/embeddings.npy")
        searchers["hybrid"] = HybridSearch(searchers["bm25"], searchers["dense"])
        by_id = dict(zip(ids, texts))
        searchers["hybrid_rerank20"] = RerankedSearch(searchers["hybrid"], by_id, top_n=20)
        searchers["hybrid_rerank50"] = RerankedSearch(searchers["hybrid"], by_id, top_n=50)
    return searchers
```
**Why two rerank sizes (20 and 50):** reranking 50 candidates costs about 2.5× as much as 20. Does it buy any quality? Measure it, don't guess. That gives you a **quality vs latency tradeoff** to talk about in interviews.

---

# Phase 9: The full comparison (≈30 min, mostly waiting) ★ your headline result

```bash
python scripts/evaluate.py eval/golden_doc2query.jsonl
python scripts/evaluate.py eval/golden_titles.jsonl
```
It takes about 10 min per set the first time. It downloads the models, and the reranker is slow on CPU. The tables land in `results/`.

**Then answer these in a notes file. They become your README's "What the numbers say":**
1. Which setup has the best MRR? What did it cost in p50 latency compared with BM25?
2. Did hybrid beat BM25 on **Recall@10**? On **MRR**? (On my test run, hybrid helped recall but *hurt* MRR, and its confidence interval touched zero. Why might fusion find more right papers but rank them lower?)
3. Is dense alone better or worse than BM25? Look at your questions: how many contain a unique name like a model or dataset? Who does that favour?
4. Did rerank@50 beat rerank@20 by more than the confidence interval? Is the extra latency worth it?
5. Do the titles results agree with the doc2query results?

For reference, my test run on doc2query (different papers, so yours will differ): BM25 MRR 0.815, dense 0.600, hybrid 0.773, rerank@20 **0.867** (CI vs BM25 [+0.011, +0.095]) at ~0.7 s/query.

```bash
git add . && git commit -m "Dense, hybrid (RRF) and cross-encoder reranking + full evaluation" && git push
```

---

# Phase 10: Answers with citations, using a local LLM (≈45 min)

### Step 10.1: Install Ollama and a model
**Where:** Terminal (a second tab).
```bash
brew install ollama
ollama serve                      # leave this tab running (or install the Ollama app from ollama.com instead)
```
In another tab:
```bash
ollama pull llama3.2:3b           # ~2 GB, runs fine on an M-series Mac
ollama run llama3.2:3b "Say hi in 5 words"
```
**Why a local model:** it's free, has no API key, works offline, and nothing leaves your laptop. It also makes "cost per query" an interesting discussion: it's measured in tokens and seconds instead of dollars.

### Step 10.2: `src/papertrail/generate.py`
```python
"""Answer a question from retrieved papers, with [n] citations, using a local LLM (Ollama)."""

import os
import re
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
CITATION = re.compile(r"\[(\d+)\]")
REFUSAL = "The sources do not answer this."


def build_prompt(question: str, passages: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{i}] {p['title']}\n{p['abstract']}" for i, p in enumerate(passages, start=1)
    )
    return (
        "You answer research questions using ONLY the numbered sources below.\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- Write 1 to 3 sentences, using only facts stated in the sources.\n"
        "- End EVERY sentence with its source number in square brackets, "
        "like this: LoRA trains small low-rank matrices [2].\n"
        f'- If the sources do not answer the question, reply exactly: "{REFUSAL}"\n\n'
        "Answer:"
    )


def split_sentences(answer: str) -> list[str]:
    # Models often write "... claim. [1]" -- move the citation inside: "... claim [1]."
    answer = re.sub(r"\.\s*((?:\[\d+\]\s*)+)", lambda m: " " + m.group(1).strip() + ". ", answer)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]


def check_citations(answer: str, n_passages: int) -> dict:
    """Which sentences cite something, and do the cited numbers exist?"""
    sentences = split_sentences(answer)
    cited = [[int(n) for n in CITATION.findall(s)] for s in sentences]
    invalid = sorted({n for nums in cited for n in nums if not 1 <= n <= n_passages})
    return {
        "n_sentences": len(sentences),
        "uncited_sentences": sum(1 for nums in cited if not nums),
        "invalid_citations": invalid,
    }


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> dict:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": data["response"],
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
    }


def answer_question(question, searcher, papers_by_id, generate_fn=ollama_generate, k=5) -> dict:
    t0 = time.perf_counter()
    hits = searcher.search(question, k)
    t1 = time.perf_counter()
    passages = [papers_by_id[d] for d, _ in hits]
    llm = generate_fn(build_prompt(question, passages))
    t2 = time.perf_counter()
    return {
        "answer": llm["text"],
        "sources": [{"n": i, "arxiv_id": p["arxiv_id"], "title": p["title"]}
                    for i, p in enumerate(passages, start=1)],
        "citations": check_citations(llm["text"], len(passages)),
        "tokens": {"prompt": llm["prompt_tokens"], "output": llm["output_tokens"]},
        "latency_ms": {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
    }
```
**Why:**
- **Numbered sources + `[n]` citations:** numbers are easy to check automatically. Free-text quotes aren't.
- **The rules come *after* the sources:** small models pay most attention to the end of the prompt. When I first put the rules at the top, a small model ignored them and cited almost nothing (88% of sentences uncited). Moving them to the end, with an example, dropped that to 25%. Prompt changes are experiments, so measure them.
- **The refusal sentence:** a model that says "I don't know" is better than one that makes something up. An exact string lets you count refusals.
- **`temperature: 0`:** same answer every time, which makes evaluation reproducible.
- **`split_sentences`:** models often write `claim. [1]`. We move the citation inside the sentence so each sentence keeps its own citation.
- **Token counts:** Ollama reports prompt and output tokens, so you can say "~1,500 prompt tokens per question". That's what a paid API would bill you for.

### Step 10.3: `tests/test_generate.py`
```python
from papertrail.generate import build_prompt, check_citations, split_sentences

PASSAGES = [{"title": "Paper A", "abstract": "A is about LoRA."},
            {"title": "Paper B", "abstract": "B is about RAG."}]


def test_prompt_numbers_every_source():
    prompt = build_prompt("What is LoRA?", PASSAGES)
    assert "[1] Paper A" in prompt and "[2] Paper B" in prompt
    assert "Question: What is LoRA?" in prompt and prompt.endswith("Answer:")


def test_citation_after_full_stop_is_moved_into_sentence():
    assert split_sentences("LoRA is cheap. [1] RAG retrieves. [2]") == [
        "LoRA is cheap [1].", "RAG retrieves [2]."
    ]


def test_check_citations_finds_uncited_and_invalid():
    result = check_citations("LoRA is cheap [1]. It is popular. RAG is old [7].", n_passages=2)
    assert result == {"n_sentences": 3, "uncited_sentences": 1, "invalid_citations": [7]}
```
✅ `pytest -q` shows 14 passed. These tests don't need Ollama.

---

# Phase 11: Faithfulness, checking the LLM (≈30 min) ★

### Step 11.1: Understand it
**Faithfulness** asks: is every sentence actually supported by the paper it cites? It's different from *correctness*. A sentence can be true but not in the source, and that's still a failure for a citing system.
An **NLI** (natural language inference) model reads (premise = abstract, hypothesis = sentence) and outputs probabilities for *entailment / neutral / contradiction*. A sentence counts as supported if entailment ≥ 0.5 for one of its cited papers, and an uncited sentence is never supported.

### Step 11.2: `src/papertrail/faithfulness.py`
```python
"""Is each answer sentence actually supported by the paper it cites? Checked with an NLI model.

NLI (natural language inference) models read (premise, hypothesis) and predict whether the
premise entails, contradicts, or is neutral towards the hypothesis.
"""

from papertrail.generate import CITATION, REFUSAL, split_sentences

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class FaithfulnessChecker:
    def __init__(self, model_name: str = NLI_MODEL, threshold: float = 0.5):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        labels = self.model.model.config.id2label  # {0: 'contradiction', 1: 'entailment', ...}
        self.entail_idx = next(i for i, name in labels.items() if name.lower() == "entailment")
        self.threshold = threshold

    def score(self, answer: str, passages: list[dict]) -> dict:
        if answer.strip() == REFUSAL:
            return {"faithfulness": None, "refused": True, "sentences": []}
        verdicts = []
        for sentence in split_sentences(answer):
            nums = [int(n) for n in CITATION.findall(sentence) if 1 <= int(n) <= len(passages)]
            claim = " ".join(CITATION.sub("", sentence).split()).replace(" .", ".")
            prob = 0.0
            if nums:  # an uncited sentence can't be supported
                pairs = [(passages[n - 1]["abstract"], claim) for n in nums]
                probs = self.model.predict(pairs, apply_softmax=True)
                prob = float(max(p[self.entail_idx] for p in probs))
            supported = prob >= self.threshold
            verdicts.append({"sentence": claim, "entailment": prob, "supported": supported})
        n_supported = sum(v["supported"] for v in verdicts)
        return {"faithfulness": n_supported / len(verdicts), "refused": False, "sentences": verdicts}
```
**Why read `id2label`:** different NLI models order their labels differently. Hard-coding "index 1 = entailment" is a classic silent bug.

### Step 11.3: Try it by hand first
In `python`:
```python
from papertrail.faithfulness import FaithfulnessChecker
c = FaithfulnessChecker()
P = [{"abstract": "We propose LoRA, which freezes the pretrained weights and injects trainable low-rank matrices, reducing trainable parameters by 10,000 times."}]
print(c.score("LoRA freezes the pretrained weights [1]. LoRA increases the number of trainable parameters [1].", P))
```
When I ran this, the first sentence scored entailment 0.96 and the second 0.0001 (a contradiction), so faithfulness was 0.5. Try your own true and false sentences. **Also find one case where the checker is wrong.** NLI models aren't perfect, and knowing their limits is part of the result.

### Step 11.4: `scripts/evaluate_answers.py`
```python
"""Ask the LLM real questions and measure the answers: citations, faithfulness, speed, tokens.

Needs Ollama running (`ollama serve`) with the model pulled (`ollama pull llama3.2:3b`).
Run:  python scripts/evaluate_answers.py --n 30
"""

import argparse
import json

import numpy as np

from papertrail.faithfulness import FaithfulnessChecker
from papertrail.generate import answer_question, ollama_generate
from papertrail.pipeline import build_searchers, load_papers


def run(generate_fn, n: int, config: str = "hybrid") -> str:
    papers = load_papers()
    papers_by_id = {p["arxiv_id"]: p for p in papers.to_dict("records")}
    searcher = build_searchers(papers)[config]
    checker = FaithfulnessChecker()
    golden = [json.loads(line) for line in open("eval/golden_doc2query.jsonl")][:n]

    rows = []
    for g in golden:
        out = answer_question(g["query"], searcher, papers_by_id, generate_fn)
        passages = [papers_by_id[s["arxiv_id"]] for s in out["sources"]]
        faith = checker.score(out["answer"], passages)
        rows.append({
            "question": g["query"], "answer": out["answer"],
            "gold_retrieved": g["relevant"][0] in [s["arxiv_id"] for s in out["sources"]],
            "refused": faith["refused"], "faithfulness": faith["faithfulness"],
            "uncited": out["citations"]["uncited_sentences"],
            "sentences": out["citations"]["n_sentences"],
            "invalid_citations": len(out["citations"]["invalid_citations"]),
            "gen_ms": out["latency_ms"]["generation"],
            "prompt_tokens": out["tokens"]["prompt"], "output_tokens": out["tokens"]["output"],
        })
        print(f"{len(rows)}/{len(golden)} faithfulness={faith['faithfulness']}", flush=True)

    with open("results/answers.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    answered = [r for r in rows if not r["refused"]]
    summary = {
        "questions": len(rows),
        "refusals": len(rows) - len(answered),
        "mean_faithfulness": float(np.mean([r["faithfulness"] for r in answered])),
        "share_sentences_uncited": sum(r["uncited"] for r in answered)
        / max(sum(r["sentences"] for r in answered), 1),
        "answers_with_invalid_citation": sum(r["invalid_citations"] > 0 for r in rows),
        "gold_paper_retrieved": sum(r["gold_retrieved"] for r in rows) / len(rows),
        "generation_p50_ms": float(np.percentile([r["gen_ms"] for r in rows], 50)),
        "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])),
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
    }
    report = json.dumps(summary, indent=2)
    with open("results/answers_summary.json", "w") as f:
        f.write(report + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--config", default="hybrid")
    args = parser.parse_args()
    print(run(ollama_generate, args.n, args.config))
```
**Run it** (with Ollama running):
```bash
python scripts/evaluate_answers.py --n 30
```
**Why these numbers:**
- **Mean faithfulness:** what share of answer sentences are actually supported by their source.
- **Share of uncited sentences:** how often the model breaks the citation rule.
- **Refusals:** how often the model says the sources don't answer.
- **Gold paper retrieved:** how often search even found the right paper. If it didn't, the LLM *should* refuse.
- **p50 generation time** and **tokens:** the cost of each answer.

Read 10 answers in `results/answers.jsonl` yourself. Find one good answer, one unfaithful one, and one wrongful refusal, and note *why* each happened. That becomes your "Failure cases" section.
```bash
git add . && git commit -m "LLM answers with citations + NLI faithfulness evaluation" && git push
```

---

# Phase 12: The API (≈30 min)

### Step 12.1: `src/papertrail/api.py`
```python
"""PaperTrail web API.

Run:  uvicorn papertrail.api:app_from_disk --factory --port 8000
Then open http://localhost:8000/docs
"""

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from papertrail.generate import answer_question, ollama_generate


class Hit(BaseModel):
    arxiv_id: str
    title: str
    score: float
    url: str


class SearchResponse(BaseModel):
    query: str
    config: str
    latency_ms: float
    results: list[Hit]


class AnswerRequest(BaseModel):
    question: str
    config: str = "hybrid"
    k: int = 5


def create_app(searchers: dict, papers_by_id: dict, generate_fn=ollama_generate) -> FastAPI:
    """Everything the app needs is passed in, so tests can give it a tiny fake setup."""
    app = FastAPI(title="PaperTrail")

    def get_searcher(config: str):
        if config not in searchers:
            raise HTTPException(400, f"unknown config {config!r}; choose from {list(searchers)}")
        return searchers[config]

    @app.get("/health")
    def health():
        return {"status": "ok", "papers": len(papers_by_id), "configs": list(searchers)}

    @app.get("/search", response_model=SearchResponse)
    def search(q: str, config: str = "bm25", k: int = 5):
        searcher = get_searcher(config)
        start = time.perf_counter()
        hits = searcher.search(q, k)
        latency = (time.perf_counter() - start) * 1000
        results = [
            Hit(arxiv_id=d, title=papers_by_id[d]["title"], score=s,
                url=f"https://arxiv.org/abs/{d}")
            for d, s in hits
        ]
        return SearchResponse(query=q, config=config, latency_ms=latency, results=results)

    @app.post("/answer")
    def answer(req: AnswerRequest):
        searcher = get_searcher(req.config)
        try:
            return answer_question(req.question, searcher, papers_by_id, generate_fn, req.k)
        except OSError as err:  # LLM server not running / unreachable
            raise HTTPException(503, f"LLM unavailable: {err}") from err

    return app


def app_from_disk() -> FastAPI:
    from papertrail.pipeline import build_searchers, load_papers

    papers = load_papers(os.environ.get("PAPERTRAIL_DATA", "data/papers.parquet"))
    with_models = os.environ.get("PAPERTRAIL_MODELS", "1") == "1"
    papers_by_id = {p["arxiv_id"]: p for p in papers.to_dict("records")}
    return create_app(build_searchers(papers, with_models=with_models), papers_by_id)
```
**Why:**
- **`create_app(...)` receives its searchers instead of loading them itself:** this is *dependency injection*. Tests can pass a tiny 2-paper setup and a fake LLM, and run in milliseconds with no downloads.
- **Pydantic models (`Hit`, `SearchResponse`):** they validate inputs and outputs and generate the `/docs` page automatically.
- **Indexes load once at startup** (`app_from_disk`), not per request. Loading models per request would take seconds every time.
- **`/health`:** deployment systems ping it to check the service is alive.
- **503 if the LLM is down:** a clear error instead of a crash. I tested this with Ollama switched off.

### Step 12.2: `tests/test_api.py`
```python
import pandas as pd
from fastapi.testclient import TestClient

from papertrail.api import create_app
from papertrail.pipeline import build_searchers

PAPERS = pd.DataFrame([
    {"arxiv_id": "1", "title": "LoRA paper", "abstract": "low-rank adaptation for fine-tuning"},
    {"arxiv_id": "2", "title": "Speech paper", "abstract": "speech recognition in Hindi"},
])
PAPERS_BY_ID = {p["arxiv_id"]: p for p in PAPERS.to_dict("records")}


def fake_llm(prompt):
    return {"text": "It is low-rank adaptation [1].", "prompt_tokens": 10, "output_tokens": 5}


client = TestClient(create_app(build_searchers(PAPERS, with_models=False), PAPERS_BY_ID, fake_llm))


def test_health():
    assert client.get("/health").json()["papers"] == 2


def test_search_returns_best_paper_first():
    body = client.get("/search", params={"q": "speech recognition"}).json()
    assert body["results"][0]["arxiv_id"] == "2"


def test_unknown_config_is_a_clear_400():
    assert client.get("/search", params={"q": "x", "config": "nope"}).status_code == 400


def test_answer_has_sources_and_citation_check():
    body = client.post("/answer", json={"question": "what is low-rank adaptation", "config": "bm25", "k": 2}).json()
    assert body["sources"][0]["arxiv_id"] == "1"
    assert body["citations"]["invalid_citations"] == []
```
**Note the test query "low-rank adaptation", not "lora":** we index abstracts only, and the tiny test abstract doesn't contain the word "lora". I hit exactly this while testing, and it's a nice concrete example of why indexing choices matter.

✅ `pytest -q` shows **18 passed**.

### Step 12.3: Run the server and try it
```bash
uvicorn papertrail.api:app_from_disk --factory --port 8000
```
Open **http://localhost:8000/docs** and try `/search` with `config=hybrid_rerank20`, then `/answer`. Watch the `latency_ms` and `tokens` fields.
```bash
git add . && git commit -m "FastAPI service: /health, /search, /answer + tests" && git push
```

---

# Phase 13: CI with GitHub Actions (≈15 min)

### Step 13.1: `.github/workflows/ci.yml`
```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: pytest -q
```
**Why:** every push now runs your tests on a clean Linux machine. It installs only `requirements.txt`, because the tests never load a model. I checked this: all 18 tests pass in a fresh environment without torch.
**Do:** push, then open your repo → **Actions** tab and watch it go green ✅. Add the badge to your README (Actions → the workflow → ⋯ → "Create status badge").

---

# Phase 14: Docker (≈30 min)

### Step 14.1: Install Docker Desktop for Mac (docker.com), and start it.

### Step 14.2: `Dockerfile`, `.dockerignore`, `docker-compose.yml`
```dockerfile
FROM python:3.12-slim
WORKDIR /app

# 1. Dependencies first: this layer is cached until a requirements file changes.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# 2. Then the code (changes often, so it goes last).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "papertrail.api:app_from_disk", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```
```
.venv
data
results
.git
__pycache__
```
```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data                         # papers + cached embeddings
      - hf-cache:/root/.cache/huggingface        # downloaded models survive restarts
    extra_hosts: ["host.docker.internal:host-gateway"]   # needed on Linux; harmless on Mac
    environment:
      OLLAMA_URL: http://host.docker.internal:11434
volumes:
  hf-cache:
```
**Why:**
- **Layer order:** requirements first, code last. Docker caches each step, so editing your code doesn't re-download 1 GB of PyTorch.
- **CPU-only torch:** the default Linux PyTorch download includes GPU libraries you don't need, and it's much larger.
- **`.dockerignore`:** keeps your venv and data out of the image.
- **Volumes:** `data/` is mounted from your Mac instead of baked into the image, and downloaded models are kept between restarts.
- **`host.docker.internal`:** how the container reaches Ollama running on your Mac.

```bash
docker compose up --build
```
Then open http://localhost:8000/docs again, now served from the container. The first build takes a while.

**Honest note:** I couldn't build the image in my test sandbox (Docker Hub rate-limited it), so this is the one step you'll be the first to run. If it fails, paste me the error.
```bash
git add . && git commit -m "Dockerfile + docker-compose" && git push
```

---

# Phase 15: README + resume (≈45 min)

### Step 15.1: Write `README.md` yourself, using *your* numbers
Use this structure:
1. **One-line pitch plus your headline number**, e.g. "Cross-encoder reranking raised MRR from X to Y over BM25 on 200 questions."
2. **Problem / Why it matters**
3. **Architecture**: paste the diagram from the top of this guide.
4. **Results tables**: copy them from `results/retrieval_*.md` and `results/answers_summary.json`.
5. **What the numbers say**: your answers to the Phase 9 questions.
6. **Failure cases**: your 3 examples from Phase 11.
7. **Limitations**: synthetic questions, one label per question (lower bounds), 2,000 papers, CPU latency.
8. **How to run**: the commands from Phases 0–14.
9. **Future work**: scale to 10k+ papers, hand-label 50 questions, try a bigger LLM, use an approximate FAISS index.

### Step 15.2: Resume bullets (fill in your numbers)
- Built a two-stage hybrid retrieval system over **2,000** arXiv papers, combining BM25 implemented from scratch with FAISS dense embeddings via Reciprocal Rank Fusion, plus cross-encoder reranking. Raised MRR from **__** to **__** and Recall@10 from **__** to **__** over a BM25 baseline (200 questions, 95% bootstrap CI).
- Designed an evaluation harness (Recall@K, MRR, nDCG, p50/p95 latency) comparing 5 retrieval configurations. Quantified the reranker's latency cost (**__** ms vs **__** ms per query on CPU) and measured LLM answer faithfulness with an NLI model (**__**% of sentences supported).
- Served the system as a FastAPI service with a local LLM (Llama 3.2 3B via Ollama) generating cited answers. Covered it with 18 pytest tests in GitHub Actions CI and containerized it with Docker.

⚠️ Only put a number on your resume if you produced it yourself and can explain it. Your resume currently says "10,000+ papers". Change it to 2,000, or re-run Phase 1 with `--n 10000` first (~10 min download, and the evaluation gets slower).

---

## Troubleshooting (Mac)
- **`OMP: Error #15 ... libomp already initialized`** (PyTorch and FAISS both bring OpenMP): run `export KMP_DUPLICATE_LIB_OK=TRUE` in that terminal, or add it to `~/.zshrc`.
- **`tokenizers` fork warning:** harmless. Silence it with `export TOKENIZERS_PARALLELISM=false`.
- **`ValueError: tiktoken is required`** when loading T5: `pip install protobuf sentencepiece`. Both are in `requirements-ml.txt`, so check your venv is active.
- **arXiv HTTP 429 / timeouts:** the retry handles a few. If it keeps failing, wait 5 minutes.
- **Changed the corpus or the embedding model?** Delete `data/embeddings.npy`.
- **`command not found: python`** in a new terminal: run `source .venv/bin/activate` again.

## The interview questions this project prepares you for
1. Why hybrid search? When does BM25 beat embeddings? *(exact names and rare terms)*
2. Why RRF instead of adding the scores? *(different scales; RRF uses only ranks)*
3. Bi-encoder vs cross-encoder, and why two stages? *(speed vs accuracy; fixed rerank cost)*
4. How did you build your evaluation set, and what are its biases? *(synthetic, one label per query, lexical bias → lower bounds)*
5. How do you know a difference isn't noise? *(paired bootstrap CI)*
6. How do you measure hallucination? *(citation checks + NLI faithfulness; the NLI model's own limits)*
7. Your p95 latency is 2 s. How would you cut it? *(rerank fewer candidates, a smaller or quantized reranker, caching, GPU)*
8. How would you scale to 1M papers? *(approximate FAISS index, batch embedding, a real search engine for BM25)*
