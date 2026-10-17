# ADR 0005: Grounded generation, citation validation, lexical faithfulness

Status: accepted (2026-10-17)

## Decisions
1. **Numbered passages + `[n]` citations.** Retrieved passages are numbered from 1 in the prompt; the model must cite by number. Numbers are cheap to parse and verify, unlike free-text quotes.
2. **Validity and faithfulness are separate metrics.** `validate_citations` catches citations pointing at passages that don't exist and uncited sentences. `faithfulness` checks that each sentence is supported by the passages *it cites*. An answer can pass the first and fail the second.
3. **Lexical support (token overlap >= 0.6) as the v0 faithfulness metric.** Free, deterministic, runs in CI. Known weakness: rewards copying, misses paraphrase and negation. NLI model / LLM-judge will be added and calibrated against a small hand-labelled set before any number is reported.
4. **Citation convention: cite before the full stop** (`... [1].`). The sentence splitter treats `. [1]` as a new sentence; a test pins this. Prompt tells the model the convention.
5. **`Generator` is a Protocol.** `ExtractiveGenerator` (copies the best sentence per passage) is an offline floor used in tests; Anthropic / HF wrappers plug in later.

## Consequences
No generation-quality or faithfulness result exists yet; the only generator is extractive and faithful by construction, so its score must not be reported as a result.
