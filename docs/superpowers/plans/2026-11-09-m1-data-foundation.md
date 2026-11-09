# Milestone 1: Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 72,613 Item 1A sections into a validated sentence table, with each filing's real filing date and industry, ready for the copying measurements in milestone 2.

**Architecture:** A new `pipeline/` package with one module per concern (config, polite EDGAR downloads, filing dates, industry, sentences) and one entry point, `python -m pipeline.run`, that builds two cached outputs: `data/interim/filings.parquet` (one row per filing) and `data/interim/sentences/year=YYYY.parquet` (one row per sentence). Pure functions are tested on small fixtures; the two stages are run once on the real data and their numbers recorded in `docs/data-foundation.md`.

**Tech Stack:** Python 3.12, uv (`pyproject.toml` + `uv.lock`), pandas, pyarrow, requests, python-dotenv, pytest, ruff. One sentence splitter, chosen in Task 6 (candidates: a regex baseline and `pysbd`).

**Spec:** `docs/superpowers/specs/2026-11-09-copying-rework-design.md`

## Global Constraints

- Python 3.12 via uv; dependencies pinned in `uv.lock`.
- SEC fair access: every request sends `User-Agent` from `SEC_USER_AGENT` in the git-ignored `.env`; never write the contact into code; at most 10 requests a second.
- Raw downloads under `data/raw/edgar/`, intermediate files under `data/interim/`; both git-ignored.
- Sentence normalization is exactly: Unicode NFKC, lowercase, every digit to `0`, every character outside `[a-z0-9 ]` to a space, whitespace collapsed, stripped.
- Industry level is the SIC division.
- No illustrative or hand-typed numbers anywhere: every figure in `docs/data-foundation.md` comes from a run.
- Leave the old `src/` modeling scripts and `requirements.txt` alone; milestone 5 removes them.

## Review Focus

1. `SEC_USER_AGENT` missing or blank: the pipeline stops before any network call with a message naming the `.env` line to add, not an HTTP 403 from SEC. (Task 1)
2. CIKs in two formats: the parquet stores CIK as a string, form indexes as an unpadded integer; both are normalized to `int`, so `"0000320193"` and `320193` join. (Task 3)
3. A company-year with several 10-K-family filings (an original and an amendment, or a 10-KT): the earliest original is used and the row is marked `multiple`; a company-year with only an amendment is dated from it and marked `amendment_only`. (Task 3)
4. A near-empty section ("Not applicable." from a smaller reporting company, or an empty string) yields zero or one sentences without an error, and is counted. (Task 6)
5. An interrupted download leaves no half-written file in the cache: files are written to a temporary name and renamed when complete, so a rerun refetches instead of parsing a truncated file. (Task 2)

---

### Task 1: uv project, package skeleton, config

**Files:**
- Create: `pyproject.toml`, `pipeline/__init__.py`, `pipeline/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Modify: `.gitignore` (add `data/interim/`, `data/raw/edgar/`)

**Interfaces:**
- Produces: `Config` (frozen dataclass: `root: Path`, `raw_dir: Path`, `interim_dir: Path`, `sec_user_agent: str`) and `load_config(env_file: Path | None = None, root: Path | None = None) -> Config`. Defaults: `root` is the repo root, `raw_dir = root / "data/raw/edgar"`, `interim_dir = root / "data/interim"`, `env_file = root / ".env"`.

- [ ] **Step 1: Create the uv project.** `pyproject.toml` with `requires-python = ">=3.12,<3.13"`, runtime deps `pandas`, `pyarrow`, `requests`, `python-dotenv`, dev group `pytest`, `ruff`, and `[tool.pytest.ini_options] testpaths = ["tests"]`. Run `uv sync` with Python 3.12 (`uv python install 3.12` if needed). Expected: `.venv` created, `uv.lock` written.
- [ ] **Step 2: Write the failing tests** in `tests/test_config.py`:

```python
def test_reads_user_agent_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    (tmp_path / ".env").write_text("SEC_USER_AGENT=Jane Doe jane@example.com\n")
    cfg = load_config(root=tmp_path)
    assert cfg.sec_user_agent == "Jane Doe jane@example.com"
    assert cfg.raw_dir == tmp_path / "data/raw/edgar"
    assert cfg.interim_dir == tmp_path / "data/interim"

def test_missing_user_agent_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit, match="SEC_USER_AGENT=Your Name you@example.com"):
        load_config(root=tmp_path)

def test_blank_user_agent_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    with pytest.raises(SystemExit, match="SEC_USER_AGENT"):
        load_config(root=tmp_path)
```

- [ ] **Step 3: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL, `load_config` not defined.
- [ ] **Step 4: Implement `load_config`** in `pipeline/config.py`: `dotenv.load_dotenv(env_file, override=False)`, then read the variable; exit with the message `Add this line to .env: SEC_USER_AGENT=Your Name you@example.com` when it's missing or blank.
- [ ] **Step 5: Run** `uv run pytest -v && uv run ruff check pipeline tests`. Expected: 3 passed, ruff clean.
- [ ] **Step 6: Commit** `git add pyproject.toml uv.lock .gitignore pipeline tests && git commit -m "Add the uv project and pipeline config"`.

### Task 2: Polite, cached EDGAR downloads

**Files:**
- Create: `pipeline/edgar.py`, `tests/test_edgar.py`

**Interfaces:**
- Consumes: `Config.sec_user_agent`, `Config.raw_dir`.
- Produces: `fetch(url: str, dest: Path, user_agent: str, *, session: requests.Session | None = None, min_interval: float = 0.15, max_retries: int = 5) -> Path`. Returns `dest`; if `dest` exists it returns at once with no request.

- [ ] **Step 1: Write the failing tests** with a fake session object whose `get` records calls and returns scripted responses (status, content):
  - `test_sends_user_agent_and_writes_file`: one 200 response; asserts the `User-Agent` header equals the argument, the file content matches, and `dest` is returned.
  - `test_cached_file_makes_no_request`: `dest` pre-exists; asserts `get` was never called.
  - `test_retries_on_429_then_succeeds`: responses 429, 503, 200; asserts 3 calls and the file is written (patch `time.sleep` to a no-op).
  - `test_gives_up_after_max_retries`: five 503s with `max_retries=5`; asserts `requests.HTTPError` and that `dest` does not exist.
  - `test_failed_write_leaves_no_partial_file`: the response's content raises mid-write; asserts `dest` does not exist afterwards.
- [ ] **Step 2: Run** `uv run pytest tests/test_edgar.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `fetch`**: stream to `dest.with_suffix(dest.suffix + ".part")`, rename on success, delete the `.part` file on any exception; exponential backoff (1 s doubling, capped at 60 s) on 429 and 5xx; sleep so successive requests are at least `min_interval` apart; create parent directories.
- [ ] **Step 4: Run** `uv run pytest tests/test_edgar.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit** `"Add cached EDGAR downloads that follow SEC fair-access rules"`.

### Task 3: Filing dates from the EDGAR form indexes

**Files:**
- Create: `pipeline/filings.py`, `tests/test_filings.py`, `tests/fixtures/form_2015_q1.idx` (about 12 lines copied in the real fixed-width layout: header, dashed rule, rows for a 10-K, a 10-K/A, a 10-KT, an 8-K and a 10-K405)

**Interfaces:**
- Produces:
  - `TEN_K_FORMS: frozenset[str] = frozenset({"10-K", "10-K405", "10-KT", "10-KSB", "10-KSB40", "10-KT405"})`; amendments are those forms with `/A`.
  - `parse_form_index(text: str) -> pd.DataFrame` with columns `form_type: str`, `cik: int`, `date_filed: datetime64[ns]`, `filename: str`, keeping only `TEN_K_FORMS` and their `/A` variants.
  - `match_filing_dates(filings: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame`: `filings` has `cik` (str or int) and `filing_year: int`; returns those rows plus `date_filed`, `form_type`, `n_candidates: int`, `match_status` ∈ {`"unique"`, `"multiple"`, `"amendment_only"`, `"none"`}.
  - `index_urls(years: range) -> list[str]`: `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.idx` for q in 1–4.

- [ ] **Step 1: Write the failing tests:**

```python
def test_parses_fixed_width_rows_and_keeps_10k_family(fixture_text):
    df = parse_form_index(fixture_text)
    assert set(df.form_type) == {"10-K", "10-K/A", "10-KT", "10-K405"}
    assert df.cik.dtype == "int64"

def test_string_and_int_ciks_join():
    filings = pd.DataFrame({"cik": ["0000320193"], "filing_year": [2015]})
    index = pd.DataFrame({"form_type": ["10-K"], "cik": [320193],
                          "date_filed": pd.to_datetime(["2015-10-28"]), "filename": ["x"]})
    out = match_filing_dates(filings, index)
    assert out.loc[0, "match_status"] == "unique"
    assert out.loc[0, "date_filed"] == pd.Timestamp("2015-10-28")

def test_earliest_original_wins_over_amendment():
    # original 2015-03-02, amendment 2015-04-15, second original (10-KT) 2015-09-30
    ...
    assert row.date_filed == pd.Timestamp("2015-03-02")
    assert row.match_status == "multiple" and row.n_candidates == 2

def test_amendment_only_is_dated_and_marked(): ...  # match_status == "amendment_only"
def test_no_match_is_kept_with_status_none(): ...   # date_filed is NaT, n_candidates == 0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_filings.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Parse after the dashed rule; split each row on runs of two or more spaces from the right (filename, date, CIK are the last three fields; form type is the first token; the company name is what's left). `n_candidates` counts originals only.
- [ ] **Step 4: Run** the tests. Expected: 5 passed.
- [ ] **Step 5: Commit** `"Match each filing to its EDGAR filing date"`.

### Task 4: Industry from SEC submissions data

**Files:**
- Create: `pipeline/industry.py`, `tests/test_industry.py`

**Interfaces:**
- Produces:
  - `SIC_DIVISIONS: dict[str, tuple[int, int, str]]`, exactly:
    `A (100, 999, "Agriculture, Forestry, Fishing")`, `B (1000, 1499, "Mining")`, `C (1500, 1799, "Construction")`, `D (2000, 3999, "Manufacturing")`, `E (4000, 4999, "Transportation, Communications, Utilities")`, `F (5000, 5199, "Wholesale Trade")`, `G (5200, 5999, "Retail Trade")`, `H (6000, 6799, "Finance, Insurance, Real Estate")`, `I (7000, 8999, "Services")`, `J (9100, 9729, "Public Administration")`, `K (9900, 9999, "Nonclassifiable")`.
  - `sic_division(sic: int | None) -> str | None` (the letter; `None` for missing or out-of-range codes).
  - `SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"`.
  - `read_sic_codes(zip_path: Path, ciks: Iterable[int]) -> dict[int, int | None]`: reads only the members `CIK{cik:010d}.json` for the given CIKs; `int(data["sic"])` when present and non-empty, else `None`; a CIK with no member maps to `None`.

- [ ] **Step 1: Write the failing tests:** boundary cases for `sic_division` (`999 → "A"`, `1000 → "B"`, `3999 → "D"`, `4000 → "E"`, `6799 → "H"`, `6800 → None`, `9999 → "K"`, `None → None`); and `read_sic_codes` against a zip built in `tmp_path` with two members (one with `"sic": "7372"`, one with `"sic": ""`) plus a third CIK with no member, expecting `{320193: 7372, 1018724: None, 999: None}`.
- [ ] **Step 2: Run** `uv run pytest tests/test_industry.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with `zipfile.ZipFile`; never extract the whole archive.
- [ ] **Step 4: Run** the tests. Expected: all passed.
- [ ] **Step 5: Commit** `"Add SIC codes and divisions from SEC submissions data"`.

### Task 5: The filings stage, run on the real data

**Files:**
- Create: `pipeline/run.py`, `pipeline/stages.py`, `tests/test_stages.py`, `docs/data-foundation.md`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `build_filings(cfg: Config, force: bool = False) -> Path` writing `data/interim/filings.parquet` with columns `cik: int`, `filing_year: int`, `item_1a_length: int`, `date_filed`, `form_type`, `n_candidates`, `match_status`, `sic: Int64`, `sic_division: str | None`; returns the path, and skips work when the file exists unless `force`. CLI: `python -m pipeline.run {filings,sentences,all} [--force] [--workers N]`.

- [ ] **Step 1: Write the failing test** `test_build_filings_on_fixtures`: patch `fetch` to copy the Task 3 fixture for every index URL and a two-member zip for submissions; a three-filing parquet in `tmp_path`; asserts the output columns above, one row per input filing, and that a second call with `force=False` makes no `fetch` calls.
- [ ] **Step 2: Run** it. Expected: FAIL.
- [ ] **Step 3: Implement** `build_filings` and the CLI (argparse).
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all passed.
- [ ] **Step 5: Run on the real data:** add the `SEC_USER_AGENT` line to `.env`, then `uv run python -m pipeline.run filings`. Expected: about 60 index files and one submissions zip cached under `data/raw/edgar/`, and `data/interim/filings.parquet` with 72,613 rows. Print and record in `docs/data-foundation.md`: the share of each `match_status`, the share with a SIC division, and filings by division. **Gate:** if fewer than 95% of filings are `unique` or `multiple`, stop and investigate the join (likely cause: the corpus year isn't the filing year) before going on.
- [ ] **Step 6: Commit** `"Build the filings table with filing dates and industry"` (code and doc; data stays ignored).

### Task 6: Sentence splitting, debris rules, normalization

**Files:**
- Create: `pipeline/sentences.py`, `tests/test_sentences.py`, `scripts/compare_splitters.py`

**Interfaces:**
- Produces:
  - `split_sentences(text: str) -> list[str]`
  - `is_debris(sentence: str) -> bool`: true when it has fewer than 15 letters, or letters are under half of its non-space characters.
  - `LONG_SENTENCE_CHARS = 1500` (kept but flagged).
  - `normalize(sentence: str) -> str` per Global Constraints.
  - `norm_hash(norm: str) -> int`: the first 8 bytes of `hashlib.blake2b(norm.encode(), digest_size=8)` as a signed int64.
  - `filing_sentences(cik: int, filing_year: int, text: str) -> pd.DataFrame` with `cik`, `filing_year`, `sent_idx` (0-based position among kept sentences), `text`, `norm`, `norm_hash`, `n_chars`, `is_long`; debris rows dropped.

- [ ] **Step 1: Write the failing tests.** `split_sentences` cases, each asserting the exact list:
  - `"We are subject to U.S. federal regulation. Changes could harm us."` → 2 sentences.
  - `"We borrowed $30.0 million under the facility. Interest rates may rise."` → 2.
  - `"Competitors include Acme Corp. and Beta Inc. These firms are larger."` → 2.
  - `"See Item 7, e.g., liquidity. We may need capital."` → 2.
  - `"Risks Related to Our Business\nWe depend on a few customers."` → 2 (a short line with no end punctuation followed by a capitalized line is a heading).
  - `"(1) our ability to borrow; (2) our ability to refinance."` → 1.
  - `""` → `[]`; `"Not applicable."` → `["Not applicable."]`.

  `normalize("The Company’s 2019 revenue fell 12.5%.") == "the company s 0000 revenue fell 00 0"`. `is_debris` true for `"2019 2018 2017 $ 1,234 $ 5,678"` and `"(a)"`, false for `"We may be unable to attract and retain key personnel."`. `filing_sentences` drops debris, numbers `sent_idx` contiguously, sets `is_long` for a 1,600-character sentence, and gives equal `norm_hash` to two sentences that differ only in digits and punctuation.
- [ ] **Step 2: Run** `uv run pytest tests/test_sentences.py -v`. Expected: FAIL.
- [ ] **Step 3: Choose the splitter.** `scripts/compare_splitters.py` runs a regex baseline and `pysbd` (`uv add pysbd`) on the Step 1 cases and on 50 sections sampled with `random_state=0`, printing per splitter: cases passed, sentences per section, share of sentences over 1,500 characters, share under 40 characters, and throughput in MB/s. Pick the splitter that passes every case, provided the full corpus (about 3.6 GB of text) would finish in under 2 hours on 8 workers; if neither qualifies, the faster one plus newline and abbreviation rules until the cases pass. Record the comparison table and the choice in `docs/data-foundation.md`.
- [ ] **Step 4: Implement** `pipeline/sentences.py` with the chosen splitter.
- [ ] **Step 5: Run** `uv run pytest -v && uv run ruff check .`. Expected: all passed.
- [ ] **Step 6: Commit** `"Split risk sections into normalized sentences"`.

### Task 7: The sentence table, run and reviewed

**Files:**
- Modify: `pipeline/stages.py`, `pipeline/run.py`, `tests/test_stages.py`, `docs/data-foundation.md`

**Interfaces:**
- Consumes: Task 6; the text is `section_1A` from `data/processed/risk_factors_2006_2020.parquet`, read one `filing_year` at a time (`pyarrow` filters) with `cik` cast to `int`.
- Produces: `build_sentences(cfg: Config, workers: int = 8, force: bool = False) -> Path` writing one zstd parquet per year to `data/interim/sentences/year=YYYY.parquet` (the `filing_sentences` columns), processing a year at a time so memory stays bounded, skipping years already written unless `force`. Milestone 2 reads these files.

- [ ] **Step 1: Write the failing test** `test_build_sentences_on_fixture`: a four-filing parquet across two years (one with empty text); asserts two year files, the empty filing contributes zero rows, `(cik, filing_year, sent_idx)` is unique, and a rerun without `force` rewrites nothing (compare file modification times).
- [ ] **Step 2: Run** it. Expected: FAIL.
- [ ] **Step 3: Implement** with `concurrent.futures.ProcessPoolExecutor` over filings within a year.
- [ ] **Step 4: Run** `uv run pytest -v`. Expected: all passed.
- [ ] **Step 5: Run on the real data:** `uv run python -m pipeline.run sentences --workers 8`, timing it. Record in `docs/data-foundation.md`: total sentences, sentences per filing (median, 10th and 90th percentiles), share dropped as debris, share flagged long, runtime, and disk size.
- [ ] **Step 6: Review the output.** Sample 50 filings (`random_state=0`) and read their first 20 sentences against the source text; note the failure types found and their rough frequency in `docs/data-foundation.md`. Check heading detection on the same sample: if headings are flagged correctly in at least 45 of 50 filings, keep `is_heading` (add it as a column, true for lines handled by the heading rule); otherwise leave it out and say so in the doc, as the spec allows.
- [ ] **Step 7: Commit** `"Build the sentence table and record the data checks"`.

---

## Later milestones (each gets its own plan when it starts)

Each is a child ticket of #1, planned in detail when it begins, because its details depend on what milestone 1 finds.

- **Milestone 2, Measurement:** exact-hash carryover between consecutive filings; MinHash on the remainder; embeddings on what's still unmatched; peer-sharing counts by year; the known-answer set (rule-built verbatim pairs, Haiku-built paraphrases and hard negatives through `claude -p`, resumable); Haiku as judge 1 on about 2,000 real pairs and Claude in-session as judge 2 on about 400; thresholds τ_v and τ_p with precision and recall; the three baselines; filing-level shares.
- **Milestone 3, Explain:** sentence chains across consecutive years; Kaplan–Meier curves; a hazard model with company frailty (or discrete-time if proportional hazards fails); the variance decomposition by company, division and year.
- **Milestone 4, Test:** the 2020 difference-in-differences by filing month against 2017–2019, the event-study plot, placebo shocks in 2018 and 2019, the COVID-term manipulation check.
- **Milestone 5, Publish:** report sections and the JSON they read, the README rewrite, the `v1-original` tag, and removal of the old modeling code, `requirements.txt`, and the course outline PDF.
