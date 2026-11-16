# Milestone 2: Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label every sentence of every filing that has a previous-year filing as carried over (exact, near-verbatim, paraphrased) or new, label the previous filing's sentences kept or removed, count how many unrelated companies share each sentence, and prove the labels work: a known-answer test set, two LLM judges on real pairs, thresholds with precision and recall, and baselines.

**Architecture:** Matching is split into a threshold-free half and a cheap half. The threshold-free half computes, for every unmatched sentence, its best word-5-shingle Jaccard and best embedding cosine against the other filing's unmatched sentences, and stores the scores. Validation then tunes two thresholds on those scores, and labels are a pure function of scores and thresholds, so thresholds can change without recomputing anything heavy. LLM work (rewrites, confirmations, judging) goes through `claude -p`, is resumable, and its prompts, raw responses and labels are committed under `validation/`, so a rebuild reads them and never needs to call a model.

**Tech Stack:** Python 3.12, uv, pandas, pyarrow, numpy, scipy (sparse products, connected components), sentence-transformers on the Apple GPU (MPS), scikit-learn (average precision, kappa), model2vec (a static-embedding baseline in the model comparison), Claude Code headless mode for LLM calls.

**Spec:** `docs/superpowers/specs/2026-11-09-copying-rework-design.md` (Layer 1). Data notes: `docs/data-foundation.md`.

## Measured before planning (2026-11-11, fiscal 2018 → 2019)

These sized the design. They are from throwaway runs; the stage outputs replace them in `docs/measurement.md`.

- 3,045 companies have both years. Exact matching alone carries 77.0% of current-year sentences and leaves about 290,000 current and 254,000 previous sentences unmatched. The median filing has 66 unmatched current sentences, the 99th percentile 509. So the later stages compare small sets, and exact Jaccard is affordable inside a company's pair.
- 35.2% of 2019 sentences appear word for word in another company's 2019 filing, but 311 companies belong to groups whose sections are more than half identical (a parent and its subsidiaries, fund families). Counting groups instead of companies gives 30.8%. The most-shared "sentences" are headings and list fragments ("Item 1A. Risk Factors.", "our reputation may suffer.").
- Embedding speed on this laptop: all-MiniLM-L6-v2 about 1,400 sentences a second, bge-small-en-v1.5 about 730. Median sentence 34 tokens, 99th percentile 106; 0.025% exceed MiniLM's 256-token limit.
- `claude -p --model haiku --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands` returns in under 2 seconds with no hooks or project context loaded.

## Global Constraints

- Python 3.12 via uv; every new dependency goes in `pyproject.toml` and `uv.lock`.
- Sentence normalization and `norm_hash` are milestone 1's, unchanged (`pipeline/sentences.py`).
- Only consecutive years link: a filing for year *y* is compared with the same company's filing for *y* − 1 and nothing else.
- Word 5-shingles on the normalized text; a sentence of fewer than 5 words has one shingle, the whole sentence.
- No human labels and no API credits. LLM calls go through `claude -p` on Joe's plan, in batches of 25–50 items, about 100–150 calls in all; every call's prompt and raw response is saved; a rerun skips batches already saved.
- Judge 2 (Claude in session) labels blind: its export has no scores, no strata and no judge-1 labels, in shuffled order.
- Random choices use seed 0.
- Interim files under `data/interim/` (git-ignored). Validation data under `validation/` and results under `results/` are committed.
- Every number in `docs/measurement.md` is printed by a command named next to it and stored in `results/measure.json`.
- Decision rules are fixed here, before any result is seen: how thresholds are tuned (Task 7), what agreement counts as good (Task 6), and the primary peer-sharing definition (Task 9).

## Decisions this plan makes (the spec left them to the builder)

- **Exact Jaccard inside a company's pair; MinHash only across companies.** The spec says "Jaccard ≥ τ_v via MinHash". Inside a pair the candidate sets are small enough to compute Jaccard exactly, which has no estimation error. MinHash with banding is used where an exact comparison is impossible: near-verbatim sharing across a year's 1.3 million distinct sentences.
- **Set semantics.** A current sentence is carried over if *some* previous sentence matches it; a previous sentence is kept if *some* current sentence matches it. Several sentences may match the same one. One-to-one links for survival chains are milestone 3's job, built from the stored best-match indexes.
- **Unmatched against unmatched.** Stages 2 and 3 compare the sentences exact matching left over on each side. A new sentence that rewords a sentence which was *also* carried over verbatim is therefore called new. Task 10 measures how often on a sample.
- **Related filers are one company for peer sharing.** Companies whose sets of distinct sentences have Jaccard ≥ 0.5 in a year form a group; sharing is counted across groups.
- **The embedding model is chosen by test** (Task 4), not by habit.

## Review Focus

1. A sentence under 5 words: Jaccard is 1.0 against an identical sentence and 0.0 otherwise, never a division by zero. (Task 1)
2. A pair where one side has nothing unmatched (everything carried over, or last year's section was "Not applicable."): scores are null, and labels come out new or removed without error. (Task 2, Task 8)
3. The same sentence twice in one filing: both copies are carried over if it was in the previous filing; counts are per sentence, not per distinct sentence. (Task 2)
4. A gap year (2017 and 2019 present, 2018 missing): the 2019 filing is not paired with 2017. (Task 2)
5. An LLM batch that returns malformed JSON, the wrong number of items, or an id that wasn't asked for: it is retried, then recorded as failed and left for the next run; a partial batch is never accepted. (Task 3)
6. An interrupted embedding run: a year's file appears only when complete, and a rerun resumes at the first missing year. (Task 5)

---

### Task 1: Shingles, Jaccard and MinHash

**Files:** Create `pipeline/shingles.py`, `tests/test_shingles.py`.

**Interfaces (produces):**
- `shingles(norm: str, k: int = 5) -> frozenset[str]`
- `jaccard(a: frozenset[str], b: frozenset[str]) -> float`
- `best_matches(left: list[frozenset[str]], right: list[frozenset[str]]) -> tuple[np.ndarray, np.ndarray]`: for each left set, the best Jaccard against any right set and that set's position (−1 and 0.0 when nothing overlaps). Uses an inverted index over `right`, so cost follows shared shingles, not `len(left) * len(right)`.
- `shingle_hashes(norms: Sequence[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]`: 64-bit shingle hashes for many sentences as one flat array plus each sentence's start offset. Vectorized: words are mapped to integer ids and a polynomial hash is taken over each window. Deterministic across processes (no use of Python's salted `hash`).
- `minhash(hashes: np.ndarray, starts: np.ndarray, n_perm: int = 126, seed: int = 0) -> np.ndarray` (`uint32`, shape `[n_sentences, n_perm]`)
- `lsh_candidates(signatures: np.ndarray, queries: np.ndarray, bands: int = 42, rows: int = 3, max_bucket: int = 2000) -> list[np.ndarray]`: for each query row, the rows sharing at least one band. A bucket over `max_bucket` is sampled down to it with seed 0.

- [ ] **Step 1: Failing tests.** `shingles("a b c d e f")` has 2 members; `shingles("a b c")` has exactly one; `jaccard` of a set with itself is 1.0, of disjoint sets 0.0, and of `shingles("a b c d e f g")` with `shingles("a b c d e f x")` is 2/4; `jaccard(frozenset(), frozenset())` is 0.0. `best_matches` equals a brute-force double loop on 200 random sentences built from a 30-word vocabulary. `shingle_hashes` gives equal hashes for equal windows and the same output in a subprocess started with a different `PYTHONHASHSEED`. Across 2,000 random pairs, the share of agreeing MinHash positions is within 0.02 of exact Jaccard on average, and no pair is off by more than 0.2. Planted pairs with Jaccard ≥ 0.5 among 5,000 sentences are returned by `lsh_candidates` at least 95% of the time.
- [ ] **Step 2: Run** `uv run pytest tests/test_shingles.py -q`. Expected: fails on import.
- [ ] **Step 3: Implement.** MinHash permutation *i* is `(a_i * h + b_i) mod 2**64` keeping the top 32 bits, with odd `a_i`; the per-sentence minimum is `np.minimum.reduceat`.
- [ ] **Step 4: Run** the tests and `uv run ruff check pipeline tests`. Expected: pass, clean.
- [ ] **Step 5: Commit** "Add shingles, exact Jaccard and MinHash".

### Task 2: Candidates: exact matches and best Jaccard for every pair

**Files:** Create `pipeline/pairs.py`, `pipeline/candidates.py`, `tests/test_pairs.py`, `tests/test_candidates.py`. Modify `pipeline/stages.py`, `pipeline/run.py`.

**Interfaces:**
- `consecutive_pairs(filings: pd.DataFrame) -> pd.DataFrame` with columns `cik`, `filing_year`, `prev_year` (always `filing_year - 1`), one row per filing whose previous year exists.
- `match_pair(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame`: `cur` and `prev` are one company's sentence rows (`sent_idx`, `norm`, `norm_hash`). Returns one row per sentence of both filings: `side` (`cur` | `prev`), `sent_idx`, `exact` (bool), `exact_idx` (the first sentence on the other side with the same hash, else −1), `best_j` (float, null when `exact` or when the other side has nothing unmatched), `j_idx` (−1 when null).
- Stage `build_candidates(cfg, workers=8, force=False) -> Path`: writes `data/interim/candidates/pair=YYYY.parquet` for each year 2007–2020, columns `cik`, `side`, `sent_idx`, `exact`, `exact_idx`, `best_j`, `j_idx`. Companies are processed in batches across a process pool, a year at a time, written through a `.part` file.
- CLI: `python -m pipeline.run candidates`, printing per pair-year: pairs, sentences, share exact on each side, and the count left unmatched.

- [ ] **Step 1: Failing tests.** Pairs: a company with 2017 and 2019 only yields no pair; one with 2018 and 2019 yields exactly `(cik, 2019, 2018)`. `match_pair` on a six-sentence fixture: an identical sentence is `exact` on both sides with `exact_idx` pointing at its counterpart; a sentence appearing twice in `cur` and once in `prev` is `exact` twice; a sentence with one word changed out of twelve has `best_j` equal to the hand-computed 3/13 and `j_idx` at its source; a sentence with no counterpart has `best_j == 0.0` and `j_idx == -1`; when every `prev` sentence is exact-matched, the leftover `cur` sentence has null `best_j`. An empty `prev` frame returns `cur` rows all not exact with null scores.
- [ ] **Step 2: Run** the two test files. Expected: fail on import.
- [ ] **Step 3: Implement**, then the stage and CLI command.
- [ ] **Step 4: Run** tests and ruff, then `uv run python -m pipeline.run candidates`. Expected: 14 files; the 2019 line shows 3,045 pairs and an exact share near 77% of current sentences.
- [ ] **Step 5: Commit** "Match consecutive filings exactly and score the rest by Jaccard".

### Task 3: The LLM runner and the known-answer set

**Files:** Create `pipeline/llm.py`, `pipeline/known_answer.py`, `tests/test_llm.py`, `tests/test_known_answer.py`, `validation/known_answer/` (data), `validation/README.md`.

**Interfaces:**
- `run_batches(items: list[dict], *, prompt_file: Path, out_dir: Path, model: str, batch_size: int, required: tuple[str, ...], call=claude_cli, max_retries: int = 2) -> list[dict]`. Each item has a string `id`. For batch *n* it writes `out_dir/batch_NNN.request.json` and, on success, `out_dir/batch_NNN.response.json` holding the CLI's raw JSON. A batch succeeds only if the reply parses as a JSON array with exactly the requested ids, each object carrying every `required` key. A batch whose response file exists is not called again. Returns the parsed objects of all successful batches. `call(prompt: str, model: str) -> str` is injectable so tests never start a process.
- `claude_cli(prompt, model)`: runs `claude -p --model <model> --output-format json --tools "" --strict-mcp-config --setting-sources "" --disable-slash-commands --no-session-persistence` from an empty temporary directory with the prompt on stdin, and returns the `result` field.
- `sample_bases(cfg, n_per_year: int = 40, seed: int = 0) -> pd.DataFrame`: 600 sentences, 40 per fiscal year, drawn from sentences that are not headings, not flagged long, and have at least 8 words.
- Stage `build_known_answer(cfg)`: writes `validation/known_answer/pairs.csv` with columns `id`, `kind` (`verbatim` | `paraphrase` | `different_hard` | `different_easy`), `split` (`tune` | `test`, assigned by base sentence, half each, seed 0), `a`, `b`, `cik`, `filing_year`, `sent_idx`, `confirmed`.

**How each kind is built** (prompts live in `validation/known_answer/prompts/`, versioned by filename):
- `verbatim`: Haiku is told to change only names, dates and numbers, or one or two words where the sentence has none, without changing what it claims. Kept if at least 60% of its words survive in order (`difflib.SequenceMatcher` on tokens).
- `paraphrase`: Haiku rewrites to keep the meaning and change the wording.
- `different_hard`: Haiku keeps the topic and most of the words and changes the claim (reverses it, or names a different risk).
- `different_easy`: a random sentence from another company and another year (no model).
- **Confirmation by a second model:** every `paraphrase` and `different_hard` pair goes to Sonnet (`claude -p --model sonnet`) in shuffled batches of 50, asked only whether A and B make the same claim. A paraphrase is kept if Sonnet says same; a hard edit is kept if Sonnet says different. Rejections are counted and reported.

- [ ] **Step 1: Failing tests** (fake `call`). A good reply writes both files and returns the objects. A reply that isn't JSON, has a missing id, an extra id, or lacks a required key is retried `max_retries` times and then leaves no response file and returns nothing for that batch, while other batches still return. A second run with all response files present makes zero calls. A reply wrapped in a Markdown code fence is accepted. `sample_bases` on a synthetic sentence table returns `n_per_year` rows per year, none a heading, none under 8 words, and the same rows on a second call.
- [ ] **Step 2: Run.** Expected: fail on import.
- [ ] **Step 3: Implement** `pipeline/llm.py` and the sampling and assembly in `pipeline/known_answer.py`.
- [ ] **Step 4: Run** tests and ruff.
- [ ] **Step 5: Generate** with `uv run python -m pipeline.run known-answer`. Expected: about 24 Haiku calls and 24 Sonnet calls; `pairs.csv` with roughly 2,400 rows less rejections; the command prints kept and rejected counts per kind. Read 20 random rows of each kind before going on, and note anything wrong in `validation/README.md`.
- [ ] **Step 6: Commit** "Build the known-answer set" (code, prompts, raw responses, `pairs.csv`).

### Task 4: Choose the embedding model

**Files:** Create `pipeline/embed.py`, `scripts/compare_embedders.py`, `tests/test_embed.py`.

**Interfaces:**
- `load_embedder(name: str) -> Embedder`, with `Embedder.encode(texts: list[str], batch_size: int = 128) -> np.ndarray` returning L2-normalized `float32` rows, texts sorted by length internally and returned in input order. Runs on MPS when available.
- `scripts/compare_embedders.py`: on the `tune` half of the known-answer set, for each candidate, the average precision of cosine at separating `paraphrase` from `different_easy` and `different_hard`, the same against each negative kind alone, sentences per second on 10,000 corpus sentences, and the share of those sentences over the model's token limit.

**Candidates:** `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `minishlab/potion-base-8M` (static; shows what word content alone gets).

**Rule:** highest average precision on the tune half wins; within 0.005, the faster model wins. The result table goes in `docs/measurement.md`, and the chosen name in `pipeline/embed.py` as `MODEL`.

- [ ] **Step 1: Failing tests.** With a stub model, `encode` returns rows in input order whatever the lengths, unit-norm, and `float32`; an empty list returns shape `(0, dim)`.
- [ ] **Step 2–4:** run, implement, run.
- [ ] **Step 5: Run** `uv run python -m scripts.compare_embedders`, set `MODEL`, commit "Choose the embedding model on the known-answer set".

### Task 5: Embed the unmatched sentences and score them by cosine

**Files:** Modify `pipeline/embed.py`, `pipeline/candidates.py`, `pipeline/stages.py`, `pipeline/run.py`. Create `tests/test_cosine.py`.

**Interfaces:**
- Stage `build_embeddings(cfg, force=False) -> Path`: for each fiscal year, the distinct `norm_hash` values of sentences that are unmatched on either side of a pair (current side of pair *y*, previous side of pair *y* + 1), each embedded once from its first raw text. Writes `data/interim/embeddings/year=YYYY.npy` (`float16`) and `year=YYYY.parquet` (`norm_hash`, `row`), both through `.part` names; years already complete are skipped.
- `best_cosine(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: per left row, the best cosine against any right row and its position; empty `right` gives nulls and −1.
- Stage `build_similarity(cfg, force=False) -> Path`: adds `best_cos` and `cos_idx` to the unmatched rows of each pair-year and writes `data/interim/similarity/pair=YYYY.parquet` with the candidates' columns plus those two.

- [ ] **Step 1: Failing tests.** `best_cosine` on three hand-built unit vectors returns the right scores and positions; with empty `right` it returns NaN and −1; identical rows score 1.0 within 1e-3 after a `float16` round trip.
- [ ] **Step 2–4:** run, implement, run.
- [ ] **Step 5: Run** `uv run python -m pipeline.run embeddings similarity` (the long step; it prints sentences per second and can be interrupted and resumed). Expected: 15 embedding files and 14 similarity files; every unmatched row whose other side is non-empty has a `best_cos`.
- [ ] **Step 6: Commit** "Embed unmatched sentences and score them by cosine".

### Task 6: Real pairs and two judges

**Files:** Create `pipeline/judging.py`, `tests/test_judging.py`, `validation/real_pairs/` (data), `validation/real_pairs/rubric_v1.md`.

**Interfaces:**
- `sample_real_pairs(cfg, n: int = 2000, seed: int = 0) -> pd.DataFrame`: unmatched current sentences with their best-cosine partner, stratified on that pair's cosine (`<0.4`, then tenths to 1.0) crossed with its Jaccard (`<0.1`, `0.1–0.3`, `0.3–0.6`, `≥0.6`), an equal number from each non-empty cell (all of a cell smaller than its share). Columns: `id`, `a` (previous sentence), `b` (current), `cik`, `filing_year`, `stratum`, `pair_j`, `pair_cos`, `weight` (the cell's population divided by its sample size).
- Writes `validation/real_pairs/pairs.csv` (all columns), `judge1_input.csv` and `judge2_input.csv` (`id`, `a`, `b` only, each separately shuffled; judge 2's is a 400-pair subset with an equal number per cell).
- `judge_with_haiku(cfg)`: `run_batches` over `judge1_input.csv`, 40 per call, writing `judge1_labels.csv` (`id`, `label`).
- `agreement(a: pd.Series, b: pd.Series) -> dict`: Cohen's kappa on three labels, kappa on carried-over (`same` or `paraphrase`) against `different`, raw agreement, and the confusion table.

**Rubric v1** (the file holds the full text with two examples per label): `same` = the same sentence with at most light edits (names, dates, numbers, a word or two) that leave its claim unchanged. `paraphrase` = the same claim about the same risk in substantially different words, with nothing material added or dropped. `different` = a different claim: another risk, the opposite claim, or materially more or less than the first sentence says.

**What counts as good agreement (fixed now):** three-label kappa ≥ 0.70 on the shared 400. If it is met, judge 1's labels are the real-data check for all 2,000. If not, the doc says so and the real-data check uses only pairs where the two judges agree.

- [ ] **Step 1: Failing tests.** On a synthetic score table, `sample_real_pairs` returns equal counts per non-empty cell, weights that sum to the population within each cell, and the same sample twice. The two input files contain only `id`, `a`, `b`. `agreement` on a hand-built 3×3 table returns the hand-computed kappa.
- [ ] **Step 2–4:** run, implement, run.
- [ ] **Step 5: Judge 1.** `uv run python -m pipeline.run judge`. Expected: about 50 Haiku calls, 2,000 labels.
- [ ] **Step 6: Judge 2.** Claude, in session, reads `judge2_input.csv` and writes `judge2_labels.csv` without opening `pairs.csv` or judge 1's labels. Record in `validation/README.md` who labeled, when, and under which rubric version.
- [ ] **Step 7: Commit** "Sample real pairs and label them with two judges".

### Task 7: Thresholds

**Files:** Create `pipeline/thresholds.py`, `tests/test_thresholds.py`.

**Interfaces:**
- `score_known_answer(cfg) -> pd.DataFrame`: `pairs.csv` plus `j` and `cos` for each pair, computed with Task 1's and Task 4's functions.
- `tune(scores: pd.DataFrame) -> dict`: on `split == "tune"` only. `tau_v` is the Jaccard threshold with the best F1 at separating `verbatim` from `paraphrase` and `different_easy`. `tau_p` is the cosine threshold with the best F1 at separating `paraphrase` from `different_easy` and `different_hard`, among pairs under `tau_v`. Ties go to the higher threshold. `different_hard` is left out of the first on purpose: a reversed claim that keeps its words is indistinguishable from a light edit by word overlap, and Task 7 reports how often that happens instead of tuning around it.
- `evaluate(scores, tau_v, tau_p) -> dict`: on the `test` half, precision and recall of each tier with Wilson 95% intervals, and how each of the four kinds was labeled. On the real pairs, the weighted share judged `same`, `paraphrase` and `different` within each predicted tier, and the weighted recall of carried-over pairs.
- Writes `results/thresholds.json`.

- [ ] **Step 1: Failing tests.** On a hand-built table where a threshold of 0.5 separates the classes perfectly, `tune` returns a threshold between the two nearest scores and F1 of 1.0; rows with `split == "test"` do not change it; `evaluate` on six rows returns the hand-computed precision, recall and interval.
- [ ] **Step 2–4:** run, implement, run.
- [ ] **Step 5: Run** `uv run python -m pipeline.run thresholds`. Commit "Tune and evaluate the two thresholds".

### Task 8: Labels and filing-level shares

**Files:** Create `pipeline/labels.py`, `tests/test_labels.py`. Modify `pipeline/stages.py`, `pipeline/run.py`.

**Interfaces:**
- `label(scores: pd.DataFrame, tau_v: float, tau_p: float) -> pd.Series`: `exact`, `near`, `paraphrase`, or `new` (current side) / `removed` (previous side). Null scores count as below threshold.
- Stage `build_labels(cfg)`: `data/interim/labels/pair=YYYY.parquet` (`cik`, `side`, `sent_idx`, `label`, `link_idx`: the matched sentence on the other side, by the tier that matched, else −1).
- Stage `build_filing_metrics(cfg)`: `data/interim/filing_metrics.parquet`, one row per paired filing: `n_sentences`, `n_prev`, `share_exact`, `share_near`, `share_paraphrase`, `share_new`, `share_removed` (of the previous filing's sentences), the same five weighted by characters (`*_chars`), and `n_chars`, `n_prev_chars`.

- [ ] **Step 1: Failing tests.** `label` on rows covering every branch, including null scores and a score exactly at each threshold (at the threshold counts as a match). On a two-filing fixture, the four current shares sum to 1 and `share_removed` equals the hand count.
- [ ] **Step 2–5:** run, implement, run, then `uv run python -m pipeline.run labels metrics`; commit "Label sentences and compute filing-level shares".

### Task 9: Peer sharing

**Files:** Create `pipeline/peers.py`, `tests/test_peers.py`. Modify stages and CLI.

**Interfaces:**
- `filer_groups(sentences: pd.DataFrame, threshold: float = 0.5) -> pd.Series`: for one year, a group id per `cik`; companies whose sets of distinct `norm_hash` have Jaccard ≥ `threshold` are joined (sparse product, then connected components).
- Stage `build_peers(cfg)`: `data/interim/peers/year=YYYY.parquet` with `norm_hash`, `n_companies`, `n_groups`; and `data/interim/filer_groups.parquet` (`cik`, `filing_year`, `group`).
- `near_peer_sample(cfg, n_per_year: int = 20000, seed: int = 0) -> pd.DataFrame`: for a random sample of sentences each year, the number of other groups holding a sentence with estimated Jaccard ≥ `tau_v` (MinHash, Task 1), next to the exact count.

**Primary definition (fixed now):** a sentence is *shared with peers* when its normalized form appears in the same year in at least 5 other groups. Reported also at 1, 10 and 25, for all sentences and for full sentences (at least 10 words, not a heading). The near-verbatim count is reported as a sensitivity on the sample.

- [ ] **Step 1: Failing tests.** Three companies where two share 4 of 5 sentences and the third shares 1: the first two form one group, the third its own; a sentence in all three then has `n_companies == 3` and `n_groups == 2`.
- [ ] **Step 2–5:** run, implement, run, `uv run python -m pipeline.run peers`; commit "Count how many unrelated companies share each sentence".

### Task 10: Baselines and robustness

**Files:** Create `pipeline/baselines.py`, `tests/test_baselines.py`.

**Interfaces (each returns a dict written into `results/baselines.json`):**
- `random_pairs(cfg, n=200_000)`: sentence pairs from different companies and different years; the share at or above each threshold.
- `stranger_filings(cfg, n_pairs=2000)`: for sampled filing pairs, the current filing matched against the previous-year filing of a random other company (and, separately, a random other company in the same SIC division) through the same three stages; the four label shares, next to the same filings' real shares.
- `shuffled_words(cfg, n=20_000)`: each sentence against itself with its words shuffled; the distribution of cosine and Jaccard, and the share at or above `tau_p`.
- `unmatched_only_check(cfg, n_pairs=500)`: for sampled pairs, sentences labeled new are compared with *all* previous sentences; the share that would be relabeled.

- [ ] **Step 1: Failing tests.** `random_pairs` never pairs a company with itself or a year with itself; `shuffled_words` keeps each sentence's multiset of words; on a fixture where a new sentence rewords an exact-matched one, `unmatched_only_check` reports it.
- [ ] **Step 2–5:** run, implement, run, `uv run python -m pipeline.run baselines`; commit "Add baselines and the unmatched-only check".

### Task 11: Results file and the measurement write-up

**Files:** Create `pipeline/report_data.py` (the measurement part), `results/measure.json`, `docs/measurement.md`. Modify `pipeline/run.py` (`measure-report`, and `all` running every stage in order).

- `build_measure_results(cfg) -> dict` gathers: pairs and sentences by year; pooled and per-filing shares by year (median and quartiles), by sentences and by characters; threshold results; judge agreement; the model comparison; baselines; peer sharing at each k by year; examples drawn by rule (the first 3 sentences by `(cik, filing_year, sent_idx)` order within each label among filings sampled with seed 0), each with its EDGAR accession.
- `docs/measurement.md` follows `docs/data-foundation.md`: what was built, each check with its command, the numbers, and the limits. Every number is read from `results/measure.json`.

- [ ] **Step 1:** test that `build_measure_results` on a small fixture directory produces every top-level key and that shares are between 0 and 1.
- [ ] **Step 2–4:** implement, run `uv run python -m pipeline.run measure-report`, write the doc from the printed tables.
- [ ] **Step 5:** `uv run pytest -q && uv run ruff check .`; commit "Record the measurement results"; open the PR with `Closes #7`.

## Self-review

- **Spec coverage.** Normalization (M1, unchanged). Labels verbatim, paraphrased, new, removed: Tasks 2, 5, 8. Shared with peers at several k: Task 9. Filing-level shares, sentence- and character-weighted: Task 8. Staged for scale, no truncation: Tasks 2, 4, 5 (Task 4 reports the share over each model's limit). Known-answer set with four kinds and second-model confirmation: Task 3. Two judges, kappa, fallback: Task 6. Thresholds tuned on half, reported on the other half and on real pairs: Task 7. Stated limits: Task 11. Three baselines: Task 10 (plus the stranger-filing baseline and the unmatched-only check, which the spec didn't ask for). Resumable labeling, saved prompts and responses: Task 3.
- **Departures from the spec, each stated above:** exact Jaccard inside a pair with MinHash across companies; `different_hard` left out of the τ_v tuning; related filers grouped for peer sharing; near-verbatim peer sharing on a sample.
- **Types.** `best_j`/`j_idx`/`best_cos`/`cos_idx`/`exact`/`exact_idx` are the same names in Tasks 2, 5, 6, 8 and 10; `tau_v`/`tau_p` in Tasks 7, 8, 9, 10.
