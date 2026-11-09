# Rework design: how much of a 10-K risk section is boilerplate?

Ticket: [#1](https://github.com/jlattanzi4/sec-risk-classifier/issues/1). Status: draft for Joe's approval, 2026-11-09.

## Intent

The first version of this project made two headline claims that a review in November 2026 retracted (see #2): a "51% vs 21%" similarity gap that compared scores computed on different amounts of text, and a 71.2% F1 from a leaky evaluation. This rework replaces them with findings that hold up.

The project is a portfolio piece. Its readers are hiring managers and senior data scientists skimming Joe's GitHub. It should leave them thinking: this person can measure something messy, prove the measurement works, and tell what's real from what isn't. The outcomes have to be real (every number computed, validated, and reproducible from one command) and the work should show professional data science skills, including some Joe is growing into, applied to a question that matters.

It matters because regulators asked it. The SEC's 2020 amendments to Regulation S-K Item 105 responded to risk-factor sections that had grown long and generic: they require a summary when the section runs past 15 pages and push generic risks to the end. This dataset ends just before those rules took effect.

## The central question

> How much of a company's risk section is boilerplate, carried over from its own last filing or shared word-for-word with other companies, and what makes companies actually rewrite?

Answered in three layers, each building on the one before and each showing a different skill:

1. **Measure** (measurement design and validation): a sentence-level copying metric, checked against human labels and baselines.
2. **Explain** (modeling): how long risk sentences survive once added, and how much rewriting is driven by the company, its industry, and the year.
3. **Test** (causal inference): when COVID hit in 2020, did companies rewrite their risk sections or bolt new paragraphs onto unchanged text?

## Goals, non-goals, anti-goals

**Goals**
- A validated, documented definition of "carried over", "paraphrased", "shared with peers", and "new" at the sentence level, with precision and recall on test cases whose answers are known.
- Every headline number shown next to a baseline that says what "high" means.
- Survival analysis of risk sentences and a variance decomposition of rewriting.
- A 2020 natural-experiment analysis with placebo checks and stated limits.
- One command rebuilds everything from the processed filings to `docs/report.html`, with pinned dependencies.

**Non-goals**
- Classification of risks into categories, DistilBERT, SHAP, and BERTopic. They're dropped (see "Old code" below).
- Predicting stock returns from disclosure changes (the "Lazy Prices" idea). It needs price data and a ticker mapping this project doesn't have; a possible follow-up, not this rework.
- Covering filings after 2020.

**Anti-goals** (failure even if it technically works)
- A metric that isn't validated, or a number shown without its baseline.
- Anything illustrative, simulated, or hand-typed presented as a result.
- Claims of cause where the design only supports association.
- Cherry-picked example sentences; examples are drawn by a stated rule.

## Data

**Have:** `data/processed/risk_factors_2006_2020.parquet`, 72,613 filings from 13,970 companies with the full Item 1A text (`section_1A`), `cik`, `filing_year`, `filename` (`{cik}_{year}.htm`), `item_1a_length`. One filing per company-year. 47,643 filings (66%) have the same company's previous-year filing present; 6,413 companies have five or more years.

**Need from SEC EDGAR** (free; SEC's fair-access policy requires a User-Agent with a contact and at most 10 requests a second; the contact is Joe's, read from `SEC_USER_AGENT` in the git-ignored `.env`, never written into code; cache everything under `data/raw/edgar/`, git-ignored):
- **Filing dates:** the quarterly form indexes (`full-index/{year}/QTR{n}/form.idx`, about 60 files) give each 10-K's filing date. Join on CIK and year; resolve 10-K/A and multiple-filing cases by taking the original 10-K.
- **Industry:** SIC codes per CIK, from the SEC submissions data. Use SIC divisions (about 10 groups) as the industry level.

**Unit of analysis: the sentence.** The text has newlines and table remnants; a naive splitter finds about 177 sentences per section (median 148 characters). Scale is roughly 20–25 million sentences.
- Pick a splitter by checking candidates against 50 hand-reviewed sections.
- Drop debris by rule (too few letters, mostly digits, table rows) and report how much was dropped.
- Keep a sentence's position in the section. Flag risk-factor headings if they can be detected reliably; if not, say so.

## Layer 1: measure

**Normalization:** lowercase, collapse whitespace, replace digits with `0`, strip punctuation. Company names stay (removing them reliably is its own project); the effect on peer-sharing is noted as a limitation.

**Sentence labels**, for each sentence in a filing with a previous-year filing from the same company:
- **Carried over, verbatim:** its normalized form appears in the previous filing, or a near-verbatim match does (word 5-shingle Jaccard ≥ τ_v via MinHash).
- **Carried over, paraphrased:** no near-verbatim match, but its best embedding match in the previous filing has cosine ≥ τ_p.
- **New:** neither.
- Previous-filing sentences with no match in the current filing are **removed**.

**Shared with peers:** a normalized sentence (or near-verbatim match) that appears in filings of at least k other companies in the same year. k is set from the distribution, and results are shown for a few values.

**Filing-level metrics:** shares of sentences carried verbatim, paraphrased, new, and removed, plus the share shared with peers. Sentence-weighted, with character-weighted as a robustness check.

**Staged for scale:** exact hashing of normalized sentences first (cheap and catches most carryover); MinHash only on what's left; embeddings only on what's still unmatched within each company's consecutive pair. The embedding model must not truncate sentences (a sentence-level model on sentences avoids the first version's 256-token problem).

**Validation** (no human labels, by Joe's choice; the page says so)
- **Known-answer test set.** Pairs built from real sentences so the right label is known by construction:
  - *verbatim:* the same sentence with only names, dates, and numbers changed;
  - *paraphrase:* an LLM rewrite told to keep the meaning and change the wording, kept only if a second model confirms the meaning survived;
  - *different, easy:* a random sentence from another company and year;
  - *different, hard:* an LLM edit that keeps the topic and most of the words but changes the claim (reverses it, or names a different risk), which tests that the matcher doesn't mistake shared words for shared meaning.
- **Real pairs, two judges.** About 2,000 real candidate pairs, sampled across similarity bands, labeled same / paraphrase / different with a versioned rubric:
  - *judge 1:* Claude Haiku 5.5, called from the pipeline through Claude Code's headless mode (`claude -p --model haiku --output-format json`, 25–50 pairs per call), on all pairs; it also generates the known-answer rewrites and edits. This runs on Joe's Claude plan, so it costs usage, not API credits. Prompts and raw responses are saved, so the labels are reproducible as data even without rerunning;
  - *judge 2:* Claude, in an interactive Claude Code session, on a stratified subset of about 400 of the same pairs, blind to judge 1's labels; its labels are committed as data, like human annotations would be.
  Report their agreement (Cohen's kappa) on the shared subset and the share they disagree on. If agreement is good, judge 1's labels serve as the real-data check for all 2,000; if not, say so and fall back to the subset.
- **Thresholds:** τ_v and τ_p are tuned on half of the known-answer set and reported with precision and recall on the other half and on the agreed real pairs.
- **Stated limits:** constructed cases cover only what we thought to construct, and two LLM judges can share blind spots.
- **Baselines:** random sentence pairs from different companies and years (the false-match rate), word-shuffled sentences (shows what the embedding responds to; this re-tests the audit's finding), and the same company's consecutive years (the signal).

## Layer 2: explain

**Survival of risk sentences.** Link each sentence to its match in the next year's filing to form chains. A chain starts when a sentence first appears and ends when it's removed; it's right-censored at the company's last filing or at 2020. A missing year breaks the chain and counts as censoring (only consecutive-year links). Report:
- Kaplan–Meier curves overall, and for peer-shared vs. company-specific sentences.
- A hazard model of removal with a company-level random effect (frailty). Covariates: peer-shared, sentence length, position in section, SIC division, section length, year added. The proportional-hazards assumption is checked; if it fails, a discrete-time hazard model is used instead. If the full set of chains is too slow to fit with a random effect, fit on a stratified sample of companies and say so.

**What drives rewriting.** A hierarchical model of each filing's share of new sentences, with random effects for company, SIC division, and year, reported as a variance decomposition (how much of rewriting is the company, the industry, the year) with intervals.

## Layer 3: test (2020)

**Question:** when COVID hit, did companies rewrite, or append?

**Exposure:** filing date. 10-Ks filed after mid-March 2020 were written during the pandemic; those filed earlier mostly weren't. Filing month follows from fiscal-year end and filer status, so it isn't random. A manipulation check counts COVID terms per filing.

**Design:** difference-in-differences by filing month. Compare 2020 filings filed after the shock with those filed before, relative to the same calendar months in 2017–2019, which absorbs the stable differences between early and late filers. Outcomes: share new, share removed, change in section length. "Append" means new sentences rise while removals don't; "rewrite" means both rise.

**Checks:** an event-study plot by filing month; placebo shocks in 2018 and 2019; results with and without companies whose fiscal year changed. Limits are stated plainly on the page.

## Deliverables

- **Pipeline** in `pipeline/` with one entry point that runs every stage in order and caches intermediate parquet files under `data/interim/` (git-ignored). Stages: EDGAR metadata, sentences, matching, validation, survival, hierarchical model, 2020 analysis, report data. The report data is one JSON file, committed, that `src/generate_report.py` reads.
- **uv project:** `pyproject.toml` plus `uv.lock`; Python 3.12.
- **Tests:** pytest on small fixtures for normalization, splitting rules, matching, chain-building, and the label pipeline.
- **Report:** new sections in `src/report_template.html` for each layer plus a methods and validation section. Charts follow the same rules as the current page (validated palette, tooltips, table views, light and dark). The "Under review" note is replaced once results hold. The current descriptive figures (sections drawn as pages, length by year) stay as context in the methods section.
- **README** rewritten around the finding.
- **Old code:** tag the current state `v1-original`, then delete the classifier, DistilBERT, SHAP, and BERTopic code and the course outline PDF. The README links the tag.

## Constraints

- A Mac laptop, no GPU. Budget hours of compute, not days; staged matching keeps embeddings to the residual.
- SEC fair-access rules for every download.
- No hand-labeling (Joe's call); Joe reviews. No API credits: LLM calls go through `claude -p` on Joe's Claude plan (about 100–150 batched calls), so they count against its usage limits. The labeling step should be resumable, skipping batches it already has, in case a limit is hit mid-run.
- Raw and interim data stay out of git.

## Decisions

**Joe's:** the project is a portfolio piece built on real outcomes; measurement and causal inference at the core; new techniques welcome where they fit, to show growth; the three-layer shape (measure, explain, test); no hand-labeling; his email as the SEC contact; no paid API credits (LLM work runs on his Claude plan).

**My calls** (Joe can override):
- The central question's wording, and making year-over-year carryover primary with peer-sharing second.
- Survival analysis, the hierarchical variance decomposition, the 2020 difference-in-differences, and validating with a known-answer set plus two LLM judges in place of human labels.
- Sentences as the unit; consecutive-year links only; company names kept in normalization.
- SIC divisions as the industry level.
- Deleting the old modeling code after tagging `v1-original`.
- Python 3.12 with uv; parquet intermediates; one JSON for the report.
- Fitting the hazard model on a company sample if needed; keeping the current descriptive figures as context.

**Left to the builder:** the sentence splitter (chosen by the 50-section check), MinHash parameters, the embedding model (any sentence-level model that doesn't truncate typical sentences), the hazard-model library, the labeling page's design, the debris rules, and the exact stage boundaries.

## Milestones (each becomes a child ticket of #1)

1. **Data foundation:** uv project, EDGAR filing dates and SIC codes, validated sentence splitting, the sentence table.
2. **Measurement:** matching stages, the known-answer set, two-judge labels and agreement, thresholds, baselines, filing-level metrics.
3. **Explain:** sentence chains, survival curves and hazard model, variance decomposition.
4. **Test:** the 2020 difference-in-differences with event study and placebos.
5. **Publish:** report sections, README, old-code cleanup.
