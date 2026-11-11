# Data foundation

What milestone 1 builds, the checks behind it, and the decisions those checks forced. Every number here was printed by the command named next to it; none is typed in by hand. Raw and interim data stay out of git, so the commands are the record.

```
uv run python -m pipeline.run filings      # data/interim/filings.parquet
uv run python -m scripts.check_filing_dates
```

## The corpus

`data/processed/risk_factors_2006_2020.parquet` holds 72,613 Item 1A sections from 13,970 companies, one per company and year. 47,643 of them (65.6%) have the same company's previous year present, which is what year-over-year comparison needs, and 6,413 companies have five or more years.

## Filing dates

Each section needs the date its 10-K was filed, because the 2020 analysis compares reports filed before and after the pandemic began.

### The corpus year is the year reported on, not the year filed

The corpus labels each section with a year and doesn't say which year it means. The first version of this stage assumed it was the year filed and joined each section to the company's 10-K filed in that calendar year. It dated 92.1% of sections, a rate that looks like a join with a few gaps.

It was mostly wrong. Most companies file every year, so joining on the wrong year still finds a 10-K: the previous one. A match rate can't see that, so the check uses the text instead. COVID-19 was named on 2020-02-11. A section filed before that day can't contain the word, and one filed from April 2020 on almost always does.

| Corpus year matched to | Sections dated | Dated before 2020-02-11 yet mentioning COVID | Sections dated January 2020 that mention COVID |
|---|---|---|---|
| Year filed (first attempt) | 92.1% | 1,107 of 3,989 | 81.5% |
| Period of report (used) | 100.0% | 8 of 893 | 0.0% |

The corpus year is the calendar year in which the report's fiscal period ended. A report on 2019 is usually filed in early 2020, so the first join attached most sections to the filing from a year earlier. Matched on the period of report, the share of sections mentioning COVID traces the pandemic's arrival month by month, which is also the exposure measure the 2020 analysis will use:

| Month filed | Sections | Mention COVID |
|---|---|---|
| 2019-12 | 91 | 1.1% |
| 2020-01 | 76 | 0.0% |
| 2020-02 | 1,409 | 7.5% |
| 2020-03 | 1,599 | 48.6% |
| 2020-04 | 158 | 87.3% |
| 2020-05 | 178 | 95.5% |
| 2021-02 | 1,312 | 98.6% |
| 2021-03 | 1,912 | 95.5% |

All 8 remaining impossible mentions are sections with more than one candidate filing (below). Among sections with a unique match there are none.

### How filings are matched

Filing dates come from SEC's bulk submissions archive (`submissions.zip`), which lists every filing a company has made with its form type, date filed, period of report, and accession number. A section for company *c* and year *y* matches *c*'s 10-K-family filings whose period of report ends in *y*.

| Match | Sections | Share |
|---|---|---|
| `unique`: one original 10-K | 71,844 | 98.9% |
| `multiple`: more than one original; the earliest is used | 769 | 1.1% |
| `amendment_only` | 0 | 0.0% |
| `none` | 0 | 0.0% |

`multiple` has three causes: a period of report mis-entered as the filing date, a 52- or 53-week fiscal year that ends twice in one calendar year (1 January and 31 December), and transition reports after a change of fiscal year. Nothing in the metadata says which of the candidates the corpus kept, so these rows are flagged and the date-sensitive analysis in milestone 4 will run on `unique` rows, with `multiple` rows added as a robustness check.

Days from the end of the period to the filing date: median 72, 5th to 95th percentile 47 to 107. 690 sections (1.0%) were filed more than a year after their period ended (late filers), and none before it.

### What this changes in the design

- The corpus runs through reports on fiscal 2020. 87.1% of those were filed on or after 2020-11-09, when the SEC's amended risk-factor rules took effect, so the last corpus year sits under the new rules instead of ending just before them.
- The COVID shock lands on reports for fiscal 2019 (corpus year 2019): 21.8% were filed before the disease had a name and 33.5% on or after 2020-03-11, when the WHO declared a pandemic. That within-year split is what the difference-in-differences needs.
- The table keeps each filing's period of report (so fiscal-year-end month is known; 74.0% end in December) and its accession number (so any example sentence can link to its filing on EDGAR).

## Industry

SIC codes come from the same archive, grouped into SIC divisions. 98.5% of sections have a division.

| Division | Sections |
|---|---|
| A Agriculture, Forestry, Fishing | 289 |
| B Mining | 4,854 |
| C Construction | 753 |
| D Manufacturing | 24,677 |
| E Transportation, Communications, Utilities | 6,421 |
| F Wholesale Trade | 1,872 |
| G Retail Trade | 3,549 |
| H Finance, Insurance, Real Estate | 17,642 |
| I Services | 11,396 |
| K Nonclassifiable | 51 |
| (no code) | 1,109 |

The archive holds each company's current SIC code, not its code at the time of each filing. A company that changed industry is classified by where it ended up.

## Sentences

The unit of analysis is the sentence. Splitting matters more than it looks: a missed boundary glues two sentences into one that matches nothing, and a false one turns a sentence into fragments.

```
uv run python -m scripts.compare_splitters             # the comparison below
uv run python -m scripts.compare_splitters --sample 80  # plus the disagreements to review
```

### What the text looks like

The sections are plain text extracted from HTML. A newline usually ends a paragraph, a heading, or a list item, but not always: in a 600-section sample, 25% of lines don't end in sentence punctuation. Most of those are bullet markers, list items and headings, and about 2% of all lines break in the middle of a sentence (the next line starts in lower case). So lines are stitched before sentences are split: a line that plainly continues is joined to the one before it, a bullet glyph on its own line marks the next line as a list item, and a short line with no end punctuation followed by a new paragraph is kept whole as a heading.

### Choosing the splitter

Three candidates, on the 16 test cases in `tests/test_sentences.py` and on 50 sections sampled with `random_state=0` (3.6 MB of text):

| Splitter | Cases passed | Sentences per section | Over 1,500 characters | Under 40 characters | MB/s | Full corpus, 8 workers |
|---|---|---|---|---|---|---|
| Naive regex (split after `.!?` before a capital) | 11/16 | 416.7 | 0.00% | 8.2% | 42.00 | 11 s |
| pysbd 0.3.4 | 12/16 | 416.7 | 0.00% | 7.8% | 0.42 | 18 min |
| `pipeline.sentences` (used) | 16/16 | 388.4 | 0.01% | 2.8% | 30.17 | 15 s |

The case count alone would be a rigged contest: all three pass the 8 cases the plan started with, and the other 8 were added from failures seen in the real text while this splitter was being written, so it passes them by construction. The fair test is where the two serious candidates disagree on text neither was tuned on.

Run on the same stitched paragraphs, pysbd and `pipeline.sentences` split 98.1% of 7,737 paragraphs identically. They differ at 298 boundaries: 282 that only pysbd makes and 16 that only this splitter makes. A random 80 of those (seed 0) were read and judged one by one; the judgments are in [`docs/reviews/splitter_disagreements.csv`](reviews/splitter_disagreements.csv):

| Who was right | Disagreements |
|---|---|
| `pipeline.sentences` | 73 |
| pysbd | 4 |
| Either reading is fine | 3 |

Nearly all of pysbd's extra boundaries fall inside a sentence: before an enumerated clause ("...the earlier of ‖ (i) the completion..."), after a cross-reference ("see Item 7. ‖ Management's Discussion..."), or after an abbreviation ("A.M. ‖ Best ratings"). This project keeps a sentence with its enumerated clauses whole on purpose, because a sentence is the unit a company keeps, edits or drops. The four pysbd wins are real misses here: two sentences ending in "Ltd." before "Mr. Williams" (a word this splitter doesn't treat as a sentence opener), and two places where the source text itself is garbled around the period.

Who judged: Claude, during the build session, reading each context. Joe didn't hand-label them. The file is committed so anyone can disagree with a specific row.

### Debris and normalization

A segment is dropped as debris when it has fewer than 15 letters or when letters are under half of its non-space characters (table rows, page numbers, stray list markers). Sentences over 1,500 characters are kept and flagged `is_long`.

For matching, each sentence also gets a normalized form: Unicode NFKC, lower case, every digit replaced by `0`, every other character outside `a-z` replaced by a space, whitespace collapsed. "Revenue fell 12.5% in 2019." and "Revenue fell 14.0% in 2020!" normalize to the same string, so a sentence whose only change is its numbers counts as carried over. `norm_hash` is the first 8 bytes of the BLAKE2b hash of that string, as a signed 64-bit integer.

### The sentence table

```
uv run python -m pipeline.run sentences --workers 8   # data/interim/sentences/year=YYYY.parquet
uv run python -m scripts.review_sentences             # the sample that was read
```

| | |
|---|---|
| Sentences | 21,348,790, in 15 yearly files (2.5 GB on disk, zstd) |
| Build time | 198 s on 8 cores |
| Filings with no sentences | 0 of 72,613 |
| Sentences per filing | median 243; 10th percentile 86, 90th 569 |
| Characters per sentence | median 161; 10th percentile 71, 90th 302 |
| Dropped as debris | 227,644 of 21,576,434 segments (1.1%) |
| Flagged long (over 1,500 characters) | 1,585 (0.01%) |
| Over 600 characters | 101,718 (0.48%) |
| Headings | 321,192 (1.5%) |

Columns: `cik`, `filing_year`, `sent_idx` (position among kept sentences), `para_idx` (position of its paragraph, so a section can be rebuilt with its paragraph breaks), `text`, `norm`, `norm_hash`, `n_chars`, `is_long`, `is_heading`. Each yearly file also records, in its metadata, how many filings and segments went in and how many segments were debris.

### Reading the output

The first 20 sentences of 50 filings (sampled with `random_state=0`) were read, 1,000 sentences in all. The per-filing notes are in [`docs/reviews/sentence_review.csv`](reviews/sentence_review.csv).

- **Boundary errors: 2**, each leaving two fragments where there should be one sentence (4 of 1,000 sentences, 0.4%). One is a line wrapped before a capitalized word ("...under the Senior" / "Secured Notes due 2018..."), which the stitching rules leave alone because a capital after a line break usually does start something new. The other is "Yahoo! Inc."
- **Fixed during the review: 1.** A sentence ending "...in Item 8." was being glued to the next one. A reference to an item now runs on only when the item's title follows it ("Item 7. Management's Discussion..."). The table above is from the rebuild after that fix, and the filing was reread.
- **Limit of this check.** Each sentence was read as its first 58 and last 32 characters, which is where splitting errors show. A missed boundary in the middle of a long sentence would not be seen this way; the 0.48% of sentences over 600 characters is the upper bound on where those could hide.

Who read them: Claude, during the build session. Joe didn't hand-check them.

### Headings

`is_heading` is kept, with a narrower meaning than the plan hoped for. It marks a line that has no end punctuation and isn't a list item: section headings ("Risks Related to Our Business") and captions written without a full stop. In the 50 filings every flagged line was a heading (no wrong flags), so the column is safe to use for setting those lines aside.

It does not find risk-factor captions that end in a period, which is how most filers write them ("We depend on a small number of customers." on a line of its own, followed by the paragraph that explains it). Those are ordinary sentences here. In the 2015 file, 55.8% of paragraphs hold a single sentence, so "a one-sentence paragraph" is a usable signal for captions if a later milestone needs one, and `para_idx` is there to compute it.
