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
