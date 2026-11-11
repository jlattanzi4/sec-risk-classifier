"""Check that each filing date belongs to the right document, not just the right company.

Most companies file a 10-K every year, so a high match rate alone can't tell
whether the corpus year is the year a report was filed or the year it reports
on. Joining on the wrong one still matches about 92% of rows, each to the
previous report's date.

COVID-19 gives a clean test. The disease was named on 2020-02-11, so a risk
section filed before that day can't contain the word, and one filed from April
2020 on almost always does. This script dates the 2019 and 2020 corpus sections
both ways and prints, for each, the share that mention COVID by month filed and
the number of impossible mentions.

Run from the repo root: uv run python -m scripts.check_filing_dates
"""

import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import REPO_ROOT
from pipeline.filings import match_filing_dates, read_ten_k_filings
from pipeline.stages import CORPUS

COVID_NAMED = pd.Timestamp("2020-02-11")
SUBMISSIONS = REPO_ROOT / "data/raw/edgar/daily-index/bulkdata/submissions.zip"
RULES = {
    "year filed (rejected)": "date_filed",
    "period of report (used)": "report_date",
}


def main() -> None:
    corpus = pd.read_parquet(REPO_ROOT / CORPUS, columns=["cik", "filing_year"])
    corpus["cik"] = corpus["cik"].astype("int64")
    ten_ks = read_ten_k_filings(SUBMISSIONS, sorted(corpus["cik"].unique().tolist()))

    texts = pq.read_table(
        REPO_ROOT / CORPUS,
        columns=["cik", "filing_year", "section_1A"],
        filters=[("filing_year", "in", [2019, 2020])],
    ).to_pandas()
    texts["cik"] = texts["cik"].astype("int64")
    texts["mentions_covid"] = texts["section_1A"].str.contains("covid", case=False)
    texts = texts.drop(columns="section_1A")

    for label, year_of in RULES.items():
        matched = match_filing_dates(corpus, ten_ks, year_of=year_of)
        dated = matched["match_status"].isin(["unique", "multiple"])
        print(f"\n== Corpus year matched to the {label} ==")
        print(f"all years: dated by an original 10-K: {dated.mean():.1%} of {len(matched):,}")

        df = texts.merge(matched[dated], on=["cik", "filing_year"], validate="one_to_one")
        early = df["date_filed"] < COVID_NAMED
        impossible = early & df["mentions_covid"]
        print(
            f"2019-2020 sections dated: {len(df):,}; dated before {COVID_NAMED:%Y-%m-%d} "
            f"yet mentioning COVID: {impossible.sum():,} of {early.sum():,} "
            f"({(impossible & (df['match_status'] == 'unique')).sum():,} of them with a unique match)"
        )
        window = df[df["date_filed"].between("2019-10-01", "2021-06-30")]
        by_month = window.groupby(window["date_filed"].dt.strftime("%Y-%m")).agg(
            sections=("mentions_covid", "size"), mention_covid=("mentions_covid", "mean")
        )
        print(by_month.to_string(formatters={"mention_covid": "{:.1%}".format}))

    used = match_filing_dates(corpus, ten_ks)
    lag = (used["date_filed"] - used["report_date"]).dt.days
    print("\n== Days from period end to filing (period-of-report match) ==")
    print(lag.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(0).to_string())
    print(
        f"filed before the period ended: {(lag < 0).sum():,}; over a year later: {(lag > 365).sum():,}"
    )


if __name__ == "__main__":
    main()
