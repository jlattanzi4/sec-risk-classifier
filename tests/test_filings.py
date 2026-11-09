from pathlib import Path

import pandas as pd
import pytest

from pipeline.filings import index_urls, match_filing_dates, parse_form_index

FIXTURE = Path(__file__).parent / "fixtures" / "form_2015_q1.idx"


@pytest.fixture
def fixture_text():
    # Real lines from EDGAR's 2015 Q1 form.idx, plus one 10-K405 row from 2002 Q1
    # (SEC retired that form in 2003). Rows keep their trailing spaces.
    return FIXTURE.read_text(encoding="latin-1")


def index_frame(rows):
    """An index as parse_form_index returns it, from (form_type, cik, date) tuples."""
    return pd.DataFrame(
        {
            "form_type": [r[0] for r in rows],
            "cik": [r[1] for r in rows],
            "date_filed": pd.to_datetime([r[2] for r in rows]),
            "filename": [f"edgar/data/{r[1]}/{i}.txt" for i, r in enumerate(rows)],
        }
    )


def test_parses_fixed_width_rows_and_keeps_10k_family(fixture_text):
    df = parse_form_index(fixture_text)
    assert set(df.form_type) == {"10-K", "10-K/A", "10-KT", "10-K405"}
    assert df.cik.dtype == "int64"
    assert len(df) == 5
    amendment = df[df.form_type == "10-K/A"].iloc[0]
    assert amendment.cik == 1453122
    assert amendment.date_filed == pd.Timestamp("2015-02-18")
    assert amendment.filename == "edgar/data/1453122/0001554795-15-000063.txt"


def test_string_and_int_ciks_join():
    filings = pd.DataFrame({"cik": ["0000320193"], "filing_year": [2015]})
    index = pd.DataFrame(
        {
            "form_type": ["10-K"],
            "cik": [320193],
            "date_filed": pd.to_datetime(["2015-10-28"]),
            "filename": ["x"],
        }
    )
    out = match_filing_dates(filings, index)
    assert out.loc[0, "match_status"] == "unique"
    assert out.loc[0, "date_filed"] == pd.Timestamp("2015-10-28")
    assert out.loc[0, "n_candidates"] == 1


def test_earliest_original_wins_over_amendment():
    # original 2015-03-02, amendment 2015-04-15, second original (10-KT) 2015-09-30
    filings = pd.DataFrame({"cik": ["1157377"], "filing_year": [2015]})
    index = index_frame(
        [
            ("10-KT", 1157377, "2015-09-30"),
            ("10-K/A", 1157377, "2015-04-15"),
            ("10-K", 1157377, "2015-03-02"),
        ]
    )
    row = match_filing_dates(filings, index).iloc[0]
    assert row.date_filed == pd.Timestamp("2015-03-02")
    assert row.form_type == "10-K"
    assert row.match_status == "multiple" and row.n_candidates == 2


def test_amendment_only_is_dated_and_marked():
    filings = pd.DataFrame({"cik": ["1453122"], "filing_year": [2015]})
    index = index_frame([("10-K/A", 1453122, "2015-06-01"), ("10-K/A", 1453122, "2015-02-18")])
    row = match_filing_dates(filings, index).iloc[0]
    assert row.match_status == "amendment_only"
    assert row.date_filed == pd.Timestamp("2015-02-18")
    assert row.form_type == "10-K/A"
    assert row.n_candidates == 0


def test_no_match_is_kept_with_status_none():
    # Same company filed in 2014 only; a different company filed in 2015.
    filings = pd.DataFrame({"cik": ["1606163"], "filing_year": [2015]})
    index = index_frame([("10-K", 1606163, "2014-03-10"), ("10-K", 999, "2015-03-10")])
    out = match_filing_dates(filings, index)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.match_status == "none"
    assert pd.isna(row.date_filed)
    assert row.n_candidates == 0


def test_index_urls_cover_every_quarter():
    assert index_urls(range(2015, 2017)) == [
        f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        for y in (2015, 2016)
        for q in (1, 2, 3, 4)
    ]
