import json
import zipfile

import pandas as pd

from pipeline.filings import match_filing_dates, read_ten_k_filings


def ten_ks(rows):
    """10-K filings as read_ten_k_filings returns them, from
    (form_type, cik, date_filed, report_date) tuples."""
    return pd.DataFrame(
        {
            "cik": [r[1] for r in rows],
            "form_type": [r[0] for r in rows],
            "date_filed": pd.to_datetime([r[2] for r in rows]),
            "report_date": pd.to_datetime([r[3] for r in rows]),
            "accession": [f"{r[1]:010d}-00-{i:06d}" for i, r in enumerate(rows)],
        }
    )


def filing_block(rows):
    """One block of SEC's submissions JSON: parallel arrays, one entry per filing."""
    return {
        "form": [r[0] for r in rows],
        "filingDate": [r[1] for r in rows],
        "reportDate": [r[2] for r in rows],
        "accessionNumber": [r[3] for r in rows],
    }


def test_reads_10k_family_from_recent_and_paged_history(tmp_path):
    zip_path = tmp_path / "submissions.zip"
    recent = filing_block(
        [
            ("10-K", "2020-10-30", "2020-09-26", "0000320193-20-000096"),
            ("8-K", "2020-10-29", "2020-10-29", "0000320193-20-000094"),
            ("10-K/A", "2019-12-02", "2019-09-28", "0000320193-19-000130"),
            ("10-KT", "2019-02-01", "", "0000320193-19-000001"),
        ]
    )
    older = filing_block([("10-K405", "2001-12-21", "2001-09-29", "0000912057-01-544436")])
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "CIK0000320193.json",
            json.dumps(
                {
                    "filings": {
                        "recent": recent,
                        "files": [{"name": "CIK0000320193-submissions-001.json"}],
                    }
                }
            ),
        )
        zf.writestr("CIK0000320193-submissions-001.json", json.dumps(older))
        zf.writestr(
            "CIK0001018724.json",
            json.dumps(
                {
                    "filings": {
                        "recent": filing_block(
                            [("10-K", "2015-01-30", "2014-12-31", "0001018724-15-000006")]
                        ),
                        "files": [],
                    }
                }
            ),
        )

    df = read_ten_k_filings(zip_path, [320193, 999])  # 999 has no member; 1018724 isn't asked for

    assert list(df.columns) == ["cik", "form_type", "date_filed", "report_date", "accession"]
    assert df["cik"].dtype == "int64"
    assert set(df["cik"]) == {320193}
    assert sorted(df["form_type"]) == ["10-K", "10-K/A", "10-K405", "10-KT"]
    apple_2020 = df[df["accession"] == "0000320193-20-000096"].iloc[0]
    assert apple_2020.date_filed == pd.Timestamp("2020-10-30")
    assert apple_2020.report_date == pd.Timestamp("2020-09-26")
    # A blank reportDate is kept as missing, not dropped and not an error.
    assert pd.isna(df[df["form_type"] == "10-KT"].iloc[0].report_date)


def test_string_and_int_ciks_join():
    filings = pd.DataFrame({"cik": ["0000320193"], "filing_year": [2015]})
    out = match_filing_dates(filings, ten_ks([("10-K", 320193, "2015-10-28", "2015-09-26")]))
    assert out.loc[0, "match_status"] == "unique"
    assert out.loc[0, "date_filed"] == pd.Timestamp("2015-10-28")
    assert out.loc[0, "report_date"] == pd.Timestamp("2015-09-26")
    assert out.loc[0, "accession"] == "0000320193-00-000000"
    assert out.loc[0, "n_candidates"] == 1
    assert out["cik"].dtype == "int64"


def test_corpus_year_is_the_period_of_report_not_the_year_filed():
    # A December year-end company: the report for 2019 is filed in February 2020.
    filings = pd.DataFrame({"cik": ["1606163"], "filing_year": [2019]})
    index = ten_ks(
        [
            ("10-K", 1606163, "2019-02-21", "2018-12-31"),
            ("10-K", 1606163, "2020-02-20", "2019-12-31"),
        ]
    )
    row = match_filing_dates(filings, index).iloc[0]
    assert row.date_filed == pd.Timestamp("2020-02-20")
    assert row.match_status == "unique"


def test_year_of_date_filed_reproduces_the_rejected_calendar_year_join():
    filings = pd.DataFrame({"cik": ["1606163"], "filing_year": [2019]})
    index = ten_ks(
        [
            ("10-K", 1606163, "2019-02-21", "2018-12-31"),
            ("10-K", 1606163, "2020-02-20", "2019-12-31"),
        ]
    )
    row = match_filing_dates(filings, index, year_of="date_filed").iloc[0]
    assert row.date_filed == pd.Timestamp("2019-02-21")


def test_earliest_original_wins_over_amendment():
    # Same report year: an original, its amendment, and a transition report.
    filings = pd.DataFrame({"cik": ["1157377"], "filing_year": [2015]})
    index = ten_ks(
        [
            ("10-KT", 1157377, "2016-02-26", "2015-12-31"),
            ("10-K/A", 1157377, "2015-10-15", "2015-06-30"),
            ("10-K", 1157377, "2015-09-02", "2015-06-30"),
        ]
    )
    row = match_filing_dates(filings, index).iloc[0]
    assert row.date_filed == pd.Timestamp("2015-09-02")
    assert row.form_type == "10-K"
    assert row.match_status == "multiple" and row.n_candidates == 2


def test_amendment_only_is_dated_and_marked():
    filings = pd.DataFrame({"cik": ["1453122"], "filing_year": [2015]})
    index = ten_ks(
        [
            ("10-K/A", 1453122, "2016-06-01", "2015-12-31"),
            ("10-K/A", 1453122, "2016-04-18", "2015-12-31"),
        ]
    )
    row = match_filing_dates(filings, index).iloc[0]
    assert row.match_status == "amendment_only"
    assert row.date_filed == pd.Timestamp("2016-04-18")
    assert row.form_type == "10-K/A"
    assert row.n_candidates == 0


def test_no_match_is_kept_with_status_none():
    # Same company reported on 2014 only; a different company reported on 2015.
    # A filing with no period of report can't be matched to any year.
    filings = pd.DataFrame({"cik": ["1606163"], "filing_year": [2015]})
    index = ten_ks(
        [
            ("10-K", 1606163, "2015-03-10", "2014-12-31"),
            ("10-K", 999, "2016-03-10", "2015-12-31"),
            ("10-K", 1606163, "2016-03-11", None),
        ]
    )
    out = match_filing_dates(filings, index)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.match_status == "none"
    assert pd.isna(row.date_filed)
    assert pd.isna(row.accession)
    assert row.n_candidates == 0
