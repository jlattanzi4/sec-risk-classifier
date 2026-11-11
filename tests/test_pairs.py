import pandas as pd

from pipeline.pairs import consecutive_pairs


def filings(*rows):
    return pd.DataFrame(rows, columns=["cik", "filing_year"])


def test_a_filing_is_paired_with_the_year_before():
    pairs = consecutive_pairs(filings((1, 2018), (1, 2019)))
    assert pairs.to_dict("records") == [{"cik": 1, "filing_year": 2019, "prev_year": 2018}]


def test_a_gap_year_breaks_the_pair():
    assert consecutive_pairs(filings((1, 2017), (1, 2019))).empty


def test_pairs_are_within_a_company():
    assert consecutive_pairs(filings((1, 2018), (2, 2019))).empty


def test_a_run_of_years_gives_a_pair_for_each_but_the_first():
    pairs = consecutive_pairs(filings((1, 2016), (1, 2017), (1, 2018), (2, 2018)))
    assert pairs[["cik", "filing_year"]].values.tolist() == [[1, 2017], [1, 2018]]
