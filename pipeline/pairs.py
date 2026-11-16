"""Which filings can be compared with the same company's filing a year earlier."""

import pandas as pd


def consecutive_pairs(filings: pd.DataFrame) -> pd.DataFrame:
    """One row per filing whose company also filed for the previous fiscal year.

    Columns: `cik`, `filing_year`, `prev_year` (always `filing_year - 1`). A
    gap year breaks the link: 2019 is never paired with 2017.
    """
    have = filings[["cik", "filing_year"]].drop_duplicates()
    earlier = have.assign(filing_year=have["filing_year"] + 1)
    pairs = have.merge(earlier, on=["cik", "filing_year"])
    pairs["prev_year"] = pairs["filing_year"] - 1
    return pairs.sort_values(["cik", "filing_year"], kind="stable").reset_index(drop=True)
