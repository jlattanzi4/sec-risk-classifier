"""Filing dates for each 10-K, from EDGAR's quarterly form indexes."""

import numpy as np
import pandas as pd

TEN_K_FORMS: frozenset[str] = frozenset(
    {"10-K", "10-K405", "10-KT", "10-KSB", "10-KSB40", "10-KT405"}
)
INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"


def index_urls(years: range) -> list[str]:
    return [INDEX_URL.format(year=y, quarter=q) for y in years for q in (1, 2, 3, 4)]


def _is_ten_k_family(form_type: str) -> bool:
    return form_type.removesuffix("/A") in TEN_K_FORMS


def parse_form_index(text: str) -> pd.DataFrame:
    """Parse a form.idx file, keeping 10-K-family filings and their amendments.

    Rows are fixed-width: form type, company name, CIK, date filed, file name.
    The last three never contain spaces, so splitting from the right isolates
    them however long the company name is; the form type is the first token.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("---")) + 1
    rows = []
    for line in lines[start:]:
        if not line.startswith("10-K"):  # every 10-K-family form starts this way
            continue
        head, cik, date_filed, filename = line.rsplit(None, 3)
        form_type = head.split(None, 1)[0]
        if _is_ten_k_family(form_type):
            rows.append((form_type, int(cik), date_filed, filename))
    df = pd.DataFrame(rows, columns=["form_type", "cik", "date_filed", "filename"])
    df["cik"] = df["cik"].astype("int64")
    df["date_filed"] = pd.to_datetime(df["date_filed"], format="%Y-%m-%d").astype("datetime64[ns]")
    return df


def match_filing_dates(filings: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Date each filing by its company's 10-K filed in the same calendar year.

    The earliest original (non-amendment) filing wins; `n_candidates` counts the
    originals. A company-year with only amendments is dated from the earliest
    one. `match_status` is "unique", "multiple", "amendment_only" or "none".
    CIKs are compared as integers, so "0000320193" matches 320193.
    """
    keys = ["cik", "filing_year"]
    out = filings.astype({"cik": "int64", "filing_year": "int64"})
    idx = index.astype({"cik": "int64"}).sort_values("date_filed", kind="stable")
    idx["filing_year"] = idx["date_filed"].dt.year.astype("int64")
    is_amendment = idx["form_type"].str.endswith("/A")

    originals = (
        idx[~is_amendment]
        .groupby(keys, as_index=False)
        .agg(
            date_filed=("date_filed", "first"),
            form_type=("form_type", "first"),
            n_candidates=("form_type", "size"),
        )
    )
    amendments = (
        idx[is_amendment]
        .groupby(keys, as_index=False)
        .agg(amended_on=("date_filed", "first"), amended_form=("form_type", "first"))
    )
    out = out.merge(originals, on=keys, how="left").merge(amendments, on=keys, how="left")

    out["n_candidates"] = out["n_candidates"].fillna(0).astype("int64")
    amendment_only = (out["n_candidates"] == 0) & out["amended_on"].notna()
    out["match_status"] = np.select(
        [out["n_candidates"] == 1, out["n_candidates"] > 1, amendment_only],
        ["unique", "multiple", "amendment_only"],
        default="none",
    )
    out["date_filed"] = out["date_filed"].where(~amendment_only, out["amended_on"])
    out["form_type"] = out["form_type"].where(~amendment_only, out["amended_form"])
    return out.drop(columns=["amended_on", "amended_form"])
