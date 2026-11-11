"""Filing dates for each 10-K, from SEC's bulk submissions data.

The corpus labels each section with a company and a year. That year is the
calendar year the report's fiscal period ended in (its "period of report"),
not the year it was filed: a report on 2019 is usually filed in early 2020.
So filings are matched on the period of report. Matching on the year filed
looks plausible (most companies file every year, so 92% still match) but
attaches the previous report's date to most rows; scripts/check_filing_dates.py
shows the evidence.
"""

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TEN_K_FORMS: frozenset[str] = frozenset(
    {"10-K", "10-K405", "10-KT", "10-KSB", "10-KSB40", "10-KT405"}
)
TEN_K_COLUMNS = ["cik", "form_type", "date_filed", "report_date", "accession"]


def _is_ten_k_family(form_type: str) -> bool:
    return form_type.removesuffix("/A") in TEN_K_FORMS


def read_ten_k_filings(zip_path: Path, ciks: Iterable[int]) -> pd.DataFrame:
    """Every 10-K-family filing (and amendment) the given companies have made.

    A company's history is split across `CIK##########.json` (its most recent
    filings) and the paged files that record lists under `filings.files`; both
    are read, and only for the CIKs asked for. A blank period of report is kept
    as missing.
    """
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        for cik in ciks:
            name = f"CIK{cik:010d}.json"
            if name not in members:
                continue
            filings = json.loads(zf.read(name))["filings"]
            blocks = [filings["recent"]]
            blocks += [
                json.loads(zf.read(page["name"]))
                for page in filings.get("files", [])
                if page["name"] in members
            ]
            for block in blocks:
                for form, filed, period, accession in zip(
                    block["form"],
                    block["filingDate"],
                    block["reportDate"],
                    block["accessionNumber"],
                    strict=True,
                ):
                    if _is_ten_k_family(form):
                        rows.append((cik, form, filed, period or None, accession))
    df = pd.DataFrame(rows, columns=TEN_K_COLUMNS)
    df["cik"] = df["cik"].astype("int64")
    for column in ("date_filed", "report_date"):
        df[column] = pd.to_datetime(df[column], format="%Y-%m-%d").astype("datetime64[ns]")
    return df


def match_filing_dates(
    filings: pd.DataFrame,
    ten_ks: pd.DataFrame,
    *,
    year_of: Literal["report_date", "date_filed"] = "report_date",
) -> pd.DataFrame:
    """Attach each corpus filing's 10-K: date filed, period of report, form, accession.

    A corpus row (company, year) matches the company's 10-Ks whose `year_of`
    date falls in that year. The default, the period of report, is what the
    corpus year means; "date_filed" reruns the calendar-year join this replaced
    and exists only so the comparison stays reproducible.

    The earliest original (non-amendment) filing wins; `n_candidates` counts the
    originals. A company-year with only amendments is dated from the earliest
    one. `match_status` is "unique", "multiple", "amendment_only" or "none".
    CIKs are compared as integers, so "0000320193" matches 320193.
    """
    keys = ["cik", "filing_year"]
    out = filings.astype({"cik": "int64", "filing_year": "int64"})
    idx = ten_ks.astype({"cik": "int64"}).dropna(subset=[year_of])
    idx = idx.sort_values("date_filed", kind="stable")
    idx["filing_year"] = idx[year_of].dt.year.astype("int64")
    is_amendment = idx["form_type"].str.endswith("/A")

    originals = (
        idx[~is_amendment]
        .groupby(keys, as_index=False)
        .agg(
            date_filed=("date_filed", "first"),
            report_date=("report_date", "first"),
            form_type=("form_type", "first"),
            accession=("accession", "first"),
            n_candidates=("form_type", "size"),
        )
    )
    amendments = (
        idx[is_amendment]
        .groupby(keys, as_index=False)
        .agg(
            amended_on=("date_filed", "first"),
            amended_report_date=("report_date", "first"),
            amended_form=("form_type", "first"),
            amended_accession=("accession", "first"),
        )
    )
    out = out.merge(originals, on=keys, how="left").merge(amendments, on=keys, how="left")

    out["n_candidates"] = out["n_candidates"].fillna(0).astype("int64")
    amendment_only = (out["n_candidates"] == 0) & out["amended_on"].notna()
    out["match_status"] = np.select(
        [out["n_candidates"] == 1, out["n_candidates"] > 1, amendment_only],
        ["unique", "multiple", "amendment_only"],
        default="none",
    )
    for column, fallback in [
        ("date_filed", "amended_on"),
        ("report_date", "amended_report_date"),
        ("form_type", "amended_form"),
        ("accession", "amended_accession"),
    ]:
        out[column] = out[column].where(~amendment_only, out[fallback])
        out = out.drop(columns=fallback)
    return out
