"""Pipeline stages. Each builds one cached output and skips the work if it exists."""

from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.edgar import fetch
from pipeline.filings import match_filing_dates, read_ten_k_filings
from pipeline.industry import SUBMISSIONS_URL, read_sic_codes, sic_division

CORPUS = Path("data/processed/risk_factors_2006_2020.parquet")
FILINGS_COLUMNS = [
    "cik",
    "filing_year",
    "item_1a_length",
    "report_date",
    "date_filed",
    "form_type",
    "accession",
    "n_candidates",
    "match_status",
    "sic",
    "sic_division",
]


def _cache_path(cfg: Config, url: str) -> Path:
    """Where a download lives: its path under EDGAR's Archives, inside raw_dir."""
    return cfg.raw_dir / url.split("/Archives/edgar/", 1)[1]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write through a temporary name so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    df.to_parquet(part, index=False, compression="zstd")
    part.replace(path)


def build_filings(cfg: Config, force: bool = False) -> Path:
    """One row per filing: when its 10-K was filed, the period it reports on, and its
    company's industry."""
    out = cfg.interim_dir / "filings.parquet"
    if out.exists() and not force:
        return out

    filings = pd.read_parquet(cfg.root / CORPUS, columns=["cik", "filing_year", "item_1a_length"])
    filings["cik"] = filings["cik"].astype("int64")
    ciks = sorted(filings["cik"].unique().tolist())

    # One archive answers both questions: when each 10-K was filed, and the
    # company's industry code.
    submissions = fetch(SUBMISSIONS_URL, _cache_path(cfg, SUBMISSIONS_URL), cfg.sec_user_agent)
    matched = match_filing_dates(filings, read_ten_k_filings(submissions, ciks))

    sic_codes = read_sic_codes(submissions, ciks)
    matched["sic"] = matched["cik"].map(sic_codes).astype("Int64")
    matched["sic_division"] = [
        None if pd.isna(code) else sic_division(int(code)) for code in matched["sic"]
    ]

    _write_parquet(matched[FILINGS_COLUMNS], out)
    return out
