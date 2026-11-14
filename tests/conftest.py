"""Fixtures shared by the stage tests: a config rooted in a temporary directory, and small
filings and sentence tables written the way the real stages write them."""

import pandas as pd
import pytest

from pipeline.config import Config
from pipeline.io import write_parquet
from pipeline.sentences import norm_hash, normalize


@pytest.fixture
def cfg(tmp_path):
    return Config(
        root=tmp_path,
        raw_dir=tmp_path / "data/raw/edgar",
        interim_dir=tmp_path / "data/interim",
        sec_user_agent="Jane Doe jane@example.com",
    )


def write_tables(cfg: Config, filings: dict[tuple[int, int], list[str]]) -> None:
    """Write `filings.parquet` and the yearly sentence files for {(cik, year): sentences}."""
    rows = []
    for (cik, year), texts in filings.items():
        for i, text in enumerate(texts):
            norm = normalize(text)
            rows.append(
                {
                    "cik": cik,
                    "filing_year": year,
                    "sent_idx": i,
                    "para_idx": i,
                    "text": text,
                    "norm": norm,
                    "norm_hash": norm_hash(norm),
                    "n_chars": len(text),
                    "is_long": False,
                    "is_heading": not text.rstrip().endswith((".", ";", ":", "?", "!")),
                }
            )
    sentences = pd.DataFrame(rows)
    for year, part in sentences.groupby("filing_year"):
        write_parquet(part, cfg.interim_dir / "sentences" / f"year={year}.parquet")
    listing = pd.DataFrame(list(filings), columns=["cik", "filing_year"])
    write_parquet(listing, cfg.interim_dir / "filings.parquet")
