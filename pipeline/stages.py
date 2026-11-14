"""Pipeline stages. Each builds one cached output and skips the work if it exists."""

import json
from concurrent.futures import ProcessPoolExecutor
from itertools import batched
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import Config
from pipeline.edgar import fetch
from pipeline.filings import match_filing_dates, read_ten_k_filings
from pipeline.industry import SUBMISSIONS_URL, read_sic_codes, sic_division
from pipeline.io import write_parquet
from pipeline.sentences import is_debris, segment, sentence_frame

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

    write_parquet(matched[FILINGS_COLUMNS], out)
    return out


STATS_KEY = b"sentence_stats"
FILINGS_PER_TASK = 50


def _sentences_for_filings(rows: list[tuple[int, int, str]]) -> tuple[pd.DataFrame, int, int]:
    """Worker: the sentence rows for a batch of filings, plus how many segments
    were found and how many of those were dropped as debris."""
    frames, n_segments, n_debris = [], 0, 0
    for cik, filing_year, text in rows:
        segments = segment(text)
        kept = [s for s in segments if not is_debris(s.text)]
        n_segments += len(segments)
        n_debris += len(segments) - len(kept)
        frames.append(sentence_frame(cik, filing_year, kept))
    return pd.concat(frames, ignore_index=True), n_segments, n_debris


def build_sentences(cfg: Config, workers: int = 8, force: bool = False) -> Path:
    """One row per sentence, written as one parquet file per corpus year.

    A year is read, split and written before the next one starts, and each
    batch of filings is written as it finishes, so memory stays bounded. Years
    already written are skipped unless `force`. Each file records in its
    metadata how many filings and segments went in and how much was debris.
    """
    out_dir = cfg.interim_dir / "sentences"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = cfg.root / CORPUS
    years = sorted(set(pq.read_table(corpus, columns=["filing_year"])["filing_year"].to_pylist()))

    with ProcessPoolExecutor(max_workers=workers) if workers > 1 else NoPool() as pool:
        for year in years:
            out = out_dir / f"year={year}.parquet"
            if out.exists() and not force:
                continue
            filings = pq.read_table(
                corpus,
                columns=["cik", "filing_year", "section_1A"],
                filters=[("filing_year", "=", year)],
            ).to_pandas()
            filings["cik"] = filings["cik"].astype("int64")
            filings = filings.sort_values("cik", kind="stable")
            rows = zip(filings["cik"], filings["filing_year"], filings["section_1A"], strict=True)

            stats = {"filings": len(filings), "segments": 0, "debris": 0, "sentences": 0}
            part = out.with_name(out.name + ".part")
            writer = None
            try:
                for frame, n_segments, n_debris in pool.map(
                    _sentences_for_filings, batched(rows, FILINGS_PER_TASK)
                ):
                    stats["segments"] += n_segments
                    stats["debris"] += n_debris
                    stats["sentences"] += len(frame)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    writer = writer or pq.ParquetWriter(part, table.schema, compression="zstd")
                    writer.write_table(table)
                if writer:
                    writer.add_key_value_metadata({STATS_KEY: json.dumps(stats)})
            finally:
                if writer:
                    writer.close()
            # An interrupted year leaves only the .part file, so a rerun redoes it.
            part.replace(out)
    return out_dir


class NoPool:
    """`workers=1`: run in this process, with the same interface as the pool."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable):
        return map(fn, iterable)


def sentence_stats(sentences_dir: Path) -> pd.DataFrame:
    """Per-year counts recorded when each sentence file was written."""
    stats = {}
    for path in sorted(sentences_dir.glob("year=*.parquet")):
        year = int(path.stem.split("=")[1])
        stats[year] = json.loads(pq.read_metadata(path).metadata[STATS_KEY])
    return pd.DataFrame.from_dict(stats, orient="index").rename_axis("filing_year")
