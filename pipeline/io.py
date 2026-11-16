"""Reading and writing the pipeline's cached files."""

from pathlib import Path

import pandas as pd

from pipeline.config import Config


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write through a temporary name so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    df.to_parquet(part, index=False, compression="zstd")
    part.replace(path)


def read_sentences(cfg: Config, year: int, columns: list[str]) -> pd.DataFrame:
    """One fiscal year of the sentence table."""
    return pd.read_parquet(cfg.interim_dir / "sentences" / f"year={year}.parquet", columns=columns)


def sentence_years(cfg: Config) -> list[int]:
    """The fiscal years the sentence table covers."""
    files = (cfg.interim_dir / "sentences").glob("year=*.parquet")
    return sorted(int(f.stem.split("=")[1]) for f in files)
