"""Entry point: `python -m pipeline.run {filings,sentences,all} [--force] [--workers N]`."""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import load_config
from pipeline.industry import SIC_DIVISIONS
from pipeline.stages import CORPUS, build_filings, build_sentences, sentence_stats


def summarize_filings(df: pd.DataFrame) -> str:
    """The checks recorded in docs/data-foundation.md, as printable text."""
    status = df["match_status"].value_counts()
    divisions = df["sic_division"].value_counts(dropna=False).sort_index(na_position="last")
    lines = [f"filings: {len(df):,}", "match_status:"]
    lines += [f"  {s:<15}{n:>7,}  {n / len(df):6.1%}" for s, n in status.items()]
    dated = df["match_status"].isin(["unique", "multiple"]).mean()
    lines.append(f"dated by an original 10-K (unique or multiple): {dated:.1%}")
    lines.append(f"with a SIC division: {df['sic_division'].notna().mean():.1%}")
    lines.append("filings by SIC division:")
    for letter, n in divisions.items():
        name = SIC_DIVISIONS[letter][2] if isinstance(letter, str) else "(none)"
        lines.append(f"  {letter if isinstance(letter, str) else '-'}  {name:<42}{n:>7,}")
    return "\n".join(lines)


def summarize_sentences(sentences_dir: Path, n_filings: int) -> str:
    """The sentence-table checks recorded in docs/data-foundation.md, as printable text."""
    files = sorted(sentences_dir.glob("year=*.parquet"))
    stats = sentence_stats(sentences_dir).sum()
    df = pd.concat(
        pd.read_parquet(f, columns=["cik", "filing_year", "n_chars", "is_long", "is_heading"])
        for f in files
    )
    per_filing = df.groupby(["cik", "filing_year"]).size()
    megabytes = sum(f.stat().st_size for f in files) / 1e6
    n = len(df)
    return "\n".join(
        [
            f"sentences: {n:,} in {len(files)} year files ({megabytes:,.0f} MB on disk)",
            f"filings with no sentences: {n_filings - len(per_filing):,} of {n_filings:,}",
            "sentences per filing: median {:,.0f}, 10th percentile {:,.0f}, 90th {:,.0f}".format(
                *per_filing.quantile([0.5, 0.1, 0.9])
            ),
            "characters per sentence: median {:,.0f}, 10th percentile {:,.0f}, 90th {:,.0f}".format(
                *df["n_chars"].quantile([0.5, 0.1, 0.9])
            ),
            (
                f"dropped as debris: {stats['debris']:,} of {stats['segments']:,} segments "
                f"({stats['debris'] / stats['segments']:.1%})"
            ),
            f"flagged long: {df['is_long'].sum():,} ({df['is_long'].mean():.2%})",
            f"headings: {df['is_heading'].sum():,} ({df['is_heading'].mean():.1%})",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description=__doc__)
    parser.add_argument("stage", choices=["filings", "sentences", "all"], help="what to build")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--workers", type=int, default=8, help="processes for sentence splitting")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.stage in ("filings", "all"):
        path = build_filings(cfg, force=args.force)
        print(f"wrote {path.relative_to(cfg.root)}")
        print(summarize_filings(pd.read_parquet(path)))
    if args.stage in ("sentences", "all"):
        start = time.perf_counter()
        path = build_sentences(cfg, workers=args.workers, force=args.force)
        print(f"wrote {path.relative_to(cfg.root)}/ in {time.perf_counter() - start:,.0f} s")
        n_filings = pq.read_metadata(cfg.root / CORPUS).num_rows
        print(summarize_sentences(path, n_filings))


if __name__ == "__main__":
    main()
