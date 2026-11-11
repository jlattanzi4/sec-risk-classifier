"""Entry point: `python -m pipeline.run STAGE [STAGE ...] [--force] [--workers N]`.

Stages run in the order given. `all` runs every stage in pipeline order.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.candidates import build_candidates, summarize_candidates
from pipeline.config import load_config
from pipeline.industry import SIC_DIVISIONS
from pipeline.known_answer import build_known_answer, summarize_known_answer
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
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], help="what to build")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--workers", type=int, default=8, help="processes for the parallel stages")
    args = parser.parse_args(argv)

    cfg = load_config()
    for stage in STAGES if "all" in args.stages else args.stages:
        start = time.perf_counter()
        STAGES[stage](cfg, args)
        print(f"[{stage}] done in {time.perf_counter() - start:,.0f} s\n")


def _filings(cfg, args) -> None:
    path = build_filings(cfg, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}")
    print(summarize_filings(pd.read_parquet(path)))


def _sentences(cfg, args) -> None:
    path = build_sentences(cfg, workers=args.workers, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}/")
    n_filings = pq.read_metadata(cfg.root / CORPUS).num_rows
    print(summarize_sentences(path, n_filings))


def _candidates(cfg, args) -> None:
    path = build_candidates(cfg, workers=args.workers, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}/")
    print(summarize_candidates(path))


def _known_answer(cfg, args) -> None:
    # Batches with a saved reply are read back, so this calls a model only for what is missing.
    path = build_known_answer(cfg)
    print(f"wrote {path.relative_to(cfg.root)}")
    print(summarize_known_answer(pd.read_csv(path)))


# In pipeline order: each stage reads what the ones before it wrote.
STAGES = {
    "filings": _filings,
    "sentences": _sentences,
    "candidates": _candidates,
    "known-answer": _known_answer,
}


if __name__ == "__main__":
    main()
