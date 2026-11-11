"""Entry point: `python -m pipeline.run filings [--force]`."""

import argparse

import pandas as pd

from pipeline.config import load_config
from pipeline.industry import SIC_DIVISIONS
from pipeline.stages import build_filings


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description=__doc__)
    parser.add_argument("stage", choices=["filings"], help="which stage to build")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)

    cfg = load_config()
    path = build_filings(cfg, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}")
    print(summarize_filings(pd.read_parquet(path)))


if __name__ == "__main__":
    main()
