"""Print the first sentences of 50 sampled filings, for reading against the source.

Each line shows a sentence's flags (H heading, L long), its length, and its
first and last characters, which is where splitting errors show. The reviewed
result is docs/reviews/sentence_review.csv.

Run from the repo root: uv run python -m scripts.review_sentences [--filings 50] [--sentences 20]
"""

import argparse

import pandas as pd

from pipeline.config import REPO_ROOT

HEAD_CHARS, TAIL_CHARS = 58, 32


def clip(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS + 3:
        return text
    return f"{text[:HEAD_CHARS]} … {text[-TAIL_CHARS:]}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filings", type=int, default=50)
    parser.add_argument("--sentences", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0, help="skip this many sampled filings")
    args = parser.parse_args(argv)

    filings = pd.read_parquet(REPO_ROOT / "data/interim/filings.parquet")
    sample = filings.sample(args.filings, random_state=0)[["cik", "filing_year"]]
    for number, (cik, year) in enumerate(sample.itertuples(index=False), start=1):
        if number <= args.skip:
            continue
        sentences = pd.read_parquet(
            REPO_ROOT / f"data/interim/sentences/year={year}.parquet",
            filters=[("cik", "=", cik), ("sent_idx", "<", args.sentences)],
            columns=["sent_idx", "para_idx", "text", "n_chars", "is_long", "is_heading"],
        )
        print(f"\n#{number} cik={cik} year={year}")
        for row in sentences.itertuples(index=False):
            flags = ("H" if row.is_heading else "-") + ("L" if row.is_long else "-")
            print(f"{row.sent_idx:>2} p{row.para_idx:<3}{flags} {row.n_chars:>4} {clip(row.text)}")


if __name__ == "__main__":
    main()
