"""Compare sentence splitters on the test cases and on 50 real sections.

Three candidates: a one-line regex (the naive baseline), pysbd (a rule-based
library built for this job), and this project's splitter in pipeline/sentences.py.
For each: test cases passed, sentences per section, the share of very long and
very short sentences (signs of missed and false boundaries), and speed.

It then puts pysbd and this project's splitter head to head on the same stitched
paragraphs and writes a fixed random sample of the places where only one of them
split, for review. The reviewed sample is docs/reviews/splitter_disagreements.csv.

Run from the repo root: uv run python -m scripts.compare_splitters [--sample N]
"""

import argparse
import random
import re
import sys
import time

import pandas as pd
import pysbd

from pipeline.config import REPO_ROOT
from pipeline.sentences import LONG_SENTENCE_CHARS, _paragraphs, segment, split_sentences
from pipeline.stages import CORPUS
from tests.test_sentences import SPLIT_CASES

SHORT_SENTENCE_CHARS = 40
CORPUS_GB = 3.6
WORKERS = 8

_NAIVE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")
_PYSBD = pysbd.Segmenter(language="en", clean=False)


def naive_regex(text: str) -> list[str]:
    return [s.strip() for s in _NAIVE.split(text) if s.strip()]


def pysbd_splitter(text: str) -> list[str]:
    return [s.strip() for line in text.split("\n") for s in _PYSBD.segment(line) if s.strip()]


SPLITTERS = {
    "naive regex": naive_regex,
    "pysbd": pysbd_splitter,
    "pipeline.sentences": split_sentences,
}


def _duration(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"


def _boundaries(sentences: list[str]) -> set[int]:
    """Where each sentence ends, counted in non-space characters so spacing can't differ."""
    ends, position = set(), 0
    for sentence in sentences:
        position += len(sentence.replace(" ", ""))
        ends.add(position)
    return ends


def _context(paragraph: str, boundary: int) -> str:
    """The text around a boundary, with a mark where the split falls."""
    seen = 0
    for i, char in enumerate(paragraph):
        seen += char != " "
        if seen == boundary:
            return f"{paragraph[max(0, i - 79) : i + 1]} ‖ {paragraph[i + 1 : i + 61].strip()}"
    raise ValueError("boundary past the end of the paragraph")


def head_to_head(sections: list[str], sample: int) -> None:
    """pysbd against this project's splitter, paragraph by paragraph."""
    paragraphs = identical = 0
    disagreements = []  # (which splitter split here, context)
    for section in sections:
        for paragraph, is_heading in _paragraphs(section):
            if is_heading:
                continue
            paragraphs += 1
            theirs = _boundaries([s.strip() for s in _PYSBD.segment(paragraph) if s.strip()])
            ours = _boundaries([s.text for s in segment(paragraph)])
            identical += theirs == ours
            disagreements += [("pysbd", _context(paragraph, b)) for b in sorted(theirs - ours)]
            disagreements += [("pipeline", _context(paragraph, b)) for b in sorted(ours - theirs)]
    only_theirs = sum(who == "pysbd" for who, _ in disagreements)
    print(
        f"\nhead to head on {paragraphs:,} paragraphs: identical on {identical:,} "
        f"({identical / paragraphs:.1%}); boundaries only pysbd makes: {only_theirs:,}; "
        f"only pipeline.sentences makes: {len(disagreements) - only_theirs:,}"
    )
    if sample:
        chosen = random.Random(0).sample(disagreements, min(sample, len(disagreements)))
        pd.DataFrame(chosen, columns=["only_split_by", "context"]).to_csv(sys.stdout, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", type=int, default=0, help="print this many disagreements as CSV, for review"
    )
    args = parser.parse_args(argv)

    sections = (
        pd.read_parquet(REPO_ROOT / CORPUS, columns=["section_1A"])
        .sample(50, random_state=0)["section_1A"]
        .tolist()
    )
    megabytes = sum(len(s.encode()) for s in sections) / 1e6
    print(f"50 sections sampled with random_state=0: {megabytes:.1f} MB of text\n")

    rows = []
    for name, splitter in SPLITTERS.items():
        passed = sum(splitter(text) == expected for text, expected in SPLIT_CASES)
        start = time.perf_counter()
        sentences = [s for section in sections for s in splitter(section)]
        seconds = time.perf_counter() - start
        lengths = pd.Series([len(s) for s in sentences])
        speed = megabytes / seconds
        rows.append(
            {
                "splitter": name,
                "cases passed": f"{passed}/{len(SPLIT_CASES)}",
                "sentences/section": round(len(sentences) / len(sections), 1),
                f"over {LONG_SENTENCE_CHARS:,} chars": f"{(lengths > LONG_SENTENCE_CHARS).mean():.2%}",
                f"under {SHORT_SENTENCE_CHARS} chars": f"{(lengths < SHORT_SENTENCE_CHARS).mean():.1%}",
                "MB/s": round(speed, 2),
                f"full corpus, {WORKERS} workers": _duration(CORPUS_GB * 1000 / speed / WORKERS),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    head_to_head(sections, args.sample)


if __name__ == "__main__":
    main()
