"""Assemble `docs/data/report.json`: everything the report page shows.

The page reads this one committed file and nothing else. It holds the fields
the page uses from the three results files, and one real filing drawn
sentence by sentence as a redline against the year before. The filing is
chosen by a rule, stated on the page, and not by eye.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.edgar import fetch
from pipeline.industry import SUBMISSIONS_URL
from pipeline.io import read_sentences

EXAMPLE_YEAR = 2019
EXAMPLE_SENTENCES = (150, 400)  # long enough to be a real section, short enough to draw
SHARES = ("exact", "near", "paraphrase", "new", "removed")
DECIMALS = 6

# What the page reads from each results file, as dotted paths. A path missing from its
# file stops the build and is named, so the page can't be published with a hole in it.
FIELDS = {
    "measure": [
        "model",
        "tau_v",
        "tau_p",
        "shares.filings",
        "shares.companies",
        "shares.sentences",
        "shares.previous_sentences",
        "shares.pooled",
        "shares.pooled_chars",
        "shares.by_filing",
        "shares.share_new_histogram",
        "shares.filings_all_new",
        "shares.filings_nothing_new",
        "shares.by_year",
        "peers.primary_k",
        "peers.full_sentence_words",
        "peers.filings_in_groups_of_two_or_more",
        "peers.largest_group",
        "peers.overall",
        "peers.by_year",
        "peers.most_shared_last_year",
        "thresholds.f1_v",
        "thresholds.tau_p_known_answer",
        "thresholds.f1_p_known_answer",
        "thresholds.known_answer",
        "thresholds.real_pairs.rounds",
        "thresholds.real_pairs.good_kappa",
        "thresholds.real_pairs.n",
        "thresholds.real_pairs.tuned_on",
        "thresholds.real_pairs.makeup",
        "thresholds.real_pairs.intervals",
        "thresholds.real_pairs.at_known_answer_tau_p",
        "thresholds.real_pairs.other_views",
        "thresholds.real_pairs.by_cosine",
        "thresholds.real_pairs.by_jaccard",
        "embedders.candidates",
        "embedders.chosen",
        "baselines.random_pairs",
        "baselines.shuffled_words",
        "baselines.stranger_filings",
        "baselines.unmatched_only",
        "baselines.near_peers",
    ],
    "explain": [
        "counts",
        "survival",
        "hazard_sample",
        "cox",
        "effect_varies_by_age",
        "hazard",
        "variance",
    ],
    "test2020": [
        "sample",
        "exposure",
        "covid_by_month",
        "did",
        "robustness",
        "event_study",
        "placebos",
        "year_after",
    ],
}


def closest_to_median(
    metrics: pd.DataFrame, year: int, sentences: tuple[int, int]
) -> tuple[int, int]:
    """The filing of `year` most like the median filing: its (cik, year).

    Candidates are the year's paired filings with a number of sentences in
    `sentences` (both ends included). The medians of the five shares (kept
    word for word, edited, reworded, new, removed) are taken over every
    paired filing of every year, and the candidate at the smallest straight-
    line distance from them is chosen. A tie goes to the lowest CIK.

    The distance is Euclidean, not a sum of absolute differences: the four
    shares of this year's sentences add up to one and their medians don't,
    so every filing with all four at or above the median ties on the sum.
    """
    columns = [f"share_{share}" for share in SHARES]
    medians = metrics[columns].median()
    pool = metrics[
        (metrics["filing_year"] == year) & metrics["n_sentences"].between(*sentences)
    ].dropna(subset=columns)
    distance = np.sqrt(((pool[columns] - medians) ** 2).sum(axis=1)).round(12)
    best = pool.assign(distance=distance).sort_values(["distance", "cik"]).iloc[0]
    return int(best["cik"]), year


def pick_example_filing(
    cfg: Config, year: int = EXAMPLE_YEAR, sentences: tuple[int, int] = EXAMPLE_SENTENCES
) -> tuple[int, int]:
    """`closest_to_median` on the pipeline's filing metrics."""
    metrics = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet")
    return closest_to_median(metrics, year, sentences)


def company_name(submissions: Path, cik: int, on: pd.Timestamp) -> str | None:
    """The name a company filed under on a date, from SEC's bulk submissions archive.

    Companies rename themselves (Kellogg Co became Kellanova in 2023), and a
    filing should be shown under the name on its cover. None if the archive
    has no record of the company.
    """
    with zipfile.ZipFile(submissions) as zf:
        try:
            record = json.loads(zf.read(f"CIK{cik:010d}.json"))
        except KeyError:
            return None
    for former in record.get("formerNames", []):
        start, end = (
            pd.Timestamp(former[key]).tz_localize(None) if former.get(key) else None
            for key in ("from", "to")
        )
        if (start is None or start <= on) and (end is None or on <= end):
            return former["name"]
    return record.get("name")


def example_redline(cfg: Config, cik: int, year: int, submissions: Path | None = None) -> dict:
    """One filing as a redline against the same company's filing the year before.

    `sentences` are this year's, in order, each with its label and, where it
    was edited or reworded, last year's wording (`previous`). `removed` are
    last year's sentences that nothing matched, each with `after`: the
    position in `sentences` of the sentence it would have followed (the last
    of last year's sentences before it that lives on), or -1 if nothing
    before it was kept.
    """
    labels = pd.read_parquet(cfg.interim_dir / "labels" / f"pair={year}.parquet")
    labels = labels[labels["cik"] == cik]
    text = ["cik", "sent_idx", "para_idx", "text", "n_chars", "is_heading"]

    def side(name: str, sentence_year: int) -> pd.DataFrame:
        sentences = read_sentences(cfg, sentence_year, text)
        joined = labels[labels["side"] == name].merge(
            sentences[sentences["cik"] == cik], on=["cik", "sent_idx"]
        )
        return joined.sort_values("sent_idx", kind="stable").reset_index(drop=True)

    current, previous = side("cur", year), side("prev", year - 1)
    position = {int(idx): i for i, idx in enumerate(current["sent_idx"])}
    previous_text = previous.set_index("sent_idx")["text"]

    sentences = []
    for row in current.itertuples():
        item = {
            "label": row.label,
            "chars": int(row.n_chars),
            "paragraph": int(row.para_idx),
            "heading": bool(row.is_heading),
            "text": row.text,
        }
        if row.label in ("near", "paraphrase"):
            item["previous"] = previous_text[row.link_idx]
        sentences.append(item)

    removed, anchor = [], -1
    for row in previous.itertuples():
        if row.label == "removed":
            removed.append({"after": anchor, "chars": int(row.n_chars), "text": row.text})
        elif row.link_idx in position:
            anchor = position[row.link_idx]

    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet",
        columns=["cik", "filing_year", "accession", "date_filed"],
    )
    filing = filings[(filings["cik"] == cik) & (filings["filing_year"] == year)].iloc[0]
    metrics = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet")
    shares = metrics[(metrics["cik"] == cik) & (metrics["filing_year"] == year)].iloc[0]
    if submissions is None:
        submissions = fetch(
            SUBMISSIONS_URL,
            cfg.raw_dir / SUBMISSIONS_URL.split("/Archives/edgar/", 1)[1],
            cfg.sec_user_agent,
        )
    accession = filing["accession"]
    return {
        "cik": cik,
        "year": year,
        "name": company_name(submissions, cik, pd.Timestamp(filing["date_filed"])),
        "accession": accession,
        "date_filed": pd.Timestamp(filing["date_filed"]).date().isoformat(),
        "url": (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{accession.replace('-', '')}/{accession}-index.htm"
        ),
        "shares": {share: float(shares[f"share_{share}"]) for share in SHARES},
        "sentences": sentences,
        "removed": removed,
    }


def _rounded(value):
    """Floats to DECIMALS places, all the way down: enough for the page, and a stable file."""
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def select(source: dict, paths: list[str], name: str) -> dict:
    """The parts of `source` at the dotted `paths`, and nothing else.

    `name` is the file `source` came from, for the error that says which
    path it lacks.
    """
    out: dict = {}
    for path in paths:
        node, target = source, out
        keys = path.split(".")
        for depth, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"{name} has no {path}")
            node = node[key]
            if depth < len(keys) - 1:
                target = target.setdefault(key, {})
        target[keys[-1]] = _rounded(node)
    return out


def build_report_json(
    cfg: Config,
    *,
    fields: dict[str, list[str]] | None = None,
    example_year: int = EXAMPLE_YEAR,
    example_sentences: tuple[int, int] = EXAMPLE_SENTENCES,
    submissions: Path | None = None,
) -> Path:
    """Write `docs/data/report.json` from the results files and the example filing."""
    report = {
        name: select(
            json.loads((cfg.results_dir / f"{name}.json").read_text()), paths, f"{name}.json"
        )
        for name, paths in (fields or FIELDS).items()
    }
    cik, year = pick_example_filing(cfg, example_year, example_sentences)
    report["example"] = _rounded(example_redline(cfg, cik, year, submissions)) | {
        "rule": {"year": example_year, "sentences": list(example_sentences)}
    }
    path = cfg.root / "docs" / "data" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return path
