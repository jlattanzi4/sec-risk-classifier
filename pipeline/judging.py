"""Real pairs of sentences, labeled by two judges.

The known-answer set shows the matcher works on cases built to have an answer.
This checks it on what it actually meets: sentences from consecutive filings,
each with the previous-year sentence the matcher found closest. Pairs are
sampled evenly across similarity bands, so there are enough of them wherever
a threshold might fall, and each carries a weight that turns the sample back
into the population.

Judge 1 is Claude Haiku, on every pair. Judge 2 is Claude in a working
session, on a subset, without seeing judge 1's labels, the similarity scores,
or which band a pair came from. Both follow `rubric_v1.md`. No person labeled
anything; the write-up says so.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pipeline.config import Config
from pipeline.io import read_sentences
from pipeline.llm import claude_cli, run_batches
from pipeline.shingles import jaccard, shingles

LABELS = ("same", "paraphrase", "different")
J_EDGES = [0.0, 0.1, 0.3, 0.6, 1.01]
COS_EDGES = [0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]


def _allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Split `n` evenly over cells, giving a small cell all it has and the rest the remainder."""
    take = pd.Series(0, index=sizes.index)
    remaining = min(n, int(sizes.sum()))
    while remaining > 0:
        open_cells = sizes.index[take < sizes]
        share = max(remaining // len(open_cells), 1)
        for cell in open_cells:
            add = min(share, sizes[cell] - take[cell], remaining)
            take[cell] += add
            remaining -= add
    return take


def _band(values: pd.Series, edges: list[float], name: str) -> pd.Series:
    labels = [
        f"{name} {lo:.2f}-{min(hi, 1.0):.2f}" for lo, hi in zip(edges, edges[1:], strict=False)
    ]
    return pd.cut(values, edges, right=False, labels=labels).astype(str)


def stratified_sample(
    population: pd.DataFrame, n: int, cos_edges: list[float], j_edges: list[float], seed: int = 0
) -> pd.DataFrame:
    """`n` rows of `population`, the same number from each band of cosine by Jaccard.

    Adds `stratum` and `weight` (the cell's population over its sample size),
    so a weighted share over the sample estimates the share in the population.
    """
    stratum = (
        _band(population["pair_cos"], cos_edges, "cos")
        + " | "
        + _band(population["pair_j"], j_edges, "j")
    )
    population = population.assign(stratum=stratum)
    sizes = population.groupby("stratum").size()
    take = _allocate(sizes, n)
    parts = [
        cell.sample(int(take[name]), random_state=seed).assign(weight=sizes[name] / take[name])
        for name, cell in population.groupby("stratum")
    ]
    return pd.concat(parts, ignore_index=True)


def write_judge_inputs(
    pairs: pd.DataFrame, folder: Path, n_second: int = 400, seed: int = 0
) -> None:
    """Save the sample, and what each judge is shown: the id and the two sentences only.

    Each judge's file is in its own shuffled order. Judge 2's is a subset
    spread evenly over the cells.
    """
    folder.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(folder / "pairs.csv", index=False)
    shown = ["id", "a", "b"]
    pairs[shown].sample(frac=1, random_state=seed).to_csv(folder / "judge1_input.csv", index=False)
    take = _allocate(pairs.groupby("stratum").size(), n_second)
    subset = pd.concat(
        cell.sample(int(take[name]), random_state=seed + 1)
        for name, cell in pairs.groupby("stratum")
    )
    subset[shown].sample(frac=1, random_state=seed + 2).to_csv(
        folder / "judge2_input.csv", index=False
    )


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """How far two judges agree on the same pairs.

    `kappa` is Cohen's kappa on the three labels; `kappa_carried` on carried
    over (same or paraphrase) against different; `table[x][y]` is the number
    of pairs the first judge called x and the second y.
    """
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    carried_a, carried_b = a != "different", b != "different"
    table = pd.crosstab(a, b).reindex(index=LABELS, columns=LABELS, fill_value=0)
    return {
        "n": len(a),
        "agreement": float((a == b).mean()),
        "kappa": float(cohen_kappa_score(a, b, labels=list(LABELS))),
        "agreement_carried": float((carried_a == carried_b).mean()),
        "kappa_carried": float(cohen_kappa_score(carried_a, carried_b)),
        "table": {x: {y: int(table.loc[x, y]) for y in LABELS} for x in LABELS},
    }


def real_pair_population(cfg: Config) -> pd.DataFrame:
    """Every current-year sentence exact matching left over, with its closest previous
    sentence by cosine: `cik`, `filing_year`, `sent_idx`, `prev_idx`, `pair_cos`, `pair_j`
    (the Jaccard of that same pair), and `best_j` (its best Jaccard against any sentence)."""
    parts = []
    for path in sorted((cfg.interim_dir / "similarity").glob("pair=*.parquet")):
        year = int(path.stem.split("=")[1])
        df = pd.read_parquet(path)
        df = df[(df["side"] == "cur") & df["best_cos"].notna()]
        pair_j = df["best_j"].to_numpy().copy()
        differs = (df["cos_idx"] != df["j_idx"]).to_numpy()
        if differs.any():
            other = df.loc[differs, ["cik", "sent_idx", "cos_idx"]]
            norm = ["cik", "sent_idx", "norm"]
            mine = other.merge(read_sentences(cfg, year, norm), on=["cik", "sent_idx"], how="left")
            theirs = other.merge(
                read_sentences(cfg, year - 1, norm).rename(columns={"sent_idx": "cos_idx"}),
                on=["cik", "cos_idx"],
                how="left",
            )
            pair_j[differs] = [
                jaccard(shingles(a), shingles(b))
                for a, b in zip(mine["norm"], theirs["norm"], strict=True)
            ]
        parts.append(
            pd.DataFrame(
                {
                    "cik": df["cik"].to_numpy(),
                    "filing_year": year,
                    "sent_idx": df["sent_idx"].to_numpy(),
                    "prev_idx": df["cos_idx"].to_numpy(),
                    "pair_cos": df["best_cos"].to_numpy(),
                    "pair_j": pair_j,
                    "best_j": df["best_j"].to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def sample_real_pairs(cfg: Config, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Draw the pairs to judge and attach their text: `a` is last year's sentence, `b` this year's."""
    sample = stratified_sample(real_pair_population(cfg), n, COS_EDGES, J_EDGES, seed)
    text = ["cik", "sent_idx", "text"]
    parts = []
    for year, rows in sample.groupby("filing_year"):
        rows = rows.merge(
            read_sentences(cfg, year, text).rename(columns={"text": "b"}), on=["cik", "sent_idx"]
        )
        previous = read_sentences(cfg, year - 1, text)
        previous = previous.rename(columns={"sent_idx": "prev_idx", "text": "a"})
        parts.append(rows.merge(previous, on=["cik", "prev_idx"]))
    pairs = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    pairs.insert(0, "id", [f"rp{i:04d}" for i in range(len(pairs))])
    return pairs


def judge_with_haiku(cfg: Config, call=claude_cli) -> pd.DataFrame:
    """Judge 1: Claude Haiku labels every pair, 40 to a call, following the rubric."""
    folder = cfg.validation_dir / "real_pairs"
    items = pd.read_csv(folder / "judge1_input.csv").to_dict("records")
    replies = run_batches(
        items,
        prompt_file=folder / "rubric_v1.md",
        out_dir=folder / "raw" / "judge1",
        model="haiku",
        batch_size=40,
        required=("label",),
        call=call,
    )
    labels = pd.DataFrame(
        {"id": [r["id"] for r in replies], "label": [r["label"].strip().lower() for r in replies]}
    )
    labels = labels[labels["label"].isin(LABELS)]
    labels.to_csv(folder / "judge1_labels.csv", index=False)
    return labels
