"""Turn scores into labels, and labels into each filing's shares.

A sentence in this year's filing is:

- `exact`: in last year's filing word for word (after normalization);
- `near`: a light edit of a sentence there (shingle Jaccard at least `tau_v`);
- `paraphrase`: a rewording of one (cosine at least `tau_p`);
- `new`: none of those.

A sentence in last year's filing gets the same first three labels when some
sentence this year matches it, and `removed` when none does.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences, write_parquet

CURRENT_LABELS = ("exact", "near", "paraphrase", "new")
PEER_GROUPS = 5  # "shared with peers": filed word for word by at least this many other groups
LABEL_COLUMNS = ["cik", "side", "sent_idx", "label", "link_idx"]


def label(scores: pd.DataFrame, tau_v: float, tau_p: float) -> pd.DataFrame:
    """`label` and `link_idx` for each row of a similarity table.

    `link_idx` is the `sent_idx` of the sentence on the other side that earned
    the label, or -1. A score exactly at a threshold is a match; a missing
    score is not.
    """
    exact = scores["exact"].to_numpy()
    near = ~exact & (scores["best_j"].to_numpy() >= tau_v)
    reworded = ~exact & ~near & (scores["best_cos"].to_numpy() >= tau_p)
    unmatched = np.where(scores["side"] == "cur", "new", "removed")
    labels = np.select([exact, near, reworded], ["exact", "near", "paraphrase"], unmatched)
    link = np.select(
        [exact, near, reworded],
        [scores["exact_idx"], scores["j_idx"], scores["cos_idx"]],
        -1,
    )
    return pd.DataFrame({"label": labels, "link_idx": link.astype(np.int64)}, index=scores.index)


def build_labels(cfg: Config, tau_v: float, tau_p: float) -> Path:
    """Label every sentence of every pair: `labels/pair=YYYY.parquet` (LABEL_COLUMNS).

    Always rewritten: it is cheap, and the thresholds may have changed.
    """
    out_dir = cfg.interim_dir / "labels"
    for path in sorted((cfg.interim_dir / "similarity").glob("pair=*.parquet")):
        scores = pd.read_parquet(path)
        labeled = pd.concat(
            [scores[["cik", "side", "sent_idx"]], label(scores, tau_v, tau_p)], axis=1
        )
        write_parquet(labeled[LABEL_COLUMNS], out_dir / path.name)
    return out_dir


def _shares(
    frame: pd.DataFrame, labels: tuple[str, ...], suffix: str, weight: str | None
) -> pd.DataFrame:
    """Per company, the share of sentences (or of characters) with each label."""
    weights = frame[weight] if weight else pd.Series(1, index=frame.index)
    total = weights.groupby(frame["cik"]).sum()
    out = {}
    for name in labels:
        out[f"share_{name}{suffix}"] = (
            weights.where(frame["label"] == name, 0).groupby(frame["cik"]).sum() / total
        )
    return pd.DataFrame(out)


def build_filing_metrics(cfg: Config) -> Path:
    """One row per filing that has a previous-year filing.

    `share_exact`, `share_near`, `share_paraphrase` and `share_new` are shares
    of this year's sentences and sum to 1. `share_removed` is the share of
    last year's sentences with no match this year. `share_peer` is the share
    of this year's sentences filed word for word by at least PEER_GROUPS other
    groups of companies. Each share also comes weighted by characters
    (`*_chars`), so one long sentence counts for more than one short one.
    """
    parts = []
    for path in sorted((cfg.interim_dir / "labels").glob("pair=*.parquet")):
        year = int(path.stem.split("=")[1])
        labels = pd.read_parquet(path)
        sides = {}
        for side, sentence_year in (("cur", year), ("prev", year - 1)):
            sentences = read_sentences(
                cfg, sentence_year, ["cik", "sent_idx", "n_chars", "norm_hash"]
            )
            sides[side] = labels[labels["side"] == side].merge(sentences, on=["cik", "sent_idx"])
        cur, prev = sides["cur"], sides["prev"]

        peers = pd.read_parquet(cfg.interim_dir / "peers" / f"year={year}.parquet")
        others = cur["norm_hash"].map(peers.set_index("norm_hash")["n_groups"]) - 1
        cur = cur.assign(label_peer=np.where(others >= PEER_GROUPS, "peer", "own"))

        metrics = pd.concat(
            [
                cur.groupby("cik").agg(n_sentences=("label", "size"), n_chars=("n_chars", "sum")),
                prev.groupby("cik").agg(n_prev=("label", "size"), n_prev_chars=("n_chars", "sum")),
                _shares(cur, CURRENT_LABELS, "", None),
                _shares(prev, ("removed",), "", None),
                _shares(cur.assign(label=cur["label_peer"]), ("peer",), "", None),
                _shares(cur, CURRENT_LABELS, "_chars", "n_chars"),
                _shares(prev, ("removed",), "_chars", "n_chars"),
                _shares(cur.assign(label=cur["label_peer"]), ("peer",), "_chars", "n_chars"),
            ],
            axis=1,
        )
        # A filing whose previous year has no sentences at all still gets a row.
        metrics[["n_prev", "n_prev_chars"]] = (
            metrics[["n_prev", "n_prev_chars"]].fillna(0).astype("int64")
        )
        parts.append(metrics.reset_index().assign(filing_year=year))
    out = cfg.interim_dir / "filing_metrics.parquet"
    table = pd.concat(parts, ignore_index=True)
    first = ["cik", "filing_year", "n_sentences", "n_prev", "n_chars", "n_prev_chars"]
    write_parquet(table[first + [c for c in table.columns if c not in first]], out)
    return out
