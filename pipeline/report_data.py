"""Gather the measurement's results into one file the write-up and the report read.

`results/measure.json` holds every number `docs/measurement.md` quotes. It is
built from the stage outputs, so a number in the write-up can be traced to the
command that printed it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.baselines import (
    random_pairs,
    shuffled_words,
    stranger_filings,
    unmatched_only_check,
)
from pipeline.config import Config
from pipeline.embed import MODEL, Embedder, load_embedder
from pipeline.io import read_sentences, sentence_years
from pipeline.labels import CURRENT_LABELS, PEER_GROUPS
from pipeline.peers import near_column, near_peer_sample
from pipeline.thresholds import read_thresholds

SHARES = ("exact", "near", "paraphrase", "new", "removed", "peer")
REACH = (1, 5, 10, 25)  # "filed by at least this many other groups"
FULL_SENTENCE_WORDS = 10
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _pooled(metrics: pd.DataFrame, by_chars: bool = False) -> dict:
    """Each share over all sentences (or characters) of the filings given."""
    suffix = "_chars" if by_chars else ""
    weights = {
        "current": metrics["n_chars" if by_chars else "n_sentences"],
        "previous": metrics["n_prev_chars" if by_chars else "n_prev"],
    }
    out = {}
    for share in SHARES:
        weight = weights["previous" if share == "removed" else "current"]
        out[share] = float(
            (metrics[f"share_{share}{suffix}"].fillna(0) * weight).sum() / weight.sum()
        )
    return out


def _spread(metrics: pd.DataFrame) -> dict:
    """Quantiles of each share across filings."""
    return {
        share: {
            f"p{int(q * 100)}": float(v)
            for q, v in zip(QUANTILES, metrics[f"share_{share}"].quantile(QUANTILES), strict=True)
        }
        for share in SHARES
    }


def shares(cfg: Config) -> dict:
    """Carried over, new, removed and shared with peers: pooled, by filing, and by year."""
    metrics = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet")
    by_year = []
    for year, part in metrics.groupby("filing_year"):
        by_year.append(
            {
                "year": int(year),
                "filings": len(part),
                "sentences": int(part["n_sentences"].sum()),
                "pooled": _pooled(part),
                "pooled_chars": _pooled(part, by_chars=True),
                "median": {s: float(part[f"share_{s}"].median()) for s in SHARES},
                "p25": {s: float(part[f"share_{s}"].quantile(0.25)) for s in SHARES},
                "p75": {s: float(part[f"share_{s}"].quantile(0.75)) for s in SHARES},
            }
        )
    histogram = np.histogram(metrics["share_new"], bins=np.linspace(0, 1, 41))[0]
    return {
        "filings": len(metrics),
        "companies": int(metrics["cik"].nunique()),
        "sentences": int(metrics["n_sentences"].sum()),
        "previous_sentences": int(metrics["n_prev"].sum()),
        "pooled": _pooled(metrics),
        "pooled_chars": _pooled(metrics, by_chars=True),
        "by_filing": _spread(metrics),
        "share_new_histogram": {"bin_width": 0.025, "counts": [int(c) for c in histogram]},
        "filings_all_new": int((metrics["share_new"] >= 0.999).sum()),
        "filings_nothing_new": int((metrics["share_new"] <= 0.001).sum()),
        "by_year": by_year,
    }


def partner_agreement(cfg: Config, tau_v: float) -> dict:
    """How often a sentence's closest partner by Jaccard and by cosine are the same sentence."""
    n = same = n_near = same_near = 0
    for path in sorted((cfg.interim_dir / "similarity").glob("pair=*.parquet")):
        df = pd.read_parquet(path, columns=["side", "best_j", "j_idx", "best_cos", "cos_idx"])
        df = df[(df["side"] == "cur") & df["best_cos"].notna() & (df["j_idx"] >= 0)]
        agree = df["j_idx"] == df["cos_idx"]
        near = df["best_j"] >= tau_v
        n, same = n + len(df), same + int(agree.sum())
        n_near, same_near = n_near + int(near.sum()), same_near + int((agree & near).sum())
    return {
        "sentences_with_both": n,
        "share_same_partner": same / n,
        "near_sentences": n_near,
        "share_same_partner_among_near": same_near / n_near,
    }


def peer_reach(cfg: Config) -> dict:
    """Share of sentences filed word for word by at least k other groups, by year."""
    groups = pd.read_parquet(cfg.interim_dir / "filer_groups.parquet")
    sizes = groups.groupby(["filing_year", "group"]).size()
    by_year = []
    totals = {"all": np.zeros(len(REACH) + 1), "full": np.zeros(len(REACH) + 1)}
    examples = None
    for year in sentence_years(cfg):
        sentences = read_sentences(cfg, year, ["cik", "norm", "norm_hash", "is_heading", "text"])
        peers = pd.read_parquet(cfg.interim_dir / "peers" / f"year={year}.parquet")
        others = sentences["norm_hash"].map(peers.set_index("norm_hash")["n_groups"]) - 1
        full = ~sentences["is_heading"] & (
            sentences["norm"].str.count(" ") + 1 >= FULL_SENTENCE_WORDS
        )
        row = {"year": year, "sentences": len(sentences), "full_sentences": int(full.sum())}
        for name, mask in (("all", np.ones(len(sentences), dtype=bool)), ("full", full.to_numpy())):
            hits = [int((others[mask] >= k).sum()) for k in REACH]
            row[name] = {f"k{k}": h / int(mask.sum()) for k, h in zip(REACH, hits, strict=True)}
            totals[name] += np.array([*hits, int(mask.sum())])
        by_year.append(row)
        if year == sentence_years(cfg)[-1]:
            # The rule for the examples: the most widely filed full sentences of the last year.
            top = sentences[full].assign(others=others[full]).drop_duplicates("norm_hash")
            top = top.nlargest(12, "others")
            examples = [
                {"text": t, "other_groups": int(o)}
                for t, o in zip(top["text"], top["others"], strict=True)
            ]
    return {
        "primary_k": PEER_GROUPS,
        "full_sentence_words": FULL_SENTENCE_WORDS,
        "filings_in_groups_of_two_or_more": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "overall": {
            name: {f"k{k}": float(total[i] / total[-1]) for i, k in enumerate(REACH)}
            for name, total in totals.items()
        },
        "by_year": by_year,
        "most_shared_last_year": examples,
    }


NEAR_THRESHOLDS = (0.5, 0.8)  # stricter cuts than tau_v, reported beside it


def near_peer_summary(cfg: Config, tau_v: float, n_per_year: int = 10_000) -> dict:
    """Word-for-word reach against reach that also counts near-verbatim copies, on a sample.

    `tau_v` was tuned for comparing a sentence with one company's previous
    filing. Against every sentence filed in a year it is a much weaker test,
    because unrelated sentences that share a stock phrase clear it. So reach
    is also given at stricter cuts.
    """
    thresholds = (tau_v, *NEAR_THRESHOLDS)
    sample = near_peer_sample(cfg, thresholds, n_per_year=n_per_year)
    full = ~sample["is_heading"] & (sample["n_words"] >= FULL_SENTENCE_WORDS)
    out = {"n": len(sample), "n_full": int(full.sum()), "thresholds": list(thresholds)}
    for name, part in (("all", sample), ("full", sample[full])):
        out[name] = {
            f"k{k}": {"exact": float((part["exact_others"] >= k).mean())}
            | {f"near_{t:.2f}": float((part[near_column(t)] >= k).mean()) for t in thresholds}
            for k in REACH
        }
    return out


def examples(cfg: Config, n_filings: int = 200, per_label: int = 3, seed: int = 0) -> list[dict]:
    """A few sentences of each label, chosen by rule and not by eye.

    The rule: sample `n_filings` paired filings with the seed given; within
    each label take the first `per_label` full sentences in (cik, year,
    position) order. Each comes with the previous-year sentence it matched,
    its scores, and its filing's accession number so it can be looked up.
    """
    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year", "accession"]
    )
    metrics = pd.read_parquet(
        cfg.interim_dir / "filing_metrics.parquet", columns=["cik", "filing_year"]
    )
    chosen = metrics.sample(min(n_filings, len(metrics)), random_state=seed)
    rows = []
    for year, part in chosen.groupby("filing_year"):
        labels = pd.read_parquet(cfg.interim_dir / "labels" / f"pair={year}.parquet")
        labels = labels[(labels["side"] == "cur") & labels["cik"].isin(part["cik"])]
        scores = pd.read_parquet(
            cfg.interim_dir / "similarity" / f"pair={year}.parquet",
            columns=["cik", "side", "sent_idx", "best_j", "best_cos"],
        )
        labels = labels.merge(scores, on=["cik", "side", "sent_idx"])
        text = ["cik", "sent_idx", "text", "norm", "is_heading"]
        current = labels.merge(read_sentences(cfg, year, text), on=["cik", "sent_idx"])
        previous = read_sentences(cfg, year - 1, ["cik", "sent_idx", "text"])
        previous = previous.rename(columns={"sent_idx": "link_idx", "text": "previous_text"})
        current = current.merge(previous, on=["cik", "link_idx"], how="left")
        rows.append(current.assign(filing_year=year))
    pool = pd.concat(rows).sort_values(["cik", "filing_year", "sent_idx"], kind="stable")
    pool = pool[~pool["is_heading"] & (pool["norm"].str.count(" ") + 1 >= FULL_SENTENCE_WORDS)]
    pool = pool.merge(filings, on=["cik", "filing_year"], how="left")
    out = []
    for name in CURRENT_LABELS:
        for row in pool[pool["label"] == name].head(per_label).itertuples():
            out.append(
                {
                    "label": name,
                    "cik": int(row.cik),
                    "year": int(row.filing_year),
                    "accession": row.accession,
                    "text": row.text,
                    "previous_text": None if pd.isna(row.previous_text) else row.previous_text,
                    "jaccard": None if pd.isna(row.best_j) else round(float(row.best_j), 3),
                    "cosine": None if pd.isna(row.best_cos) else round(float(row.best_cos), 3),
                }
            )
    return out


def build_baselines(cfg: Config, embedder: Embedder | None = None) -> Path:
    """Run the four baselines and the near-verbatim peer sample: `results/baselines.json`."""
    tau_v, tau_p = read_thresholds(cfg)
    embedder = embedder or load_embedder()
    out = {
        "tau_v": tau_v,
        "tau_p": tau_p,
        "random_pairs": random_pairs(cfg, tau_v, tau_p, embedder),
        "shuffled_words": shuffled_words(cfg, tau_v, tau_p, embedder),
        "stranger_filings": stranger_filings(cfg, tau_v, tau_p, embedder),
        "unmatched_only": unmatched_only_check(cfg, tau_v, tau_p, embedder),
        "near_peers": near_peer_summary(cfg, tau_v),
    }
    path = cfg.results_dir / "baselines.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def build_measure_results(cfg: Config) -> Path:
    """Everything the measurement write-up quotes, in `results/measure.json`."""
    tau_v, tau_p = read_thresholds(cfg)
    read = lambda name: json.loads((cfg.results_dir / name).read_text())
    out = {
        "model": MODEL,
        "tau_v": tau_v,
        "tau_p": tau_p,
        "shares": shares(cfg),
        "partner_agreement": partner_agreement(cfg, tau_v),
        "peers": peer_reach(cfg),
        "thresholds": read("thresholds.json"),
        "embedders": read("embedders.json"),
        "baselines": read("baselines.json"),
        "examples": examples(cfg),
    }
    path = cfg.results_dir / "measure.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path
