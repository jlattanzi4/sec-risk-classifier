"""Set the two thresholds and say how well they work.

`tau_v` is the shingle Jaccard at or above which two sentences are the same
sentence lightly edited. `tau_p` is the cosine at or above which a sentence
that isn't a light edit is a rewording. Both are tuned on half of the
known-answer set, by rules fixed before any result was seen, and reported on
the other half and on real pairs labeled by two judges.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.embed import Embedder, load_embedder
from pipeline.sentences import normalize
from pipeline.shingles import jaccard, shingles

TIERS = ("near", "paraphrase", "new")
CARRIED_KINDS = ("verbatim", "paraphrase")
KINDS = ("verbatim", "paraphrase", "different_hard", "different_easy")
JUDGE_LABELS = ("same", "paraphrase", "different")
GOOD_KAPPA = 0.70


def best_threshold(scores: pd.Series, positive: pd.Series) -> tuple[float, float]:
    """The cut on `scores` with the best F1 at picking out `positive`, and that F1.

    Candidates are the midpoints between neighboring distinct scores. Between
    equally good cuts the higher wins: when in doubt, match less.
    """
    scores, positive = scores.to_numpy(dtype=float), positive.to_numpy(dtype=bool)
    distinct = np.unique(scores)
    best, best_f1 = float(distinct[-1]) + 1.0, 0.0
    for cut in (distinct[:-1] + distinct[1:]) / 2:
        predicted = scores >= cut
        hits = int((predicted & positive).sum())
        f1 = 2 * hits / (predicted.sum() + positive.sum()) if hits else 0.0
        if f1 >= best_f1:
            best, best_f1 = float(cut), float(f1)
    return best, best_f1


def tune(scores: pd.DataFrame) -> dict:
    """`tau_v` and `tau_p` from the tuning half (`split == "tune"`).

    `tau_v` separates `verbatim` pairs from `paraphrase` and `different_easy`
    by Jaccard. `different_hard` is left out on purpose: a sentence whose
    claim was reversed in a word or two shares as many shingles as a light
    edit, so no cut on word overlap can tell them apart, and tuning against
    them would only push the threshold up for nothing. `evaluate_known`
    reports what happens to them. `tau_p` separates `paraphrase` from both
    kinds of different sentence by cosine, among pairs under `tau_v`.
    """
    half = scores[scores["split"] == "tune"]
    first = half[half["kind"].isin(["verbatim", "paraphrase", "different_easy"])]
    tau_v, f1_v = best_threshold(first["j"], first["kind"] == "verbatim")
    second = half[(half["j"] < tau_v) & (half["kind"] != "verbatim")]
    tau_p, f1_p = best_threshold(second["cos"], second["kind"] == "paraphrase")
    return {"tau_v": tau_v, "f1_v": f1_v, "tau_p": tau_p, "f1_p": f1_p}


def predict(j: pd.Series, cos: pd.Series, tau_v: float, tau_p: float) -> pd.Series:
    """The matcher's call for each pair: `near`, then `paraphrase`, else `new`.
    A score exactly at a threshold counts as a match; a missing score never does."""
    tier = np.where(
        j.to_numpy() >= tau_v, "near", np.where(cos.to_numpy() >= tau_p, "paraphrase", "new")
    )
    return pd.Series(tier, index=j.index)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a share of `hits` in `n`."""
    if n == 0:
        return 0.0, 1.0
    share = hits / n
    center = share + z * z / (2 * n)
    spread = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    return float((center - spread) / (1 + z * z / n)), float((center + spread) / (1 + z * z / n))


def _share(hits: int, n: int) -> dict:
    low, high = wilson(hits, n)
    return {
        "value": hits / n if n else float("nan"),
        "low": low,
        "high": high,
        "hits": hits,
        "n": n,
    }


def evaluate_known(scores: pd.DataFrame, tau_v: float, tau_p: float) -> dict:
    """Precision and recall on the held-out half (`split == "test"`).

    `labeled[kind][tier]` counts how each kind of pair was labeled.
    `carried_over` treats `near` and `paraphrase` as one answer (the sentence
    was carried over) and `verbatim` and `paraphrase` pairs as the ones that
    should get it; precision is given with and without the `different_hard`
    pairs, which word overlap cannot catch. `near` and `paraphrase` report
    each tier against its own kind.
    """
    half = scores[scores["split"] == "test"]
    tier = predict(half["j"], half["cos"], tau_v, tau_p)
    kind = half["kind"]
    labeled = {
        k: {t: int(((kind == k) & (tier == t)).sum()) for t in TIERS}
        for k in KINDS
        if (kind == k).any()
    }
    carried, should = tier != "new", kind.isin(CARRIED_KINDS)
    easy_only = kind != "different_hard"

    def tier_report(name: str, own_kind: str) -> dict:
        called, is_kind = tier == name, kind == own_kind
        return {
            "precision": _share(int((called & is_kind).sum()), int(called.sum())),
            "precision_without_hard": _share(
                int((called & is_kind).sum()), int((called & easy_only).sum())
            ),
            "recall": _share(int((called & is_kind).sum()), int(is_kind.sum())),
        }

    return {
        "n": len(half),
        "labeled": labeled,
        "carried_over": {
            "precision": _share(int((carried & should).sum()), int(carried.sum())),
            "precision_without_hard": _share(
                int((carried & should).sum()), int((carried & easy_only).sum())
            ),
            "recall": _share(int((carried & should).sum()), int(should.sum())),
        },
        "near": tier_report("near", "verbatim"),
        "paraphrase": tier_report("paraphrase", "paraphrase"),
    }


def tier_makeup(pairs: pd.DataFrame, tau_v: float, tau_p: float) -> dict:
    """What the judges made of the real pairs in each tier, weighted to the population.

    `pairs` has `pair_j`, `pair_cos`, `label` and `weight`. For each tier:
    the weighted share judged same, paraphrase and different, how many
    sampled pairs that rests on (`n`) and how many pairs in the corpus they
    stand for. `recall_carried` is the weighted share of pairs judged same or
    paraphrase that the matcher called carried over.
    """
    tier = predict(pairs["pair_j"], pairs["pair_cos"], tau_v, tau_p)
    weight = pairs["weight"]
    out = {}
    for name in TIERS:
        inside = tier == name
        total = weight[inside].sum()
        out[name] = {
            label: float(weight[inside & (pairs["label"] == label)].sum() / total)
            if total
            else float("nan")
            for label in JUDGE_LABELS
        }
        out[name] |= {"n": int(inside.sum()), "stands_for": float(total)}
    judged_carried = pairs["label"] != "different"
    out["recall_carried"] = float(
        weight[judged_carried & (tier != "new")].sum() / weight[judged_carried].sum()
    )
    return out


def score_known_answer(cfg: Config, embedder: Embedder | None = None) -> pd.DataFrame:
    """The confirmed known-answer pairs with their Jaccard (`j`) and cosine (`cos`),
    computed exactly as the pipeline computes them for real sentences."""
    pairs = pd.read_csv(cfg.validation_dir / "known_answer" / "pairs.csv")
    pairs = pairs[pairs["confirmed"]].reset_index(drop=True)
    pairs["j"] = [
        jaccard(shingles(normalize(a)), shingles(normalize(b)))
        for a, b in zip(pairs["a"], pairs["b"], strict=True)
    ]
    embedder = embedder or load_embedder()
    a, b = embedder.encode(pairs["a"].tolist()), embedder.encode(pairs["b"].tolist())
    # Stored vectors are float16, so score these the same way.
    a, b = a.astype(np.float16).astype(np.float32), b.astype(np.float16).astype(np.float32)
    pairs["cos"] = (a * b).sum(axis=1)
    return pairs


def read_thresholds(cfg: Config) -> tuple[float, float]:
    """(`tau_v`, `tau_p`) as last tuned."""
    saved = json.loads((cfg.results_dir / "thresholds.json").read_text())
    return saved["tau_v"], saved["tau_p"]


def results_path(cfg: Config) -> Path:
    return cfg.results_dir / "thresholds.json"
