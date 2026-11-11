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
from pipeline.judging import agreement
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


def load_real_pairs(cfg: Config) -> pd.DataFrame:
    """The sampled real pairs with each judge's label (`label_1`, `label_2`; missing where
    a judge didn't label the pair)."""
    folder = cfg.validation_dir / "real_pairs"
    pairs = pd.read_csv(folder / "pairs.csv")
    for judge in (1, 2):
        path = folder / f"judge{judge}_labels.csv"
        labels = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["id", "label"])
        labels = labels.rename(columns={"label": f"label_{judge}"})
        pairs = pairs.merge(labels[["id", f"label_{judge}"]], on="id", how="left")
    return pairs


def _reweighted(subset: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """`subset` of the sampled pairs, with weights that again add up to each cell's population."""
    population = pairs.groupby("stratum")["weight"].sum()
    count = subset.groupby("stratum").size()
    return subset.assign(weight=subset["stratum"].map(population / count))


def bootstrap_makeup(
    pairs: pd.DataFrame, tau_v: float, tau_p: float, n: int = 1000, seed: int = 0
) -> dict:
    """95% intervals for `tier_makeup`, resampling pairs within each sampling cell."""
    rng = np.random.default_rng(seed)
    cells = [cell for _, cell in pairs.groupby("stratum")]
    draws: dict[tuple, list[float]] = {}
    for _ in range(n):
        resample = pd.concat(cell.iloc[rng.integers(0, len(cell), len(cell))] for cell in cells)
        makeup = tier_makeup(resample, tau_v, tau_p)
        for tier in TIERS:
            for label in JUDGE_LABELS:
                draws.setdefault((tier, label), []).append(makeup[tier][label])
            carried = makeup[tier]["same"] + makeup[tier]["paraphrase"]
            draws.setdefault((tier, "carried"), []).append(carried)
        draws.setdefault(("recall_carried",), []).append(makeup["recall_carried"])
    out: dict = {tier: {} for tier in TIERS}
    for key, values in draws.items():
        seen = np.array(values)[~np.isnan(values)]  # a tier can be empty in a resample
        low, high = np.quantile(seen, [0.025, 0.975]) if len(seen) else (np.nan, np.nan)
        if len(key) == 1:
            out[key[0]] = (float(low), float(high))
        else:
            out[key[0]][key[1]] = (float(low), float(high))
    return out


def evaluate_real(cfg: Config, tau_v: float, tau_p: float, n_boot: int = 1000) -> dict:
    """The matcher against the judged real pairs.

    The rule, fixed in the plan: if the two judges' three-label kappa on the
    pairs both labeled is at least GOOD_KAPPA, judge 1's labels stand for
    every pair; otherwise only pairs the two agree on are used. Either way
    the weighted makeup of each tier is reported with bootstrap intervals,
    and the other views are kept beside it for comparison.
    """
    pairs = load_real_pairs(cfg)
    both = pairs[pairs["label_1"].notna() & pairs["label_2"].notna()]
    agree = agreement(both["label_1"], both["label_2"])
    views = {
        "judge 1 on all pairs": pairs[pairs["label_1"].notna()].assign(
            label=lambda d: d["label_1"]
        ),
        "judge 2 on its subset": _reweighted(both.assign(label=both["label_2"]), pairs),
        "pairs both judges agree on": _reweighted(
            both[both["label_1"] == both["label_2"]].assign(label=lambda d: d["label_1"]), pairs
        ),
    }
    basis = "judge 1 on all pairs" if agree["kappa"] >= GOOD_KAPPA else "pairs both judges agree on"
    chosen = views[basis]
    return {
        "agreement": agree,
        "good_kappa": GOOD_KAPPA,
        "basis": basis,
        "n": len(chosen),
        "makeup": tier_makeup(chosen, tau_v, tau_p),
        "intervals": bootstrap_makeup(chosen, tau_v, tau_p, n=n_boot),
        "other_views": {
            name: tier_makeup(view, tau_v, tau_p) | {"n_pairs": len(view)}
            for name, view in views.items()
            if name != basis and len(view)
        },
        "labels": {
            "judge_1": {k: int(v) for k, v in pairs["label_1"].value_counts().items()},
            "judge_2": {k: int(v) for k, v in pairs["label_2"].value_counts().items()},
        },
        "curve_tau_p": [
            {"tau_p": float(t), **_curve_point(chosen, tau_v, float(t))}
            for t in np.round(np.arange(0.60, 1.0001, 0.01), 2)
        ],
        "curve_tau_v": [
            {"tau_v": float(t), **_curve_point(chosen, float(t), tau_p)}
            for t in np.round(np.arange(0.05, 1.0001, 0.05), 2)
        ],
    }


def _curve_point(pairs: pd.DataFrame, tau_v: float, tau_p: float) -> dict:
    """Weighted precision of the carried-over call and its recall at one pair of thresholds."""
    tier = predict(pairs["pair_j"], pairs["pair_cos"], tau_v, tau_p)
    weight, carried = pairs["weight"], pairs["label"] != "different"
    called = tier != "new"
    return {
        "precision": float(weight[called & carried].sum() / weight[called].sum())
        if called.any()
        else float("nan"),
        "recall": float(weight[called & carried].sum() / weight[carried].sum()),
        "share_called": float(weight[called].sum() / weight.sum()),
    }


def build_thresholds(cfg: Config, embedder: Embedder | None = None) -> Path:
    """Tune on the tuning half, evaluate on the other half and the real pairs, and save.

    Writes `results/thresholds.json` and `results/known_answer_scores.csv`
    (each known-answer pair's Jaccard and cosine, for the charts).
    """
    scores = score_known_answer(cfg, embedder)
    found = tune(scores)
    tau_v, tau_p = found["tau_v"], found["tau_p"]
    out = {
        **found,
        "tuned_on": {
            k: int(v) for k, v in scores[scores["split"] == "tune"]["kind"].value_counts().items()
        },
        "known_answer": evaluate_known(scores, tau_v, tau_p),
        "score_summary": {
            kind: {
                "j": [float(q) for q in np.quantile(part["j"], [0.1, 0.5, 0.9])],
                "cos": [float(q) for q in np.quantile(part["cos"], [0.1, 0.5, 0.9])],
            }
            for kind, part in scores.groupby("kind")
        },
    }
    if (cfg.validation_dir / "real_pairs" / "judge1_labels.csv").exists():
        out["real_pairs"] = evaluate_real(cfg, tau_v, tau_p)
    cfg.results_dir.mkdir(exist_ok=True)
    columns = ["id", "kind", "split", "j", "cos"]
    scores[columns].round(4).to_csv(cfg.results_dir / "known_answer_scores.csv", index=False)
    results_path(cfg).write_text(json.dumps(out, indent=1) + "\n")
    return results_path(cfg)


def read_thresholds(cfg: Config) -> tuple[float, float]:
    """(`tau_v`, `tau_p`) as last tuned."""
    saved = json.loads(results_path(cfg).read_text())
    return saved["tau_v"], saved["tau_p"]


def results_path(cfg: Config) -> Path:
    return cfg.results_dir / "thresholds.json"
