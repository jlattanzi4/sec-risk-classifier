"""What the numbers look like when there is nothing to find.

A share of sentences "carried over" means little without knowing what the same
method reports for text that was not carried over. Four checks:

- `random_pairs`: two sentences from different companies and years. How often
  do the thresholds call them a match? (The false-match rate of one comparison.)
- `stranger_filings`: a filing matched against another company's previous
  filing in place of its own. (The false-match rate of the whole pipeline,
  where each sentence is compared with a filing's worth of candidates.)
- `shuffled_words`: a sentence against itself with its words in random order.
  (What the embedding responds to: the words, or what they say?)
- `unmatched_only_check`: sentences called new, compared with everything in
  last year's filing and not only with what exact matching left over. (What
  the staged design misses.)
"""

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.embed import Embedder
from pipeline.io import read_sentences, sentence_years
from pipeline.labels import CURRENT_LABELS, label
from pipeline.matching import score_pairs
from pipeline.pairs import consecutive_pairs
from pipeline.sentences import normalize
from pipeline.shingles import jaccard, shingles
from pipeline.thresholds import predict

MIN_WORDS = 8
SCORED = ["cik", "sent_idx", "text", "norm", "norm_hash"]


def _full_sentences(cfg: Config, year: int, n: int, seed: int) -> pd.DataFrame:
    """`n` random full sentences of a fiscal year (not headings, at least MIN_WORDS words)."""
    df = read_sentences(cfg, year, ["cik", "filing_year", "text", "norm", "is_heading"])
    df = df[~df["is_heading"] & (df["norm"].str.count(" ") + 1 >= MIN_WORDS)]
    return df.sample(min(n, len(df)), random_state=seed)[["cik", "filing_year", "text"]]


def pair_up(pool: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """`n` pairs of rows of `pool` from different companies and different years.

    Returns the columns of `pool` twice, suffixed `_a` and `_b`.
    """
    shuffled = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    half = len(shuffled) // 2
    a = shuffled.iloc[:half].reset_index(drop=True).add_suffix("_a")
    b = shuffled.iloc[half : 2 * half].reset_index(drop=True).add_suffix("_b")
    pairs = pd.concat([a, b], axis=1)
    apart = (pairs["cik_a"] != pairs["cik_b"]) & (pairs["filing_year_a"] != pairs["filing_year_b"])
    return pairs[apart].head(n).reset_index(drop=True)


def shuffle_words(text: str, rng: np.random.Generator) -> str:
    """The same words in a random order (never the original order, if there is another)."""
    words = text.split()
    for _ in range(10):
        mixed = [words[i] for i in rng.permutation(len(words))]
        if mixed != words:
            break
    return " ".join(mixed)


def _cosine(embedder: Embedder, a: list[str], b: list[str]) -> np.ndarray:
    left = embedder.encode(a).astype(np.float16).astype(np.float32)
    right = embedder.encode(b).astype(np.float16).astype(np.float32)
    return (left * right).sum(axis=1)


def _jaccard(a: list[str], b: list[str]) -> np.ndarray:
    return np.array(
        [jaccard(shingles(normalize(x)), shingles(normalize(y))) for x, y in zip(a, b, strict=True)]
    )


def _quartiles(values: np.ndarray) -> dict:
    low, mid, high = np.quantile(values, [0.25, 0.5, 0.75])
    return {"p25": float(low), "median": float(mid), "p75": float(high)}


def random_pairs(
    cfg: Config, tau_v: float, tau_p: float, embedder: Embedder, n: int = 50_000, seed: int = 0
) -> dict:
    """Sentence pairs from different companies and years, and how the thresholds label them."""
    years = sentence_years(cfg)
    per_year = int(2.5 * n / len(years)) + 1  # some pairs are dropped for sharing a year
    pool = pd.concat(_full_sentences(cfg, year, per_year, seed) for year in years)
    pairs = pair_up(pool, n, seed)
    a, b = pairs["text_a"].tolist(), pairs["text_b"].tolist()
    j, cos = _jaccard(a, b), _cosine(embedder, a, b)
    tier = predict(pd.Series(j), pd.Series(cos), tau_v, tau_p)
    return {
        "n": len(pairs),
        "share_near": float((tier == "near").mean()),
        "share_paraphrase": float((tier == "paraphrase").mean()),
        "n_near": int((tier == "near").sum()),
        "n_paraphrase": int((tier == "paraphrase").sum()),
        "cosine": _quartiles(cos),
        "cosine_p99": float(np.quantile(cos, 0.99)),
        "cosine_max": float(cos.max()),
    }


def shuffled_words(
    cfg: Config, tau_v: float, tau_p: float, embedder: Embedder, n: int = 20_000, seed: int = 0
) -> dict:
    """Each sentence against itself with its words shuffled."""
    years = sentence_years(cfg)
    pool = pd.concat(_full_sentences(cfg, year, n // len(years) + 1, seed) for year in years)
    texts = pool["text"].head(n).tolist()
    rng = np.random.default_rng(seed)
    mixed = [shuffle_words(t, rng) for t in texts]
    j, cos = _jaccard(texts, mixed), _cosine(embedder, texts, mixed)
    return {
        "n": len(texts),
        "cosine": _quartiles(cos),
        "share_cosine_at_or_above_tau_p": float((cos >= tau_p).mean()),
        "jaccard": _quartiles(j),
        "share_jaccard_at_or_above_tau_v": float((j >= tau_v).mean()),
    }


def _label_shares(scores: pd.DataFrame, tau_v: float, tau_p: float) -> dict:
    """Shares of current-side sentences with each label, pooled over the pairs scored."""
    current = scores[scores["side"] == "cur"]
    labels = label(current, tau_v, tau_p)["label"]
    return {name: float((labels == name).mean()) for name in CURRENT_LABELS} | {"n": len(current)}


def related_filers(filings: pd.DataFrame, cik: int, year: int) -> set[int]:
    """The company and every filer grouped with it in `year` or the year before.

    Groups are found a year at a time and named for their lowest CIK, so a
    group's name changes when its members do. Being in the company's group in
    either year is what makes a filer kin, whatever the group was called.
    """
    both = filings[filings["filing_year"].isin([year, year - 1])]
    own = both.loc[both["cik"] == cik, ["filing_year", "group"]]
    return set(both.merge(own, on=["filing_year", "group"])["cik"])


def stranger_filings(
    cfg: Config, tau_v: float, tau_p: float, embedder: Embedder, n_pairs: int = 500, seed: int = 0
) -> dict:
    """Filings matched against a stranger's previous filing, next to their own.

    For `n_pairs` sampled filings: the label shares against the company's own
    previous filing (what the pipeline reports), against the previous-year
    filing of a random unrelated company, and against one from a random
    unrelated company in the same SIC division. A filing with no unrelated
    company in its division that year is left out; `n_filings` is how many
    were scored, from `n_companies` companies.
    """
    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year", "sic_division"]
    )
    groups = pd.read_parquet(cfg.interim_dir / "filer_groups.parquet")
    filings = filings.merge(groups, on=["cik", "filing_year"])
    sample = consecutive_pairs(filings).sample(n_pairs, random_state=seed)
    sample = sample.merge(filings, on=["cik", "filing_year"])
    rng = np.random.default_rng(seed)

    kinds = ("own", "stranger", "stranger_same_division")
    scores: dict[str, list[pd.DataFrame]] = {kind: [] for kind in kinds}
    for year, rows in sample.groupby("filing_year"):
        current = read_sentences(cfg, year, SCORED)
        previous = read_sentences(cfg, year - 1, SCORED)
        last_year = filings[filings["filing_year"] == year - 1]
        cur_parts = []
        prev_parts: dict[str, list[pd.DataFrame]] = {kind: [] for kind in kinds}
        for row in rows.itertuples():
            kin = related_filers(filings, row.cik, year)
            unrelated = last_year[~last_year["cik"].isin(kin)]
            same_division = unrelated[unrelated["sic_division"] == row.sic_division]
            if same_division.empty:
                continue
            chosen = {
                "own": row.cik,
                "stranger": int(rng.choice(unrelated["cik"].to_numpy())),
                "stranger_same_division": int(rng.choice(same_division["cik"].to_numpy())),
            }
            cur_parts.append(current[current["cik"] == row.cik])
            for kind, cik in chosen.items():
                prev_parts[kind].append(previous[previous["cik"] == cik].assign(cik=row.cik))
        if not cur_parts:
            continue
        cur = pd.concat(cur_parts)
        for kind in kinds:
            scores[kind].append(score_pairs(cur, pd.concat(prev_parts[kind]), embedder))
    out = {kind: _label_shares(pd.concat(parts), tau_v, tau_p) for kind, parts in scores.items()}
    out["n_filings"] = int(sum(part["cik"].nunique() for part in scores["own"]))
    out["n_companies"] = int(pd.concat(scores["own"])["cik"].nunique()) if scores["own"] else 0
    return out


def unmatched_only_check(
    cfg: Config, tau_v: float, tau_p: float, embedder: Embedder, n_pairs: int = 500, seed: int = 1
) -> dict:
    """How many sentences called new would not be, compared with all of last year's filing.

    The pipeline compares a leftover sentence only with last year's leftovers.
    A new sentence that restates one which was also carried over word for
    word (a summary of an existing risk, say) finds nothing there. Here the
    new sentences of sampled filings are scored against every sentence of the
    previous filing.
    """
    filings = pd.read_parquet(cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year"])
    pairs = consecutive_pairs(filings)
    sample = pairs.sample(min(n_pairs, len(pairs)), random_state=seed)
    parts, n_pairs_used = [], 0
    for year, rows in sample.groupby("filing_year"):
        labels = pd.read_parquet(cfg.interim_dir / "labels" / f"pair={year}.parquet")
        new = labels[(labels["side"] == "cur") & (labels["label"] == "new")]
        new = new[new["cik"].isin(rows["cik"])][["cik", "sent_idx"]]
        if new.empty:
            continue
        current = read_sentences(cfg, year, SCORED).merge(new, on=["cik", "sent_idx"])
        previous = read_sentences(cfg, year - 1, SCORED)
        previous = previous[previous["cik"].isin(new["cik"])]
        scores = score_pairs(current, previous, embedder)
        parts.append(label(scores[scores["side"] == "cur"], tau_v, tau_p)["label"])
        n_pairs_used += new["cik"].nunique()
    relabeled = pd.concat(parts) if parts else pd.Series(dtype=str)
    counts = relabeled.value_counts()
    n = len(relabeled)
    moved = n - int(counts.get("new", 0))
    return {
        "pairs": n_pairs_used,
        "new_sentences": n,
        "would_be_exact": int(counts.get("exact", 0)),
        "would_be_near": int(counts.get("near", 0)),
        "would_be_paraphrase": int(counts.get("paraphrase", 0)),
        "share_relabeled": moved / n if n else float("nan"),
    }
