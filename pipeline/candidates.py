"""Match a filing's sentences against the same company's previous filing.

Two steps, neither of which needs a threshold:

1. Exact: a sentence whose normalized form is in the other filing.
2. Jaccard: for every sentence step 1 left over, its best word-shingle Jaccard
   against the other filing's leftovers, and which sentence that was.

A sentence may be matched by several on the other side: it is carried over if
it is anywhere in last year's filing, however many times it appears in this
one. Deciding which scores count as a match comes later, once the thresholds
have been validated.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences, write_parquet
from pipeline.pairs import consecutive_pairs
from pipeline.shingles import best_matches, shingles

CANDIDATE_COLUMNS = ["side", "sent_idx", "exact", "exact_idx", "best_j", "j_idx"]


def exact_matches(cur: pd.DataFrame, prev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `cur` and of `prev` (columns `cik`, `sent_idx`, `norm_hash`), the
    `sent_idx` of the first sentence in the same company's other filing with the same
    normalized form, or -1."""

    def first_on(other: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        first = (
            other.sort_values("sent_idx", kind="stable")
            .drop_duplicates(["cik", "norm_hash"])
            .rename(columns={"sent_idx": "other_idx"})[["cik", "norm_hash", "other_idx"]]
        )
        found = rows[["cik", "norm_hash"]].merge(first, on=["cik", "norm_hash"], how="left")
        return found["other_idx"].fillna(-1).to_numpy(dtype=np.int64)

    return first_on(prev, cur), first_on(cur, prev)


def jaccard_matches(cur: pd.DataFrame, prev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best Jaccard for each unmatched sentence against the other filing's unmatched ones.

    `cur` and `prev` hold only the sentences exact matching left over (columns
    `cik`, `sent_idx`, `norm`). Returns the two frames with `best_j` and
    `j_idx` added: 0.0 and -1 when nothing overlaps, NaN and -1 when the other
    filing has nothing left to compare with.
    """
    cur, prev = cur.copy(), prev.copy()
    cur_groups = cur.groupby("cik", sort=False).indices
    prev_groups = prev.groupby("cik", sort=False).indices
    cur_norm, prev_norm = cur["norm"].to_numpy(), prev["norm"].to_numpy()
    cur_idx, prev_idx = cur["sent_idx"].to_numpy(), prev["sent_idx"].to_numpy()
    out = {
        "cur": (np.full(len(cur), np.nan), np.full(len(cur), -1, dtype=np.int64)),
        "prev": (np.full(len(prev), np.nan), np.full(len(prev), -1, dtype=np.int64)),
    }
    for cik, c_rows in cur_groups.items():
        p_rows = prev_groups.get(cik)
        if p_rows is None:
            continue
        c_sets = [shingles(n) for n in cur_norm[c_rows]]
        p_sets = [shingles(n) for n in prev_norm[p_rows]]
        for side, rows, mine, theirs, their_idx in (
            ("cur", c_rows, c_sets, p_sets, prev_idx[p_rows]),
            ("prev", p_rows, p_sets, c_sets, cur_idx[c_rows]),
        ):
            scores, where = best_matches(mine, theirs)
            out[side][0][rows] = scores
            out[side][1][rows] = np.where(where >= 0, their_idx[np.maximum(where, 0)], -1)
    cur["best_j"], cur["j_idx"] = out["cur"]
    prev["best_j"], prev["j_idx"] = out["prev"]
    return cur, prev


def match_frames(cur: pd.DataFrame, prev: pd.DataFrame, jaccard=jaccard_matches) -> pd.DataFrame:
    """Candidates for many companies at once.

    `cur` and `prev` are sentence rows (`cik`, `sent_idx`, `norm`, `norm_hash`)
    for one fiscal year and the year before. Returns one row per sentence of
    both: `cik`, then CANDIDATE_COLUMNS. `jaccard` can be swapped for a version
    that spreads companies over processes.
    """
    sides = {"cur": cur.reset_index(drop=True), "prev": prev.reset_index(drop=True)}
    exact_cur, exact_prev = exact_matches(sides["cur"], sides["prev"])
    for side, exact_idx in (("cur", exact_cur), ("prev", exact_prev)):
        sides[side] = sides[side].assign(side=side, exact=exact_idx >= 0, exact_idx=exact_idx)
    left = {s: f.loc[~f["exact"], ["cik", "sent_idx", "norm"]] for s, f in sides.items()}
    scored = dict(zip(("cur", "prev"), jaccard(left["cur"], left["prev"]), strict=True))
    frames = []
    for side, frame in sides.items():
        frame = frame.assign(best_j=np.nan, j_idx=-1)
        frame.loc[scored[side].index, "best_j"] = scored[side]["best_j"]
        frame.loc[scored[side].index, "j_idx"] = scored[side]["j_idx"]
        frames.append(frame[["cik", *CANDIDATE_COLUMNS]])
    return pd.concat(frames, ignore_index=True)


def match_pair(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Candidates for one company's pair: `cur` and `prev` are its sentence rows
    (`sent_idx`, `norm`, `norm_hash`) for a fiscal year and the year before."""
    return match_frames(cur.assign(cik=0), prev.assign(cik=0))[CANDIDATE_COLUMNS]


def _jaccard_batch(batch: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return jaccard_matches(*batch)


def _pooled_jaccard(
    cur: pd.DataFrame, prev: pd.DataFrame, pool, n_batches: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`jaccard_matches`, with companies spread over the pool's processes."""
    batches = [
        (cur[cur["cik"] % n_batches == b], prev[prev["cik"] % n_batches == b])
        for b in range(n_batches)
    ]
    done = list(pool.map(_jaccard_batch, batches))
    return (
        pd.concat([c for c, _ in done]).loc[cur.index],
        pd.concat([p for _, p in done]).loc[prev.index],
    )


def pair_years(cfg: Config) -> dict[int, np.ndarray]:
    """For each fiscal year with pairs, the companies that also filed the year before."""
    filings = pd.read_parquet(cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year"])
    pairs = consecutive_pairs(filings)
    return {int(y): g["cik"].to_numpy() for y, g in pairs.groupby("filing_year")}


def build_candidates(cfg: Config, workers: int = 8, force: bool = False) -> Path:
    """Exact matches and best Jaccard for every consecutive pair of filings.

    Writes `candidates/pair=YYYY.parquet`, one per fiscal year: the rows of
    `match_frames` for that year's filings against the year before.
    """
    out_dir = cfg.interim_dir / "candidates"
    columns = ["cik", "sent_idx", "norm", "norm_hash"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jaccard = partial(_pooled_jaccard, pool=pool, n_batches=workers * 8)
        for year, ciks in pair_years(cfg).items():
            out = out_dir / f"pair={year}.parquet"
            if out.exists() and not force:
                continue
            cur = read_sentences(cfg, year, columns)
            prev = read_sentences(cfg, year - 1, columns)
            result = match_frames(
                cur[cur["cik"].isin(ciks)], prev[prev["cik"].isin(ciks)], jaccard=jaccard
            )
            write_parquet(result, out)
    return out_dir


def summarize_candidates(candidates_dir: Path) -> str:
    """Per fiscal year: pairs, sentences on each side, and how many matched exactly."""
    header = (
        f"{'year':<6}{'pairs':>7}{'sentences':>12}{'exact':>8}{'unmatched':>11}"
        f"{'previous':>12}{'exact':>8}{'unmatched':>11}"
    )
    lines = [header]
    total = np.zeros(6, dtype=np.int64)
    for path in sorted(candidates_dir.glob("pair=*.parquet")):
        df = pd.read_parquet(path, columns=["cik", "side", "exact"])
        cur, prev = df[df["side"] == "cur"], df[df["side"] == "prev"]
        row = np.array(
            [cur["cik"].nunique(), len(cur), cur["exact"].sum(), len(prev), prev["exact"].sum(), 0]
        )
        total += row
        lines.append(_candidate_line(path.stem.split("=")[1], row))
    lines.append(_candidate_line("all", total))
    return "\n".join(lines)


def _candidate_line(label: str, row: np.ndarray) -> str:
    pairs, n_cur, exact_cur, n_prev, exact_prev, _ = row
    return (
        f"{label:<6}{pairs:>7,}{n_cur:>12,}{exact_cur / n_cur:>8.1%}{n_cur - exact_cur:>11,}"
        f"{n_prev:>12,}{exact_prev / n_prev:>8.1%}{n_prev - exact_prev:>11,}"
    )
