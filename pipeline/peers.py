"""How many unrelated companies file the same sentence in the same year.

Boilerplate travels between companies: law firms reuse language, and filers
borrow from each other. Counting companies overstates that, because some
filers are the same business filing more than once (a parent and its
subsidiaries, a family of funds) and their sections are near-copies of each
other. So companies whose sections mostly coincide in a year are put in one
group, and a sentence's reach is the number of groups it appears in.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pipeline.config import Config
from pipeline.io import read_sentences, sentence_years, write_parquet
from pipeline.shingles import lsh_candidates, minhash, shingle_hashes

GROUP_THRESHOLD = 0.5
PEER_COLUMNS = ["norm_hash", "n_companies", "n_groups"]


def filer_groups(sentences: pd.DataFrame, threshold: float = GROUP_THRESHOLD) -> pd.Series:
    """Group companies whose sections mostly coincide in one fiscal year.

    `sentences` has `cik` and `norm_hash` for one year. Two companies are
    joined when the Jaccard of their sets of distinct sentences is at least
    `threshold`, and groups are the connected components. Returns a Series
    from `cik` to its group, named for the group's lowest CIK.
    """
    distinct = sentences[["cik", "norm_hash"]].drop_duplicates()
    company, ciks = pd.factorize(distinct["cik"], sort=True)
    sentence, _ = pd.factorize(distinct["norm_hash"])
    has = sp.csr_matrix((np.ones(len(distinct), dtype=np.int32), (company, sentence)))
    sizes = np.asarray(has.sum(axis=1)).ravel()
    shared = (has @ has.T).tocoo()
    above = shared.row < shared.col
    a, b, n = shared.row[above], shared.col[above], shared.data[above]
    joined = n / (sizes[a] + sizes[b] - n) >= threshold
    graph = sp.csr_matrix((np.ones(joined.sum()), (a[joined], b[joined])), shape=(len(ciks),) * 2)
    _, component = connected_components(graph, directed=False)
    lowest = pd.Series(ciks).groupby(component).min()
    return pd.Series(lowest[component].to_numpy(), index=pd.Index(ciks, name="cik"), name="group")


def peer_counts(sentences: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """For each distinct sentence of one year: the companies and the groups that file it."""
    with_group = sentences[["cik", "norm_hash"]].assign(group=sentences["cik"].map(groups))
    counts = with_group.groupby("norm_hash").agg(
        n_companies=("cik", "nunique"), n_groups=("group", "nunique")
    )
    return counts.reset_index()[PEER_COLUMNS]


def near_column(threshold: float) -> str:
    return f"near_others_{round(threshold * 100):03d}"


def near_peer_counts(
    sentences: pd.DataFrame, groups: pd.Series, queries: np.ndarray, thresholds: tuple[float, ...]
) -> pd.DataFrame:
    """For chosen sentences, how many other groups file it word for word, and how many
    file it word for word or nearly so.

    `sentences` has `cik`, `norm` and `norm_hash` for one year; `queries` are
    row labels in it. "Nearly" is a Jaccard between shingle sets of at least a
    threshold, estimated from MinHash signatures, with candidates found by
    banding. Returns one row per query: `exact_others`, and for each
    threshold a column named by `near_column`.
    """
    group = sentences["cik"].map(groups).to_numpy()
    row, _ = pd.factorize(sentences["norm_hash"])
    first = pd.Series(np.arange(len(sentences))).groupby(row).first().to_numpy()
    signatures = minhash(*shingle_hashes(sentences["norm"].to_numpy()[first]))
    holders = pd.Series(group).groupby(row).unique()  # distinct sentence -> its groups

    at = sentences.index.get_indexer(queries)
    found = lsh_candidates(signatures, row[at])
    counts: dict[str, list[int]] = {"exact_others": []}
    counts |= {near_column(t): [] for t in thresholds}
    for position, candidates in zip(at, found, strict=True):
        mine, own = row[position], group[position]
        exact = set(holders[mine]) - {own}
        counts["exact_others"].append(len(exact))
        agreement = (signatures[candidates] == signatures[mine]).mean(axis=1)
        for threshold in thresholds:
            others = set(exact)
            for candidate in candidates[agreement >= threshold]:
                others.update(holders[candidate])
            others.discard(own)
            counts[near_column(threshold)].append(len(others))
    return pd.DataFrame(counts, index=queries)


def build_peers(cfg: Config, force: bool = False) -> Path:
    """Groups and per-sentence reach for every fiscal year.

    Writes `peers/year=YYYY.parquet` (PEER_COLUMNS) and `filer_groups.parquet`
    (`cik`, `filing_year`, `group`).
    """
    out_dir = cfg.interim_dir / "peers"
    groups_path = cfg.interim_dir / "filer_groups.parquet"
    years = sentence_years(cfg)
    if (
        groups_path.exists()
        and not force
        and all((out_dir / f"year={y}.parquet").exists() for y in years)
    ):
        return out_dir
    all_groups = []
    for year in years:
        sentences = read_sentences(cfg, year, ["cik", "norm_hash"])
        groups = filer_groups(sentences)
        write_parquet(peer_counts(sentences, groups), out_dir / f"year={year}.parquet")
        all_groups.append(groups.reset_index().assign(filing_year=year))
    write_parquet(pd.concat(all_groups)[["cik", "filing_year", "group"]], groups_path)
    return out_dir


def near_peer_sample(
    cfg: Config, thresholds: tuple[float, ...], n_per_year: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """`near_peer_counts` for a random sample of each year's sentences.

    Near-verbatim reach needs every pair of a year's 1.3 million distinct
    sentences considered, so it is estimated on a sample and reported as a
    sensitivity; the exact count is computed for every sentence.
    """
    groups = pd.read_parquet(cfg.interim_dir / "filer_groups.parquet")
    parts = []
    for year in sentence_years(cfg):
        sentences = read_sentences(cfg, year, ["cik", "norm", "norm_hash", "is_heading"])
        queries = sentences.sample(n_per_year, random_state=seed).index.to_numpy()
        in_year = groups[groups["filing_year"] == year].set_index("cik")["group"]
        counts = near_peer_counts(sentences, in_year, queries, thresholds)
        words = sentences.loc[queries, "norm"].str.count(" ") + 1
        parts.append(
            counts.assign(
                filing_year=year, n_words=words, is_heading=sentences.loc[queries, "is_heading"]
            )
        )
    return pd.concat(parts, ignore_index=True)
