"""Word shingles, exact Jaccard, and MinHash.

Two sentences are near-verbatim copies when they share most of their runs of
five consecutive words. Inside one company's pair of filings the candidates are
few, so Jaccard is computed exactly (`best_matches`). Across a whole year's
sentences it can't be, so each sentence is reduced to a MinHash signature whose
agreement with another signature estimates their Jaccard, and banding finds the
candidates worth comparing (`minhash`, `lsh_candidates`).
"""

from collections import defaultdict
from collections.abc import Sequence
from hashlib import blake2b

import numpy as np
import pandas as pd

K = 5
_MULTIPLIER = 0x9E3779B97F4A7C15  # odd, so multiplying by it mod 2**64 loses nothing
_MASK = 2**64 - 1
_P = np.uint64(_MULTIPLIER)


def shingles(norm: str, k: int = K) -> frozenset[str]:
    """The runs of `k` consecutive words in a normalized sentence.

    A sentence shorter than `k` words is one shingle, the whole sentence, so it
    can match an identical sentence and nothing else.
    """
    words = norm.split()
    if not words:
        return frozenset()
    if len(words) < k:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared shingles over all shingles; 0.0 when both are empty."""
    if not a or not b:
        return 0.0
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


def best_matches(
    left: list[frozenset[str]], right: list[frozenset[str]]
) -> tuple[np.ndarray, np.ndarray]:
    """For each left set, its best Jaccard against any right set, and where.

    Returns (scores, positions); a left set that shares nothing has score 0.0
    and position -1. An inverted index over `right` means only sets that share
    a shingle are ever compared.
    """
    scores = np.zeros(len(left), dtype=np.float64)
    where = np.full(len(left), -1, dtype=np.int64)
    index: dict[str, list[int]] = defaultdict(list)
    for j, b in enumerate(right):
        for shingle in b:
            index[shingle].append(j)
    sizes = [len(b) for b in right]
    for i, a in enumerate(left):
        shared: dict[int, int] = defaultdict(int)
        for shingle in a:
            for j in index.get(shingle, ()):
                shared[j] += 1
        for j, n in shared.items():
            score = n / (len(a) + sizes[j] - n)
            if score > scores[i]:
                scores[i], where[i] = score, j
    return scores, where


def _word_hash(word: str) -> int:
    return int.from_bytes(blake2b(word.encode(), digest_size=8).digest(), "little")


def shingle_hashes(norms: Sequence[str], k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """64-bit hashes of every sentence's shingles, as one flat array.

    Returns (hashes, starts): sentence i's shingles are
    `hashes[starts[i]:starts[i + 1]]` (to the end for the last). The hash of a
    shingle depends only on its words, never on the process or on what else is
    in `norms`, so signatures built in separate runs can be compared.
    """
    token_lists = [norm.split() for norm in norms]
    counts = np.fromiter((len(t) for t in token_lists), dtype=np.int64, count=len(token_lists))
    n_shingles = np.where(counts >= k, counts - k + 1, (counts > 0).astype(np.int64))
    starts = np.concatenate([[0], np.cumsum(n_shingles)[:-1]]) if len(counts) else counts
    total = int(n_shingles.sum())
    if total == 0:
        return np.zeros(0, dtype=np.uint64), starts.astype(np.int64)

    codes, vocabulary = pd.factorize(np.array([w for t in token_lists for w in t], dtype=object))
    word_hashes = np.fromiter((_word_hash(w) for w in vocabulary), dtype=np.uint64)
    tokens = word_hashes[codes]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # Hash of the window starting at each token: sum of word_hash * P**position.
    padded = np.concatenate([tokens, np.zeros(k, dtype=np.uint64)])
    window = np.zeros(len(tokens), dtype=np.uint64)
    for j in range(k):
        window += padded[j : j + len(tokens)] * np.uint64(pow(_MULTIPLIER, j, 2**64))

    sentence = np.repeat(np.arange(len(counts)), n_shingles)
    position = offsets[sentence] + (np.arange(total) - starts[sentence])
    hashes = window[position]

    # A short sentence's single shingle must cover only its own words, not run
    # on into the next sentence's.
    for i in np.flatnonzero((counts > 0) & (counts < k)):
        value = int(counts[i])
        for j, token in enumerate(tokens[offsets[i] : offsets[i] + counts[i]].tolist(), start=1):
            value = (value + token * pow(_MULTIPLIER, j, 2**64)) & _MASK
        hashes[starts[i]] = value
    return hashes, starts.astype(np.int64)


def minhash(hashes: np.ndarray, starts: np.ndarray, n_perm: int = 126, seed: int = 0) -> np.ndarray:
    """MinHash signatures, one row per sentence (`uint32`, `n_perm` columns).

    The share of positions where two rows agree estimates the Jaccard of their
    shingle sets. A sentence with no shingles gets a row of its own that agrees
    with nothing.
    """
    n = len(starts)
    signatures = np.empty((n, n_perm), dtype=np.uint32)
    ends = np.append(starts[1:], len(hashes))
    filled = ends > starts
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63, size=n_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**63, size=n_perm, dtype=np.uint64)
    if filled.any():
        for i in range(n_perm):
            permuted = ((hashes * a[i] + b[i]) >> np.uint64(32)).astype(np.uint32)
            signatures[filled, i] = np.minimum.reduceat(permuted, starts[filled])
    # Distinct values per empty row, so two empty sentences never look alike.
    empty = np.flatnonzero(~filled)
    signatures[empty] = (np.arange(len(empty), dtype=np.uint32)[:, None] * 2 + 1) * np.arange(
        1, n_perm + 1, dtype=np.uint32
    )
    return signatures


def lsh_candidates(
    signatures: np.ndarray,
    queries: np.ndarray,
    bands: int = 42,
    rows: int = 3,
    max_bucket: int = 2000,
) -> list[np.ndarray]:
    """For each query row, the other rows that agree with it on at least one band.

    With 42 bands of 3, a pair with Jaccard 0.4 is found 94% of the time and
    one with Jaccard 0.5, 99.6%. A bucket larger than `max_bucket` (a phrase
    half the corpus uses) is sampled down to that size.
    """
    assert signatures.shape[1] >= bands * rows
    rng = np.random.default_rng(0)
    found: list[list[np.ndarray]] = [[] for _ in queries]
    for band in range(bands):
        key = np.zeros(len(signatures), dtype=np.uint64)
        for column in range(band * rows, (band + 1) * rows):
            key = key * _P + signatures[:, column].astype(np.uint64)
        order = np.argsort(key, kind="stable")
        ordered = key[order]
        low = np.searchsorted(ordered, key[queries], side="left")
        high = np.searchsorted(ordered, key[queries], side="right")
        for q, (lo, hi) in enumerate(zip(low, high, strict=True)):
            members = order[lo:hi]
            if len(members) > max_bucket:
                members = rng.choice(members, size=max_bucket, replace=False)
            found[q].append(members)
    out = []
    for query, parts in zip(queries, found, strict=True):
        rows_found = np.unique(np.concatenate(parts))
        out.append(rows_found[rows_found != query])
    return out
