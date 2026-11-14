"""Score two sets of filings against each other in one call.

The pipeline scores the whole corpus in stages (candidates, embeddings,
similarity) so that each can be cached and resumed. The baselines need the
same scores for small, made-up pairings (a filing against a stranger's, a new
sentence against everything in last year's filing), where a single function is
simpler. `tests/test_baselines.py` checks the two give the same answer.
"""

import numpy as np
import pandas as pd

from pipeline.candidates import match_frames
from pipeline.embed import Embedder, Leftover, cosine_columns


def score_pairs(cur: pd.DataFrame, prev: pd.DataFrame, embedder: Embedder) -> pd.DataFrame:
    """The similarity table for `cur` against `prev`.

    Both have `cik`, `sent_idx`, `text`, `norm` and `norm_hash`; rows with the
    same `cik` are compared, so `cik` can be any key that names a pairing.
    Returns the columns of a `similarity/pair=YYYY.parquet` file.
    """
    df = match_frames(cur, prev)

    def leftover(side: str, frame: pd.DataFrame) -> Leftover:
        rows = np.flatnonzero(((df["side"] == side) & ~df["exact"]).to_numpy())
        found = df.iloc[rows][["cik", "sent_idx"]].merge(
            frame[["cik", "sent_idx", "norm_hash", "text"]], on=["cik", "sent_idx"], how="left"
        )
        distinct = found.drop_duplicates("norm_hash")
        vectors = embedder.encode(distinct["text"].tolist())
        vectors = vectors.astype(np.float16).astype(np.float32)  # as the stored vectors are
        position = pd.Series(np.arange(len(distinct)), index=distinct["norm_hash"].to_numpy())
        return rows, vectors[position.reindex(found["norm_hash"]).to_numpy(dtype=np.int64)]

    best_cos, cos_idx = cosine_columns(df, leftover("cur", cur), leftover("prev", prev))
    return df.assign(best_cos=best_cos, cos_idx=cos_idx)
