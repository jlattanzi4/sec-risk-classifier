"""Sentence embeddings, for telling a reworded sentence from a new one.

`MODEL` is the model the pipeline uses. It was chosen by test, on the half of
the known-answer set kept for tuning (`scripts/compare_embedders.py`), not by
default.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences, write_parquet

MODEL = "BAAI/bge-small-en-v1.5"
CANDIDATES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "minishlab/potion-base-8M",  # static: word vectors averaged, no attention
)


class Embedder:
    """A model that turns sentences into unit vectors."""

    def __init__(self, model, name: str, dim: int | None = None, max_tokens: int | None = None):
        self.model, self.name, self.dim, self.max_tokens = model, name, dim, max_tokens

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """One L2-normalized `float32` row per text, in the order given.

        Texts are embedded longest first, so each batch holds sentences of
        similar length and little of it is padding.
        """
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        order = np.argsort([-len(t) for t in texts], kind="stable")
        vectors = np.asarray(self.model.encode([texts[i] for i in order], batch_size=batch_size))
        vectors = vectors.astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        out = np.empty_like(vectors)
        out[order] = vectors
        return out

    def over_limit(self, texts: list[str]) -> float:
        """Share of texts longer than the model can read (they would be cut off)."""
        if self.max_tokens is None:
            return 0.0
        lengths = self.model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return float(np.mean([len(ids) > self.max_tokens for ids in lengths["input_ids"]]))


class _SentenceTransformer:
    def __init__(self, name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        on_gpu = torch.backends.mps.is_available()
        self.inner = SentenceTransformer(name, device="mps" if on_gpu else "cpu")
        if on_gpu:
            # Half precision is about 1.3 times faster here and moves a cosine by
            # under 0.001, less than the float16 the vectors are stored in.
            self.inner.half()
        self.tokenizer = self.inner.tokenizer

    def encode(self, texts, batch_size):
        return self.inner.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )


class _Static:
    def __init__(self, name: str):
        from model2vec import StaticModel

        self.inner = StaticModel.from_pretrained(name)

    def encode(self, texts, batch_size):
        return self.inner.encode(texts)


def load_embedder(name: str = MODEL) -> Embedder:
    """Load a candidate by its Hugging Face name. Uses the Apple GPU when there is one."""
    if name.startswith("minishlab/"):
        model = _Static(name)
        return Embedder(model, name, dim=model.inner.dim, max_tokens=None)
    model = _SentenceTransformer(name)
    dim = model.inner.get_embedding_dimension()
    return Embedder(model, name, dim=dim, max_tokens=model.inner.max_seq_length)


def best_cosine(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `left`, its highest cosine against any row of `right`, and where.

    Rows are unit vectors. With no rows in `right`, scores are NaN and positions -1.
    """
    if len(right) == 0 or len(left) == 0:
        return np.full(len(left), np.nan), np.full(len(left), -1, dtype=np.int64)
    scores = left @ right.T
    where = scores.argmax(axis=1)
    return scores[np.arange(len(left)), where].astype(np.float64), where.astype(np.int64)


def _unmatched(cfg: Config) -> dict[int, pd.DataFrame]:
    """For each fiscal year, the sentences (`cik`, `sent_idx`) that exact matching left
    over in any pair: as this year's side of one pair or last year's side of the next."""
    by_year: dict[int, list[pd.DataFrame]] = {}
    for path in sorted((cfg.interim_dir / "candidates").glob("pair=*.parquet")):
        year = int(path.stem.split("=")[1])
        df = pd.read_parquet(path, columns=["cik", "side", "sent_idx", "exact"])
        df = df[~df["exact"]]
        for side, sentence_year in (("cur", year), ("prev", year - 1)):
            rows = df.loc[df["side"] == side, ["cik", "sent_idx"]]
            by_year.setdefault(sentence_year, []).append(rows)
    return {year: pd.concat(parts).drop_duplicates() for year, parts in sorted(by_year.items())}


def build_embeddings(cfg: Config, force: bool = False, embedder: Embedder | None = None) -> Path:
    """Embed every sentence exact matching left over, once per distinct sentence per year.

    For each fiscal year writes `embeddings/year=YYYY.npy` (`float16`, one row
    per distinct normalized sentence) and `year=YYYY.parquet` (`norm_hash`,
    `row`). A year's two files are written through temporary names, the
    vectors last, so a year counts as done only when both are whole; a rerun
    skips finished years and redoes an interrupted one.
    """
    out_dir = cfg.interim_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year, rows in _unmatched(cfg).items():
        vectors_path, index_path = out_dir / f"year={year}.npy", out_dir / f"year={year}.parquet"
        if vectors_path.exists() and index_path.exists() and not force:
            continue
        sentences = read_sentences(cfg, year, ["cik", "sent_idx", "norm_hash", "text"])
        sentences = sentences.merge(rows, on=["cik", "sent_idx"])
        distinct = sentences.sort_values(["cik", "sent_idx"], kind="stable").drop_duplicates(
            "norm_hash"
        )
        embedder = embedder or load_embedder()
        start = time.perf_counter()
        vectors = embedder.encode(distinct["text"].tolist()).astype(np.float16)
        elapsed = time.perf_counter() - start
        print(
            f"  {year}: {len(distinct):,} sentences, {len(distinct) / elapsed:,.0f} a second",
            flush=True,
        )

        index = pd.DataFrame(
            {"norm_hash": distinct["norm_hash"].to_numpy(), "row": range(len(distinct))}
        )
        write_parquet(index, index_path)
        part = vectors_path.with_name(vectors_path.name + ".part")
        with part.open("wb") as handle:
            np.save(handle, vectors)
        part.replace(vectors_path)
    return out_dir


def _year_vectors(cfg: Config, year: int) -> tuple[pd.Series, np.ndarray]:
    """A fiscal year's embeddings: `norm_hash` -> row, and the rows as `float32`."""
    folder = cfg.interim_dir / "embeddings"
    index = pd.read_parquet(folder / f"year={year}.parquet").set_index("norm_hash")["row"]
    return index, np.load(folder / f"year={year}.npy").astype(np.float32)


def _leftover(cfg: Config, df: pd.DataFrame, side: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """One side's sentences that exact matching left over: their row numbers in `df`, and
    their vectors in the same order."""
    rows = np.flatnonzero(((df["side"] == side) & ~df["exact"]).to_numpy())
    hashes = read_sentences(cfg, year, ["cik", "sent_idx", "norm_hash"])
    hashes = df.iloc[rows][["cik", "sent_idx"]].merge(hashes, on=["cik", "sent_idx"], how="left")
    index, vectors = _year_vectors(cfg, year)
    return rows, vectors[index.reindex(hashes["norm_hash"]).to_numpy(dtype=np.int64)]


def _by_company(ciks: np.ndarray) -> dict[int, np.ndarray]:
    """Positions of each company's entries in `ciks`."""
    return pd.Series(np.arange(len(ciks))).groupby(ciks).indices


Leftover = tuple[np.ndarray, np.ndarray]  # row numbers in a candidates table, and their vectors


def cosine_columns(
    df: pd.DataFrame, cur: Leftover, prev: Leftover
) -> tuple[np.ndarray, np.ndarray]:
    """`best_cos` and `cos_idx` for every row of a candidates table.

    `cur` and `prev` give, for each side, the rows exact matching left over
    and their vectors. Within each company, every leftover sentence gets its
    highest cosine against the other side's leftovers and that sentence's
    `sent_idx`. Other rows, and rows with nothing to compare with, get NaN
    and -1.
    """
    ciks, sent_idx = df["cik"].to_numpy(), df["sent_idx"].to_numpy()
    best_cos = np.full(len(df), np.nan)
    cos_idx = np.full(len(df), -1, dtype=np.int64)
    (cur_rows, cur_vectors), (prev_rows, prev_vectors) = cur, prev
    prev_of = _by_company(ciks[prev_rows])
    for cik, cur_at in _by_company(ciks[cur_rows]).items():
        prev_at = prev_of.get(cik)
        if prev_at is None:
            continue  # last year's filing has nothing left over to compare with
        for rows, at, vectors, other_rows, other_at, other_vectors in (
            (cur_rows, cur_at, cur_vectors, prev_rows, prev_at, prev_vectors),
            (prev_rows, prev_at, prev_vectors, cur_rows, cur_at, cur_vectors),
        ):
            scores, where = best_cosine(vectors[at], other_vectors[other_at])
            best_cos[rows[at]] = scores
            cos_idx[rows[at]] = sent_idx[other_rows[other_at]][where]
    return best_cos, cos_idx


def build_similarity(cfg: Config, force: bool = False) -> Path:
    """Add the best cosine to every sentence exact matching left over.

    Writes `similarity/pair=YYYY.parquet`: the candidates' rows and columns
    plus `best_cos` and `cos_idx` (see `cosine_columns`).
    """
    out_dir = cfg.interim_dir / "similarity"
    for path in sorted((cfg.interim_dir / "candidates").glob("pair=*.parquet")):
        out = out_dir / path.name
        if out.exists() and not force:
            continue
        year = int(path.stem.split("=")[1])
        df = pd.read_parquet(path)
        best_cos, cos_idx = cosine_columns(
            df, _leftover(cfg, df, "cur", year), _leftover(cfg, df, "prev", year - 1)
        )
        write_parquet(df.assign(best_cos=best_cos, cos_idx=cos_idx), out)
    return out_dir
