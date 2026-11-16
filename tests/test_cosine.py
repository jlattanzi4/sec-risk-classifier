import numpy as np
import pandas as pd
import pytest

from pipeline.candidates import build_candidates
from pipeline.embed import Embedder, best_cosine, build_embeddings, build_similarity
from tests.conftest import write_tables

SAME = "We depend on a small number of customers for most of our revenue."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
REWORDED = "Losing senior managers or other key personnel could harm our business."
GONE = "We have a history of operating losses and may never be profitable again."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for what we sell."


class Letters:
    """Embeds a text as its letter counts, so texts with the same words land close together."""

    calls = 0

    def encode(self, texts, batch_size):
        Letters.calls += 1
        return np.array(
            [[t.lower().count(chr(97 + i)) for i in range(26)] for t in texts], dtype=float
        )


def embedder():
    return Embedder(Letters(), "letters", dim=26)


def unit(*rows):
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_best_cosine_finds_the_closest_row_and_where_it_is():
    left = unit([1, 0, 0], [0, 1, 1])
    right = unit([0, 1, 0], [1, 1, 0], [1, 0.1, 0])
    scores, where = best_cosine(left, right)
    assert where.tolist() == [2, 0]
    assert scores[0] == pytest.approx(1 / np.sqrt(1.01), rel=1e-5)
    assert scores[1] == pytest.approx(1 / np.sqrt(2), rel=1e-5)


def test_best_cosine_with_nothing_to_compare_to():
    scores, where = best_cosine(unit([1, 0]), np.zeros((0, 2), dtype=np.float32))
    assert np.isnan(scores).all() and where.tolist() == [-1]
    scores, where = best_cosine(np.zeros((0, 2), dtype=np.float32), unit([1, 0]))
    assert len(scores) == 0 and len(where) == 0


def test_identical_rows_score_one_after_being_stored_as_float16():
    rows = unit(*np.random.default_rng(0).normal(size=(50, 384)))
    stored = rows.astype(np.float16).astype(np.float32)
    scores, where = best_cosine(stored, stored)
    assert where.tolist() == list(range(50))
    assert np.abs(scores - 1).max() < 1e-3


@pytest.fixture
def built(cfg):
    write_tables(
        cfg,
        {
            (1, 2017): [SAME, OLD, GONE],
            (1, 2018): [SAME, REWORDED, FRESH],
            (2, 2017): [SAME],
            (2, 2018): [SAME, FRESH],  # nothing unmatched last year to compare FRESH with
        },
    )
    build_candidates(cfg, workers=1)
    Letters.calls = 0
    build_embeddings(cfg, embedder=embedder())
    return cfg


def test_only_unmatched_sentences_are_embedded_once_each(built):
    index = pd.read_parquet(built.interim_dir / "embeddings/year=2018.parquet")
    vectors = np.load(built.interim_dir / "embeddings/year=2018.npy")
    assert vectors.dtype == np.float16 and vectors.shape == (
        2,
        26,
    )  # REWORDED and FRESH; FRESH twice
    assert sorted(index["row"]) == [0, 1] and index["norm_hash"].is_unique
    assert np.load(built.interim_dir / "embeddings/year=2017.npy").shape == (2, 26)  # OLD and GONE


def test_a_finished_year_is_not_embedded_again(built):
    before = Letters.calls
    build_embeddings(built, embedder=embedder())
    assert Letters.calls == before
    (built.interim_dir / "embeddings/year=2018.npy").unlink()  # as if the run stopped mid-year
    build_embeddings(built, embedder=embedder())
    assert Letters.calls == before + 1
    assert not list((built.interim_dir / "embeddings").glob("*.part"))


def test_similarity_adds_the_best_cosine_for_unmatched_sentences(built):
    out = build_similarity(built)
    df = pd.read_parquet(out / "pair=2018.parquet")
    assert {"best_j", "j_idx", "best_cos", "cos_idx"} <= set(df.columns)
    one = df[df["cik"] == 1].set_index(["side", "sent_idx"])
    assert np.isnan(one.loc[("cur", 0), "best_cos"])  # exact match: not scored
    assert one.loc[("cur", 1), "cos_idx"] == 1  # REWORDED is closest to OLD
    assert one.loc[("cur", 1), "best_cos"] > one.loc[("cur", 2), "best_cos"]
    assert one.loc[("prev", 1), "cos_idx"] == 1  # and OLD to REWORDED
    two = df[df["cik"] == 2].set_index(["side", "sent_idx"])
    assert np.isnan(two.loc[("cur", 1), "best_cos"]) and two.loc[("cur", 1), "cos_idx"] == -1
