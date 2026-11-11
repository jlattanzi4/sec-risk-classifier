import numpy as np
import pandas as pd
import pytest

from pipeline.baselines import pair_up, shuffle_words, unmatched_only_check
from pipeline.candidates import build_candidates
from pipeline.embed import Embedder, build_embeddings, build_similarity
from pipeline.io import read_sentences
from pipeline.labels import build_labels
from pipeline.matching import score_pairs
from tests.conftest import write_tables
from tests.test_cosine import Letters

SAME = "We depend on a small number of customers for most of our revenue."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
EDITED = "Our business could be harmed if we lose many of our key personnel or senior managers."
REWORDED = "Losing senior managers or other key personnel could harm our business."
GONE = "We have a history of operating losses and may never be profitable again."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for what we sell."
SAME_AGAIN = "Most of our revenue depends on a small number of customers, we find."


def embedder():
    return Embedder(Letters(), "letters", dim=26)


def test_random_pairs_never_join_a_company_or_a_year_with_itself():
    rng = np.random.default_rng(0)
    pool = pd.DataFrame(
        {
            "cik": rng.integers(1, 6, 400),
            "filing_year": rng.integers(2010, 2013, 400),
            "text": [f"t{i}" for i in range(400)],
        }
    )
    pairs = pair_up(pool, n=100, seed=0)
    assert len(pairs) == 100
    assert (pairs["cik_a"] != pairs["cik_b"]).all() and (
        pairs["filing_year_a"] != pairs["filing_year_b"]
    ).all()
    pd.testing.assert_frame_equal(pairs, pair_up(pool, n=100, seed=0))


def test_shuffling_keeps_every_word():
    text = "We may not be able to protect our intellectual property rights everywhere."
    mixed = shuffle_words(text, np.random.default_rng(0))
    assert sorted(mixed.split()) == sorted(text.split()) and mixed != text


COLUMNS = ["cik", "sent_idx", "text", "norm", "norm_hash"]


def test_scoring_pairs_directly_agrees_with_the_staged_pipeline(cfg):
    write_tables(
        cfg,
        {
            (1, 2017): [SAME, OLD, GONE],
            (1, 2018): [SAME, EDITED, FRESH],
            (2, 2017): [OLD, GONE],
            (2, 2018): [REWORDED, GONE, FRESH],
        },
    )
    build_candidates(cfg, workers=1)
    build_embeddings(cfg, embedder=embedder())
    staged = pd.read_parquet(build_similarity(cfg) / "pair=2018.parquet")
    direct = score_pairs(
        read_sentences(cfg, 2018, COLUMNS), read_sentences(cfg, 2017, COLUMNS), embedder()
    )
    key = ["cik", "side", "sent_idx"]
    staged, direct = (
        staged.sort_values(key).reset_index(drop=True),
        direct.sort_values(key).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(staged, direct[staged.columns], check_exact=False, atol=1e-3)


def test_a_new_sentence_that_rewords_a_carried_over_one_is_reported(cfg):
    # SAME is carried over word for word, so it leaves the comparison. SAME_AGAIN rewords it,
    # finds nothing among the leftovers, and is called new. FRESH really is new.
    write_tables(cfg, {(1, 2017): [SAME, GONE], (1, 2018): [SAME, SAME_AGAIN, FRESH]})
    build_candidates(cfg, workers=1)
    build_embeddings(cfg, embedder=embedder())
    build_similarity(cfg)
    build_labels(cfg, tau_v=0.3, tau_p=0.97)
    labels = pd.read_parquet(cfg.interim_dir / "labels" / "pair=2018.parquet")
    assert labels[labels["side"] == "cur"]["label"].tolist() == ["exact", "new", "new"]

    check = unmatched_only_check(cfg, tau_v=0.3, tau_p=0.97, embedder=embedder(), n_pairs=10)
    assert check["pairs"] == 1 and check["new_sentences"] == 2
    assert check["would_be_paraphrase"] == 1 and check["would_be_near"] == 0
    assert check["share_relabeled"] == pytest.approx(0.5)
