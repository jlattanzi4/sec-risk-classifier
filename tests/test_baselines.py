import numpy as np
import pandas as pd
import pytest

from pipeline.baselines import (
    pair_up,
    related_filers,
    shuffle_words,
    stranger_filings,
    unmatched_only_check,
)
from pipeline.candidates import build_candidates
from pipeline.embed import Embedder, build_embeddings, build_similarity
from pipeline.io import read_sentences, write_parquet
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


def test_a_filer_grouped_with_the_company_in_either_year_is_related():
    # Groups are named for their lowest CIK, a year at a time. Company 5 files with 3 this
    # year (group 3) and filed with 7 last year (group 5): the name changed, 7 is still kin.
    filings = pd.DataFrame(
        {
            "cik": [3, 5, 9, 5, 7, 9],
            "filing_year": [2018, 2018, 2018, 2017, 2017, 2017],
            "group": [3, 3, 9, 5, 5, 9],
        }
    )
    assert related_filers(filings, cik=5, year=2018) == {3, 5, 7}
    assert related_filers(filings, cik=9, year=2018) == {9}


def test_a_stranger_is_never_the_company_s_own_kin(cfg):
    # Companies 1 and 2 file the same section (one group); company 3 shares nothing with them.
    # A stranger for 1 or 2 can only be 3, so nothing matches word for word or nearly.
    twin = {2017: [SAME, OLD, GONE], 2018: [SAME, EDITED, FRESH]}
    other = ["Zinc quotas vex us.", "Jumpy quartz buyers fix zinc prices weekly in bulk lots."]
    write_tables(
        cfg,
        {(cik, year): texts for cik in (1, 2) for year, texts in twin.items()}
        | {(3, year): other for year in (2017, 2018, 2019)},
    )
    listing = pd.read_parquet(cfg.interim_dir / "filings.parquet").assign(sic_division="D")
    write_parquet(listing, cfg.interim_dir / "filings.parquet")
    groups = listing[["cik", "filing_year"]].assign(group=listing["cik"].replace({2: 1}))
    write_parquet(groups, cfg.interim_dir / "filer_groups.parquet")

    result = stranger_filings(cfg, tau_v=0.3, tau_p=0.97, embedder=embedder(), n_pairs=4)
    assert result["n_filings"] == 4 and result["n_companies"] == 3
    assert result["own"]["n"] == result["stranger"]["n"] == 10
    assert result["own"]["exact"] == pytest.approx(0.6)  # SAME twice, and 3's four sentences
    for kind in ("stranger", "stranger_same_division"):
        assert result[kind]["exact"] == 0 and result[kind]["near"] == 0
