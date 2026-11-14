import numpy as np
import pandas as pd
import pytest

from pipeline.io import write_parquet
from pipeline.labels import build_filing_metrics, build_labels, label
from tests.conftest import write_tables

TAU_V, TAU_P = 0.3, 0.8


def scores(rows):
    columns = ["side", "exact", "exact_idx", "best_j", "j_idx", "best_cos", "cos_idx"]
    return pd.DataFrame(rows, columns=columns)


def test_every_branch_of_the_labeling_rule():
    df = scores(
        [
            ("cur", True, 4, np.nan, -1, np.nan, -1),  # in last year's filing word for word
            ("cur", False, -1, 0.5, 7, 0.95, 7),  # lightly edited
            ("cur", False, -1, 0.3, 2, 0.10, 9),  # exactly at tau_v: a match
            ("cur", False, -1, 0.1, 3, 0.85, 8),  # reworded
            ("cur", False, -1, 0.0, -1, 0.8, 6),  # exactly at tau_p: a match
            ("cur", False, -1, 0.1, 3, 0.5, 8),  # new
            ("cur", False, -1, np.nan, -1, np.nan, -1),  # nothing to compare with: new
            ("prev", True, 0, np.nan, -1, np.nan, -1),
            ("prev", False, -1, 0.0, -1, 0.2, 5),  # gone this year
            ("prev", False, -1, 0.9, 1, 0.99, 1),
        ]
    )
    out = label(df, TAU_V, TAU_P)
    assert out["label"].tolist() == [
        "exact",
        "near",
        "near",
        "paraphrase",
        "paraphrase",
        "new",
        "new",
        "exact",
        "removed",
        "near",
    ]
    assert out["link_idx"].tolist() == [4, 7, 2, 8, 6, -1, -1, 0, -1, 1]


SAME = "We depend on a small number of customers for most of our revenue."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
EDITED = "Our business could be harmed if we lose many of our key personnel or senior managers."
GONE = "We have a history of operating losses and may never be profitable again."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for what we sell."


@pytest.fixture
def built(cfg):
    """One pair: this year keeps SAME, edits OLD, drops GONE and adds FRESH twice."""
    write_tables(cfg, {(1, 2017): [SAME, OLD, GONE], (1, 2018): [SAME, EDITED, FRESH, FRESH]})
    similarity = pd.DataFrame(
        {
            "cik": 1,
            "side": ["cur"] * 4 + ["prev"] * 3,
            "sent_idx": [0, 1, 2, 3, 0, 1, 2],
            "exact": [True, False, False, False, True, False, False],
            "exact_idx": [0, -1, -1, -1, 0, -1, -1],
            "best_j": [np.nan, 7 / 17, 0.0, 0.0, np.nan, 7 / 17, 0.0],
            "j_idx": [-1, 1, -1, -1, -1, 1, -1],
            "best_cos": [np.nan, 0.97, 0.4, 0.4, np.nan, 0.97, 0.3],
            "cos_idx": [-1, 1, 2, 2, -1, 1, 2],
        }
    )
    write_parquet(similarity, cfg.interim_dir / "similarity" / "pair=2018.parquet")
    peers = pd.read_parquet(cfg.interim_dir / "sentences" / "year=2018.parquet")[
        ["norm_hash"]
    ].drop_duplicates()
    peers["n_companies"], peers["n_groups"] = 1, 1
    peers.loc[peers.index[0], ["n_companies", "n_groups"]] = [9, 6]  # SAME is in 5 other groups
    write_parquet(peers, cfg.interim_dir / "peers" / "year=2018.parquet")
    build_labels(cfg, TAU_V, TAU_P)
    return cfg


def test_labels_are_written_for_both_sides(built):
    labels = pd.read_parquet(built.interim_dir / "labels" / "pair=2018.parquet")
    assert list(labels.columns) == ["cik", "side", "sent_idx", "label", "link_idx"]
    assert labels[labels["side"] == "cur"]["label"].tolist() == ["exact", "near", "new", "new"]
    assert labels[labels["side"] == "prev"]["label"].tolist() == ["exact", "near", "removed"]


def test_a_filing_s_shares_add_up(built):
    metrics = pd.read_parquet(build_filing_metrics(built))
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert (row["cik"], row["filing_year"], row["n_sentences"], row["n_prev"]) == (1, 2018, 4, 3)
    assert (row["share_exact"], row["share_near"], row["share_paraphrase"], row["share_new"]) == (
        0.25,
        0.25,
        0.0,
        0.5,
    )
    assert row["share_removed"] == pytest.approx(1 / 3)
    assert row["share_peer"] == 0.25


def test_shares_by_characters_weight_long_sentences_more(built):
    row = pd.read_parquet(build_filing_metrics(built)).iloc[0]
    total = len(SAME) + len(EDITED) + 2 * len(FRESH)
    assert row["n_chars"] == total
    assert row["share_new_chars"] == pytest.approx(2 * len(FRESH) / total)
    assert row["share_removed_chars"] == pytest.approx(
        len(GONE) / (len(SAME) + len(OLD) + len(GONE))
    )
    by_chars = row[
        ["share_exact_chars", "share_near_chars", "share_paraphrase_chars", "share_new_chars"]
    ]
    assert by_chars.sum() == pytest.approx(1.0)
