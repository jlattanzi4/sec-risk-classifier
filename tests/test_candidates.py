import numpy as np
import pandas as pd
import pytest

from pipeline.candidates import match_pair
from pipeline.sentences import norm_hash, normalize

SAME = "We depend on a small number of customers for most of our revenue."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
EDITED = "Our business could be harmed if we lose many of our key personnel or senior managers."
TWICE = "Our stock price may be volatile."
GONE = "We have a history of operating losses and may never be profitable again."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for what we sell."


def frame(*texts):
    norms = [normalize(t) for t in texts]
    return pd.DataFrame(
        {"sent_idx": range(len(texts)), "norm": norms, "norm_hash": [norm_hash(n) for n in norms]}
    )


def row(result, side, sent_idx):
    return result[(result["side"] == side) & (result["sent_idx"] == sent_idx)].iloc[0]


@pytest.fixture
def result():
    #                 0     1       2      3      4
    cur = frame(SAME, EDITED, TWICE, TWICE, FRESH)
    prev = frame(TWICE, OLD, SAME, GONE)
    return match_pair(cur, prev)


def test_one_row_for_every_sentence_of_both_filings(result):
    assert result.groupby("side").size().to_dict() == {"cur": 5, "prev": 4}
    assert list(result.columns) == ["side", "sent_idx", "exact", "exact_idx", "best_j", "j_idx"]


def test_an_identical_sentence_is_exact_on_both_sides_and_points_at_its_counterpart(result):
    assert row(result, "cur", 0)["exact"] and row(result, "cur", 0)["exact_idx"] == 2
    assert row(result, "prev", 2)["exact"] and row(result, "prev", 2)["exact_idx"] == 0
    assert np.isnan(row(result, "cur", 0)["best_j"]) and row(result, "cur", 0)["j_idx"] == -1


def test_a_sentence_repeated_this_year_is_carried_over_each_time(result):
    assert row(result, "cur", 2)["exact"] and row(result, "cur", 3)["exact"]
    assert row(result, "cur", 2)["exact_idx"] == 0 and row(result, "cur", 3)["exact_idx"] == 0
    assert row(result, "prev", 0)["exact_idx"] == 2  # the first of the two copies


def test_an_edited_sentence_scores_its_jaccard_against_its_source(result):
    # 16 words give 12 shingles each; the changed word (the 9th) sits in 5 of them,
    # so 7 are shared out of 12 + 12 - 7.
    edited = row(result, "cur", 1)
    assert not edited["exact"] and edited["exact_idx"] == -1
    assert edited["best_j"] == pytest.approx(7 / 17)
    assert edited["j_idx"] == 1
    source = row(result, "prev", 1)
    assert source["best_j"] == pytest.approx(7 / 17) and source["j_idx"] == 1


def test_a_sentence_with_no_counterpart_scores_zero(result):
    assert row(result, "cur", 4)["best_j"] == 0.0 and row(result, "cur", 4)["j_idx"] == -1
    assert row(result, "prev", 3)["best_j"] == 0.0 and row(result, "prev", 3)["j_idx"] == -1


def test_scores_are_null_when_the_other_side_has_nothing_unmatched():
    result = match_pair(frame(SAME, FRESH), frame(SAME))
    assert np.isnan(row(result, "cur", 1)["best_j"]) and row(result, "cur", 1)["j_idx"] == -1
    assert not row(result, "cur", 1)["exact"]


def test_an_empty_previous_filing_leaves_everything_unmatched():
    result = match_pair(frame(SAME, FRESH), frame())
    assert len(result) == 2 and not result["exact"].any()
    assert result["best_j"].isna().all() and (result["j_idx"] == -1).all()


def test_indexes_are_sentence_numbers_not_positions():
    cur = frame(EDITED).assign(sent_idx=[7])
    prev = frame(GONE, OLD).assign(sent_idx=[3, 9])
    result = match_pair(cur, prev)
    assert row(result, "cur", 7)["j_idx"] == 9 and row(result, "prev", 9)["j_idx"] == 7


# --- the stage ---------------------------------------------------------------------------------

from pipeline.candidates import (
    _pooled_jaccard,
    build_candidates,
    jaccard_matches,
)
from pipeline.stages import NoPool
from tests.conftest import write_tables


def test_spreading_companies_over_processes_changes_nothing():
    cur = pd.concat([frame(EDITED, FRESH).assign(cik=c) for c in range(1, 8)], ignore_index=True)
    prev = pd.concat([frame(OLD, GONE).assign(cik=c) for c in range(1, 8)], ignore_index=True)
    cur.index, prev.index = cur.index * 3 + 5, prev.index * 2 + 1  # not 0..n-1, as in the stage
    one = jaccard_matches(cur, prev)
    many = _pooled_jaccard(cur, prev, pool=NoPool(), n_batches=3)
    for a, b in zip(one, many, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_the_stage_writes_one_file_per_year_that_has_pairs(cfg):
    write_tables(
        cfg,
        {
            (1, 2017): [SAME, OLD],
            (1, 2018): [SAME, EDITED, FRESH],
            (1, 2020): [SAME],  # 2019 is missing: no pair for 2020
            (2, 2018): [GONE],  # no 2017 for company 2: not paired
        },
    )
    out = build_candidates(cfg, workers=1)
    assert [p.name for p in sorted(out.glob("*.parquet"))] == ["pair=2018.parquet"]
    df = pd.read_parquet(out / "pair=2018.parquet")
    assert set(df["cik"]) == {1}
    assert df.groupby("side").size().to_dict() == {"cur": 3, "prev": 2}
    assert df[(df["side"] == "cur") & (df["sent_idx"] == 1)]["best_j"].iloc[0] == pytest.approx(
        7 / 17
    )


def test_a_finished_year_is_not_rebuilt(cfg):
    write_tables(cfg, {(1, 2017): [SAME], (1, 2018): [SAME]})
    path = build_candidates(cfg, workers=1) / "pair=2018.parquet"
    stamp = path.stat().st_mtime_ns
    build_candidates(cfg, workers=1)
    assert path.stat().st_mtime_ns == stamp
