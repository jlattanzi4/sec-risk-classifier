import numpy as np
import pandas as pd
import pytest

from pipeline.judging import agreement, stratified_sample, write_judge_inputs


def population(seed=0, n=5000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cik": rng.integers(1, 50, n),
            "filing_year": 2019,
            "sent_idx": np.arange(n),
            "pair_cos": rng.uniform(0.5, 1.0, n),
            "pair_j": rng.choice([0.0, 0.2, 0.45, 0.8], n, p=[0.6, 0.2, 0.1, 0.1]),
        }
    )


COS_EDGES = [0.0, 0.6, 0.7, 0.8, 0.9, 1.01]
J_EDGES = [0.0, 0.1, 0.3, 0.6, 1.01]


def test_every_non_empty_cell_gets_the_same_number():
    sample = stratified_sample(population(), n=400, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    per_cell = sample.groupby("stratum").size()
    assert len(per_cell) == 20 and per_cell.nunique() == 1 and len(sample) == 400


def test_a_small_cell_gives_all_it_has_and_the_rest_make_up_the_number():
    pop = population()
    rare = (pop["pair_j"] == 0.8) & (pop["pair_cos"] >= 0.9)
    pop = pd.concat([pop[~rare], pop[rare].head(3)])
    sample = stratified_sample(pop, n=400, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    per_cell = sample.groupby("stratum").size()
    assert per_cell.min() == 3 and len(sample) == 400
    assert per_cell.drop(per_cell.idxmin()).max() - per_cell.drop(per_cell.idxmin()).min() <= 1


def test_weights_add_back_up_to_the_population():
    pop = population()
    sample = stratified_sample(pop, n=400, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    assert sample["weight"].sum() == pytest.approx(len(pop))
    assert (sample.groupby("stratum")["weight"].nunique() == 1).all()  # one weight per cell
    assert (sample["weight"] >= 1).all()


def test_the_same_seed_draws_the_same_sample():
    a = stratified_sample(population(), n=200, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    b = stratified_sample(population(), n=200, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    c = stratified_sample(population(), n=200, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a["sent_idx"].equals(c["sent_idx"])


def test_judges_see_only_the_two_sentences(tmp_path):
    pairs = stratified_sample(population(), n=200, cos_edges=COS_EDGES, j_edges=J_EDGES, seed=0)
    pairs["a"], pairs["b"] = (
        "last year's sentence " + pairs["sent_idx"].astype(str),
        "this year's sentence",
    )
    pairs.insert(0, "id", [f"rp{i:04d}" for i in range(len(pairs))])
    write_judge_inputs(pairs, tmp_path, n_second=40, seed=0)
    first, second = (
        pd.read_csv(tmp_path / "judge1_input.csv"),
        pd.read_csv(tmp_path / "judge2_input.csv"),
    )
    assert list(first.columns) == list(second.columns) == ["id", "a", "b"]
    assert (
        sorted(first["id"]) == sorted(pairs["id"]) and first["id"].tolist() != pairs["id"].tolist()
    )
    assert len(second) == 40 and set(second["id"]) <= set(pairs["id"])
    strata = pairs.set_index("id").loc[second["id"], "stratum"].value_counts()
    assert strata.max() - strata.min() <= 1  # the subset is spread evenly too
    saved = pd.read_csv(tmp_path / "pairs.csv")
    assert {"stratum", "pair_j", "pair_cos", "weight"} <= set(saved.columns)


def test_kappa_matches_a_hand_computed_table():
    # 100 pairs: 40 agreed same, 20 agreed paraphrase, 25 agreed different, 15 off the diagonal.
    a = ["same"] * 45 + ["paraphrase"] * 25 + ["different"] * 30
    b = (
        (["same"] * 40 + ["paraphrase"] * 5)
        + (["paraphrase"] * 20 + ["different"] * 5)
        + (["different"] * 25 + ["same"] * 5)
    )
    result = agreement(pd.Series(a), pd.Series(b))
    observed = 0.85
    expected = 0.45 * 0.45 + 0.25 * 0.25 + 0.30 * 0.30
    assert result["agreement"] == pytest.approx(observed)
    assert result["kappa"] == pytest.approx((observed - expected) / (1 - expected))
    # Carried over (same or paraphrase) against different: 5 + 5 disagreements.
    binary_expected = 0.70 * 0.70 + 0.30 * 0.30
    assert result["kappa_carried"] == pytest.approx(
        (0.90 - binary_expected) / (1 - binary_expected)
    )
    assert result["table"]["same"]["paraphrase"] == 5 and result["n"] == 100
