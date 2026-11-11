import pandas as pd
import pytest

from pipeline.thresholds import best_threshold, evaluate_known, predict, tier_makeup, tune, wilson


def test_a_clean_split_puts_the_threshold_in_the_gap():
    scores = pd.Series([0.05, 0.10, 0.20, 0.60, 0.70, 0.90])
    positive = pd.Series([False, False, False, True, True, True])
    threshold, f1 = best_threshold(scores, positive)
    assert threshold == pytest.approx(0.40) and f1 == 1.0


def test_between_equally_good_thresholds_the_higher_wins():
    # Cutting at 0.2 finds both positives and lets two negatives through; cutting at 0.65
    # lets nothing through and misses one positive. Both score F1 = 2/3.
    scores = pd.Series([0.1, 0.3, 0.4, 0.5, 0.8])
    positive = pd.Series([False, True, False, False, True])
    threshold, f1 = best_threshold(scores, positive)
    assert f1 == pytest.approx(2 / 3) and threshold == pytest.approx(0.65)


def known(rows):
    return pd.DataFrame(rows, columns=["kind", "split", "j", "cos"])


TUNE_ROWS = [
    ("verbatim", "tune", 0.9, 0.99),
    ("verbatim", "tune", 0.5, 0.97),
    ("paraphrase", "tune", 0.1, 0.90),
    ("paraphrase", "tune", 0.0, 0.85),
    ("different_easy", "tune", 0.0, 0.40),
    ("different_easy", "tune", 0.0, 0.55),
    ("different_hard", "tune", 0.8, 0.98),
    ("different_hard", "tune", 0.0, 0.60),
]


def test_tuning_uses_only_the_tuning_half():
    scores = known(TUNE_ROWS)
    found = tune(scores)
    assert found["tau_v"] == pytest.approx(0.30)  # between 0.1 (paraphrase) and 0.5 (verbatim)
    assert found["tau_p"] == pytest.approx(0.725)  # between 0.60 (hard, under tau_v) and 0.85
    noisy = pd.concat(
        [scores, known([("paraphrase", "test", 0.95, 0.10), ("verbatim", "test", 0.0, 0.0)])]
    )
    assert tune(noisy) == found


def test_reversed_claims_are_left_out_of_the_first_threshold():
    # The hard edit at j = 0.8 would be a false match for any tau_v under 0.8; it is not counted.
    assert tune(known(TUNE_ROWS))["f1_v"] == 1.0


def test_predict_applies_the_stages_in_order():
    scores = known(
        [
            ("x", "test", 0.5, 0.2),
            ("x", "test", 0.1, 0.9),
            ("x", "test", 0.1, 0.5),
            ("x", "test", 0.3, 0.725),
        ]
    )
    assert predict(scores["j"], scores["cos"], tau_v=0.3, tau_p=0.725).tolist() == [
        "near",
        "paraphrase",
        "new",
        "near",
    ]
    assert predict(pd.Series([0.0]), pd.Series([0.725]), 0.3, 0.725).tolist() == [
        "paraphrase"
    ]  # at the threshold


def test_wilson_interval():
    low, high = wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4) and high == pytest.approx(0.9433, abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)


def test_precision_and_recall_on_the_held_out_half():
    rows = [
        ("verbatim", "test", 0.9, 0.99),
        ("verbatim", "test", 0.1, 0.95),
        ("verbatim", "test", 0.0, 0.50),
        ("paraphrase", "test", 0.0, 0.90),
        ("paraphrase", "test", 0.0, 0.60),
        ("different_easy", "test", 0.0, 0.40),
        ("different_hard", "test", 0.8, 0.98),
        ("verbatim", "tune", 0.0, 0.0),  # not counted
    ]
    result = evaluate_known(known(rows), tau_v=0.3, tau_p=0.725)
    assert result["n"] == 7
    assert result["labeled"]["verbatim"] == {"near": 1, "paraphrase": 1, "new": 1}
    assert result["labeled"]["different_hard"] == {"near": 1, "paraphrase": 0, "new": 0}
    carried = result["carried_over"]
    # Carried over means near or paraphrase. Predicted carried: 2 verbatim, 1 paraphrase, 1 hard = 4.
    assert carried["precision"]["value"] == pytest.approx(3 / 4)
    assert carried["precision_without_hard"]["value"] == pytest.approx(3 / 3)
    assert carried["recall"]["value"] == pytest.approx(3 / 5)
    assert carried["recall"]["low"] < 3 / 5 < carried["recall"]["high"]


def test_tier_makeup_weights_the_sample_back_to_the_population():
    pairs = pd.DataFrame(
        {
            "pair_j": [0.8, 0.8, 0.0, 0.0, 0.0],
            "pair_cos": [0.99, 0.99, 0.9, 0.9, 0.3],
            "label": ["same", "different", "paraphrase", "different", "different"],
            "weight": [30.0, 10.0, 5.0, 15.0, 100.0],
        }
    )
    makeup = tier_makeup(pairs, tau_v=0.3, tau_p=0.725)
    assert makeup["near"]["same"] == pytest.approx(0.75) and makeup["near"][
        "different"
    ] == pytest.approx(0.25)
    assert makeup["paraphrase"]["paraphrase"] == pytest.approx(0.25)
    assert makeup["new"]["different"] == pytest.approx(1.0)
    assert makeup["near"]["n"] == 2 and makeup["near"]["stands_for"] == 40
    # Of everything the judges called carried over (30 + 5), all of it was found.
    assert makeup["recall_carried"] == pytest.approx(1.0)
