import pandas as pd
import pytest

from pipeline.survival import life_table, median_survival


def chains(*rows):
    return pd.DataFrame(rows, columns=["duration", "event", "group"])


TEN = chains(
    (1, True, "a"),
    (1, True, "a"),  # 2 of 10 removed after one filing
    (2, True, "a"),
    (2, True, "a"),  # 2 of the remaining 8 after two
    (3, False, "a"),
    (3, False, "a"),  # 2 leave the data after three filings
    (3, True, "a"),
    (3, True, "a"),  # 2 of the 4 still watched are removed
    (4, False, "a"),
    (4, False, "a"),
)


def test_hazard_and_survival_match_a_hand_count():
    table = life_table(TEN)
    assert table["age"].tolist() == [1, 2, 3]
    assert table["at_risk"].tolist() == [10, 8, 4]
    assert table["removed"].tolist() == [2, 2, 2]
    assert table["hazard"].tolist() == pytest.approx([0.2, 0.25, 0.5])
    assert table["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3])


def test_a_chain_censored_at_an_age_is_not_at_risk_at_that_age():
    table = life_table(chains((1, False, "a"), (1, True, "a"), (2, True, "a")))
    assert table["at_risk"].tolist() == [2, 1]  # the censored chain was never watched


def test_an_age_with_nobody_at_risk_has_no_row():
    table = life_table(chains((1, True, "a"), (3, False, "a")))
    assert table["age"].tolist() == [1, 2]  # at age 3 the last chain is censored


def test_groups_get_their_own_tables():
    both = pd.concat([TEN, chains((1, True, "b"), (2, False, "b"))])
    table = life_table(both, by="group")
    assert table[table["group"] == "a"]["survival"].tolist() == pytest.approx([0.8, 0.6, 0.3])
    assert table[table["group"] == "b"]["survival"].tolist() == pytest.approx([0.5])


def test_bounds_contain_the_estimate_and_survival_never_rises():
    table = life_table(TEN)
    assert (table["low"] <= table["survival"]).all() and (table["survival"] <= table["high"]).all()
    assert table["survival"].is_monotonic_decreasing
    assert (table["low"] >= 0).all() and (table["high"] <= 1).all()


def test_median_is_the_first_age_at_or_below_half():
    assert median_survival(life_table(TEN)) == 3
    assert (
        median_survival(
            life_table(chains((1, True, "a"), (2, False, "a"), (2, False, "a"), (2, False, "a")))
        )
        is None
    )
