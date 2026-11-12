import numpy as np
import pandas as pd
import pytest

from pipeline.hazard import (
    ConvergenceError,
    company_sample,
    cox_check,
    fit_hazard,
    person_periods,
)


def simulate_chains(
    n_companies=300, per_company=150, hazard_ratio=0.5, sigma=0.0, only_first_age=False, seed=0
):
    """Chains with a known effect of `peer_shared` on the hazard of removal, watched for up to 8 filings."""
    rng = np.random.default_rng(seed)
    n = n_companies * per_company
    cik = np.repeat(np.arange(n_companies), per_company)
    frailty = rng.normal(0, sigma, n_companies)[cik]
    peer = rng.random(n) < 0.3
    duration, event = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    watched = rng.integers(2, 9, n)  # filings before the data end for this chain
    for age in range(1, 9):
        effect = np.log(hazard_ratio) * peer * (age == 1 if only_first_age else 1)
        hazard = 1 - np.exp(-np.exp(np.log(0.25) + effect + frailty))
        duration[alive] = age
        dies = alive & (age < watched) & (rng.random(n) < hazard)
        event[dies] = True
        alive &= ~dies & (age < watched)
    return pd.DataFrame(
        {
            "cik": cik,
            "duration": duration,
            "event": event,
            "peer_shared": peer,
            "n_words": rng.integers(8, 60, n),
            "position": rng.random(n),
            "section_sentences": rng.integers(50, 600, n),
            "birth_year": rng.integers(2007, 2020, n),
            "sic_division": rng.choice(["D", "H", "I"], n_companies)[cik],
        }
    )


def test_company_sample_keeps_whole_companies_from_every_division():
    chains = simulate_chains(n_companies=300, per_company=5)
    sample = company_sample(chains, n=90, seed=0)
    assert sample["cik"].nunique() == 90
    assert set(sample["sic_division"]) == {"D", "H", "I"}
    assert len(sample) == 90 * 5  # every chain of a sampled company
    assert company_sample(chains, n=90, seed=0)["cik"].tolist() == sample["cik"].tolist()


def test_person_periods_count_who_was_watched_at_each_age():
    chains = pd.DataFrame(
        {
            "cik": 1,
            "duration": [3, 3, 1],
            "event": [True, False, False],
            "peer_shared": False,
            "n_words": 20,
            "position": 0.5,
            "section_sentences": 200,
            "birth_year": 2012,
            "sic_division": "D",
        }
    )
    periods = person_periods(chains).groupby("age")[["at_risk", "removed"]].sum()
    # Removed at 3: watched at 1, 2, 3. Censored at 3: watched at 1 and 2. Censored at 1: never watched.
    assert periods["at_risk"].to_dict() == {1: 2, 2: 2, 3: 1}
    assert periods["removed"].to_dict() == {1: 0, 2: 0, 3: 1}


def test_ages_past_eight_are_pooled():
    chains = pd.DataFrame(
        {
            "cik": 1,
            "duration": [11],
            "event": [True],
            "peer_shared": False,
            "n_words": 20,
            "position": 0.5,
            "section_sentences": 200,
            "birth_year": 2008,
            "sic_division": "D",
        }
    )
    periods = person_periods(chains).groupby("age")[["at_risk", "removed"]].sum()
    assert (
        periods.index.max() == 8
        and periods.loc[8, "at_risk"] == 4
        and periods.loc[8, "removed"] == 1
    )


def test_cox_recovers_a_proportional_effect_and_the_check_passes():
    result = cox_check(simulate_chains(hazard_ratio=0.5, seed=1))
    ratio = result["hazard_ratios"]["peer_shared"]["value"]
    assert 0.45 < ratio < 0.56
    assert not result["proportional_hazards_rejected"]


def test_the_check_fails_when_the_effect_is_only_at_the_first_age():
    result = cox_check(simulate_chains(hazard_ratio=0.3, only_first_age=True, seed=2))
    assert result["proportional_hazards_rejected"]
    assert result["tests"]["peer_shared"]["p"] < 0.01


def test_the_hazard_model_recovers_the_effect_and_the_company_spread():
    chains = simulate_chains(n_companies=150, per_company=120, hazard_ratio=0.5, sigma=0.6, seed=3)
    fit = fit_hazard(
        person_periods(chains), vary_by_age=False, warmup=400, draws=400, chains=2, seed=0
    )
    low, high = (
        fit["hazard_ratios"]["peer_shared"]["low"],
        fit["hazard_ratios"]["peer_shared"]["high"],
    )
    assert low < 0.5 < high and high - low < 0.15
    assert fit["company_sd"]["low"] < 0.6 < fit["company_sd"]["high"]
    assert fit["diagnostics"]["max_r_hat"] <= 1.01 and fit["diagnostics"]["divergences"] == 0


def test_a_fit_that_has_not_converged_raises():
    chains = simulate_chains(n_companies=40, per_company=40, sigma=0.6, seed=4)
    with pytest.raises(ConvergenceError):
        fit_hazard(person_periods(chains), vary_by_age=False, warmup=3, draws=20, chains=2, seed=0)
