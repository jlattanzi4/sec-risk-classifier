import numpy as np
import pandas as pd
import pytest

from pipeline.variance import fit_variance

TRUE = {"company": 0.10, "year": 0.03, "division": 0.02, "residual": 0.08}


def simulate(seed=0, n_companies=600, n_divisions=8, n_years=10):
    rng = np.random.default_rng(seed)
    division_of = rng.integers(0, n_divisions, n_companies)
    company = rng.normal(0, TRUE["company"], n_companies)
    division = rng.normal(0, TRUE["division"], n_divisions)
    year = rng.normal(0, TRUE["year"], n_years)
    rows = []
    for c in range(n_companies):
        for y in rng.choice(n_years, size=rng.integers(3, 9), replace=False):
            value = (
                0.25
                + company[c]
                + division[division_of[c]]
                + year[y]
                + rng.normal(0, TRUE["residual"])
            )
            rows.append((c, 2010 + y, "ABCDEFGH"[division_of[c]], value))
    return pd.DataFrame(rows, columns=["cik", "filing_year", "sic_division", "share_new"])


@pytest.fixture(scope="module")
def fit():
    return fit_variance(simulate(), warmup=500, draws=500, chains=2, seed=0)


def test_each_spread_is_recovered(fit):
    for name, truth in TRUE.items():
        sd = fit["sd"][name]
        assert sd["low"] < truth < sd["high"], name
    assert (
        fit["sd"]["company"]["high"] - fit["sd"]["company"]["low"] < 0.03
    )  # and tightly, where data allow


def test_shares_of_variance_sum_to_one(fit):
    assert sum(part["value"] for part in fit["mean_share"].values()) == pytest.approx(1.0)
    total = sum(v**2 for v in TRUE.values())
    truth = TRUE["company"] ** 2 / total
    assert fit["share"]["company"]["low"] < truth < fit["share"]["company"]["high"]


def test_the_fit_reports_its_diagnostics(fit):
    assert fit["diagnostics"]["max_r_hat"] <= 1.01 and fit["diagnostics"]["divergences"] == 0
    assert fit["n"] == len(simulate()) and fit["levels"] == {
        "company": 600,
        "division": 8,
        "year": 10,
    }


def test_the_logit_scale_fits_the_logit_of_the_share():
    data = simulate(n_companies=150)
    data["share_new"] = data["share_new"].clip(0, 1)
    fit = fit_variance(data, scale="logit", warmup=300, draws=300, chains=2, seed=0, check=False)
    assert (
        fit["scale"] == "logit" and fit["sd"]["residual"]["value"] > 0.2
    )  # logits spread wider than shares
