"""What makes an added sentence last: a hazard model of removal.

Two models. A Cox model is the check the design asks for: does a covariate
multiply the hazard by the same factor at every age? The model that is
reported is a discrete-time hazard model, because filings are annual and
durations are whole numbers: a complementary log-log link (the grouped-time
form of proportional hazards) with a baseline per age and a random intercept
per company, so that companies which simply rewrite more don't pass for an
effect of their sentences. It is fit by NUTS on a sample of companies, with
the data collapsed to counts.
"""

import numpy as np
import pandas as pd

MAX_AGE = 8  # ages 8 and over share one baseline: few chains get there
BIRTH_BANDS = [2006, 2010, 2014, 2017, 2020]
COX_REFERENCE_DIVISION = "D"  # manufacturing, the largest; the Cox check needs a baseline
R_HAT_LIMIT, MIN_EFFECTIVE = 1.01, 400
AGE_BANDS = {"age 1": (1, 1), "ages 2-3": (2, 3), "ages 4+": (4, MAX_AGE)}


class ConvergenceError(RuntimeError):
    """The sampler's draws can't be trusted, so no intervals are returned."""


def company_sample(chains: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Every chain of `n` companies, drawn in proportion from each SIC division."""
    companies = chains[["cik", "sic_division"]].drop_duplicates("cik")
    companies = companies.assign(sic_division=companies["sic_division"].fillna("none"))
    share = min(n / len(companies), 1.0)
    quota = (companies.groupby("sic_division").size() * share).round().astype(int)
    quota.iloc[quota.argmax()] += n - quota.sum() if share < 1 else 0  # rounding leftovers
    chosen = pd.concat(
        part.sample(int(quota[division]), random_state=seed)
        for division, part in companies.groupby("sic_division")
    )
    return chains[chains["cik"].isin(chosen["cik"])]


def _covariates(chains: pd.DataFrame) -> pd.DataFrame:
    """The model's covariates as columns of 0/1 and centered numbers."""
    out = pd.DataFrame(index=chains.index)
    out["peer_shared"] = chains["peer_shared"].astype(float)
    out["log_words"] = np.log(chains["n_words"]) - np.log(25)
    out["position"] = chains["position"] - 0.5
    out["log_section"] = np.log(chains["section_sentences"]) - np.log(250)
    out["birth_year"] = (chains["birth_year"] - 2013) / 5
    division = chains["sic_division"].fillna("none")
    for name in sorted(set(division) - {COX_REFERENCE_DIVISION}):
        out[f"division_{name}"] = (division == name).astype(float)
    return out


def cox_check(sample: pd.DataFrame, max_chains: int = 200_000, seed: int = 0) -> dict:
    """A Cox model of removal, and the test of proportional hazards for each covariate.

    Standard errors are clustered by company. A censored chain is entered
    just before its last age, since it was not watched at that age.
    `proportional_hazards_rejected` applies the rule fixed in the plan: the
    Schoenfeld test rejects at 0.01 for `peer_shared`, or for any covariate
    after a Bonferroni correction.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    if len(sample) > max_chains:
        sample = sample.sample(max_chains, random_state=seed)
    frame = _covariates(sample)
    frame = frame.loc[:, frame.std() > 0]
    frame["time"] = sample["duration"] - 0.5 * (~sample["event"].to_numpy())
    frame["event"] = sample["event"].astype(int)
    frame["cik"] = sample["cik"]
    frame = frame[frame["time"] > 0.5]  # censored at one filing: never watched
    model = CoxPHFitter().fit(frame, "time", "event", cluster_col="cik")
    # Time is transformed by Kaplan-Meier, not by rank: ranks break ties by row order, the rows
    # are in company order and nearly every time is tied, so a rank transform makes any
    # company-level covariate look as if its effect changed with age.
    test = proportional_hazard_test(model, frame, time_transform="km").summary
    summary = model.summary
    p_values = test["p"].to_dict()
    rejected = (
        p_values.get("peer_shared", 1.0) < 0.01 or min(p_values.values()) * len(p_values) < 0.01
    )
    return {
        "chains": len(frame),
        "companies": int(frame["cik"].nunique()),
        "hazard_ratios": {
            name: {
                "value": float(row["exp(coef)"]),
                "low": float(row["exp(coef) lower 95%"]),
                "high": float(row["exp(coef) upper 95%"]),
            }
            for name, row in summary.iterrows()
        },
        "tests": {
            name: {"statistic": float(test.loc[name, "test_statistic"]), "p": float(p)}
            for name, p in p_values.items()
        },
        "proportional_hazards_rejected": bool(rejected),
    }


def _thirds(values: pd.Series, name: str) -> pd.Series:
    cuts = values.quantile([1 / 3, 2 / 3]).to_numpy()
    return pd.Series(np.digitize(values, cuts), index=values.index).map(
        {0: f"{name}_low", 1: f"{name}_mid", 2: f"{name}_high"}
    )


def person_periods(sample: pd.DataFrame) -> pd.DataFrame:
    """Counts of chains watched (`at_risk`) and `removed`, per company, age and covariate cell.

    A chain removed at age d is watched at ages 1 to d. A chain censored at
    d is watched at ages 1 to d - 1. Ages over MAX_AGE are pooled into it.
    Cells: `peer_shared`; sentence length, position and section length in
    thirds; the year added in four bands; SIC division.
    """
    cells = pd.DataFrame(
        {
            "cik": sample["cik"],
            "peer_shared": sample["peer_shared"].astype(bool),
            "words": _thirds(sample["n_words"], "words"),
            "position": pd.cut(
                sample["position"], [-0.01, 1 / 3, 2 / 3, 1.0], labels=["top", "middle", "bottom"]
            ).astype(str),
            "section": _thirds(sample["section_sentences"], "section"),
            "born": pd.cut(
                sample["birth_year"],
                BIRTH_BANDS,
                labels=["2007-10", "2011-14", "2015-17", "2018-20"],
            ).astype(str),
            "division": sample["sic_division"].fillna("none"),
        }
    )
    duration, event = sample["duration"].to_numpy(), sample["event"].to_numpy(dtype=bool)
    keys = list(cells.columns)
    parts = []
    for age in range(1, int(duration.max()) + 1):
        removed = event & (duration == age)
        watched = (duration > age) | removed
        if not watched.any():
            continue
        counts = cells[watched].assign(removed=removed[watched].astype(int), at_risk=1)
        counts = counts.groupby(keys, as_index=False)[["at_risk", "removed"]].sum()
        parts.append(counts.assign(age=min(age, MAX_AGE)))
    periods = pd.concat(parts)
    return periods.groupby([*keys, "age"], as_index=False)[["at_risk", "removed"]].sum()


SENTENCE_FACTORS = {
    "words": "words_low",
    "position": "top",
    "section": "section_low",
    "born": "2007-10",
}


def _design(periods: pd.DataFrame, vary_by_age: bool) -> tuple[np.ndarray, list[str]]:
    """The sentence-level part of the model matrix: one column per effect, reference levels
    left out. (SIC division belongs to the company, and enters through the company effect.)"""
    columns: dict[str, np.ndarray] = {}
    peer = periods["peer_shared"].to_numpy(dtype=float)
    if vary_by_age:
        for name, (low, high) in AGE_BANDS.items():
            columns[f"peer_shared, {name}"] = peer * periods["age"].between(low, high).to_numpy()
    else:
        columns["peer_shared"] = peer
    for factor, reference in SENTENCE_FACTORS.items():
        for level in sorted(set(periods[factor]) - {reference}):
            columns[f"{factor}: {level}"] = (periods[factor] == level).to_numpy(dtype=float)
    return np.column_stack(list(columns.values())), list(columns)


def fit_hazard(
    periods: pd.DataFrame,
    vary_by_age: bool,
    warmup: int = 1000,
    draws: int = 1500,
    chains: int = 4,
    seed: int = 0,
    check: bool = True,
) -> dict:
    """The discrete-time hazard model with a company random effect.

        cloglog(hazard) = age effect + sentence effects + company effect
        company effect ~ Normal(level + division effect, company_sd)
        division effects ~ Normal(0, division_sd), summing to zero

    The level and the division effects sit in the mean of the company effect
    because they are facts about companies; written that way the sampler
    mixes well. Division effects are shrunk toward each other, since some
    divisions have only a handful of companies. Returns hazard ratios
    (exponentials of the effects; for a division, relative to the average
    division) with 95% credible intervals. With `check`, raises
    ConvergenceError unless every reported parameter has R-hat at most 1.01
    and at least 400 effective draws, with no divergent transitions.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    numpyro.enable_x64()
    matrix, names = _design(periods, vary_by_age)
    company, companies = pd.factorize(periods["cik"])
    of_company = periods.drop_duplicates("cik").set_index("cik")["division"].reindex(companies)
    division, divisions = pd.factorize(of_company, sort=True)
    age = periods["age"].to_numpy() - 1
    at_risk, removed = periods["at_risk"].to_numpy(), periods["removed"].to_numpy()

    def model():
        level = numpyro.sample("level", dist.Normal(-1.0, 2.0))
        later = numpyro.sample("age", dist.Normal(0.0, 2.0).expand([MAX_AGE - 1]))
        by_age = jnp.concatenate([jnp.zeros(1), later])  # relative to age 1
        effects = numpyro.sample("effects", dist.Normal(0.0, 1.0).expand([matrix.shape[1]]))
        division_sd = numpyro.sample("division_sd", dist.HalfNormal(0.5))
        # Scaled from unit draws: with few divisions and a spread that may be near zero,
        # sampling the effects directly sends the sampler down a funnel.
        unit = numpyro.sample(
            "division_unit", dist.ZeroSumNormal(1.0, event_shape=(len(divisions),))
        )
        by_division = numpyro.deterministic("division", division_sd * unit)
        company_sd = numpyro.sample("company_sd", dist.HalfNormal(1.0))
        effect = numpyro.sample("company", dist.Normal(level + by_division[division], company_sd))
        linear = by_age[age] + jnp.asarray(matrix) @ effects + effect[company]
        hazard = jnp.clip(1 - jnp.exp(-jnp.exp(linear)), 1e-9, 1 - 1e-9)
        numpyro.sample("removed", dist.Binomial(at_risk, probs=hazard), obs=removed)

    # The effects shared by all companies are correlated with each other (a sentence added in
    # 2019 can only be seen at ages 1 and 2), so they get a full mass matrix.
    mcmc = MCMC(
        NUTS(model, dense_mass=[("level", "age", "effects")]),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    grouped = mcmc.get_samples(group_by_chain=True)
    reported = ("level", "age", "effects", "division", "division_sd", "company_sd")
    stats = summary({k: grouped[k] for k in reported}, prob=0.95)
    max_r_hat = float(max(np.max(s["r_hat"]) for s in stats.values()))
    min_effective = float(min(np.min(s["n_eff"]) for s in stats.values()))
    slowest = min(stats, key=lambda k: np.min(stats[k]["n_eff"]))
    divergences = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    if check and (max_r_hat > R_HAT_LIMIT or min_effective < MIN_EFFECTIVE or divergences):
        raise ConvergenceError(
            f"R-hat {max_r_hat:.3f}, effective draws {min_effective:.0f} (slowest: {slowest}), "
            f"divergences {divergences}"
        )

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}

    def interval(values: np.ndarray) -> dict:
        low, mid, high = np.quantile(values, [0.025, 0.5, 0.975])
        return {"value": float(mid), "low": float(low), "high": float(high)}

    ratios = {name: interval(np.exp(samples["effects"][:, i])) for i, name in enumerate(names)}
    ratios |= {
        f"division: {name}": interval(np.exp(samples["division"][:, i]))
        for i, name in enumerate(divisions)
    }
    by_age = np.column_stack([np.zeros(len(samples["level"])), samples["age"]])
    baseline = 1 - np.exp(-np.exp(samples["level"][:, None] + by_age))
    return {
        "rows": len(periods),
        "companies": len(companies),
        "chains_watched_at_age_1": int(periods.loc[periods["age"] == 1, "at_risk"].sum()),
        "hazard_ratios": ratios,
        "baseline_hazard": {str(a + 1): interval(baseline[:, a]) for a in range(MAX_AGE)},
        "company_sd": interval(samples["company_sd"]),
        "diagnostics": {
            "max_r_hat": max_r_hat,
            "min_effective_draws": min_effective,
            "divergences": divergences,
            "draws": draws * chains,
            "slowest_parameter": slowest,
        },
    }
