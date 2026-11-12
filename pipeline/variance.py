"""How much of a filing's rewriting belongs to the company, its industry, and the year.

Each paired filing has a share of new sentences. A hierarchical model gives
every company, SIC division and fiscal year its own offset, and the spread of
each set of offsets says how much of the variation in rewriting sits at that
level; what is left is the filing-to-filing variation within a company.

    share = level + division effect + year effect + company effect + residual

Fit by NUTS, so each spread and each share of the variance has a credible
interval. The ten thousand company effects are integrated out exactly, which
leaves about thirty parameters to sample and makes the fit quick and stable. The division spread rests on ten or so divisions and its interval is
wide; that is the honest size of what ten groups can tell.
"""

import numpy as np
import pandas as pd

from pipeline.hazard import MIN_EFFECTIVE, R_HAT_LIMIT, ConvergenceError

COMPONENTS = ("company", "division", "year", "residual")
CLIP = 0.005


def fit_variance(
    metrics: pd.DataFrame,
    outcome: str = "share_new",
    scale: str = "share",
    warmup: int = 1000,
    draws: int = 1000,
    chains: int = 4,
    seed: int = 0,
    check: bool = True,
) -> dict:
    """Random intercepts for company, SIC division and year, with normal residuals.

    `metrics` has `cik`, `filing_year`, `sic_division` and the outcome.
    `scale="share"` models the share as it is, so the variance shares read as
    shares of the variation in rewriting; `scale="logit"` models its logit
    (after clipping to [0.005, 0.995]) as a check that the answer doesn't
    hang on the scale. Returns each component's standard deviation (`sd`) and
    share of the total variance (`share`), with 95% credible intervals, and
    `mean_share`, the posterior means, which sum to 1. With `check`, raises
    ConvergenceError on the standard used for the hazard model.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    numpyro.enable_x64()
    data = metrics.dropna(subset=[outcome]).assign(
        sic_division=lambda d: d["sic_division"].fillna("none")
    )
    y = data[outcome].to_numpy(dtype=float)
    if scale == "logit":
        y = np.clip(y, CLIP, 1 - CLIP)
        y = np.log(y / (1 - y))
    company, companies = pd.factorize(data["cik"])
    year, years = pd.factorize(data["filing_year"])
    division_of = data.drop_duplicates("cik").set_index("cik")["sic_division"].reindex(companies)
    division, divisions = pd.factorize(division_of)
    spread = float(y.std())

    n_companies = len(companies)
    size = jnp.asarray(np.bincount(company, minlength=n_companies), dtype=float)
    company_j, year_j, y_j = jnp.asarray(company), jnp.asarray(year), jnp.asarray(y)
    division_of_row = jnp.asarray(division[company])

    def model():
        level = numpyro.sample("level", dist.Normal(float(y.mean()), 5 * spread))
        sd = {
            name: numpyro.sample(f"sd_{name}", dist.HalfNormal(2 * spread)) for name in COMPONENTS
        }
        # Year and division offsets sum to zero: otherwise their average and the overall level
        # trade off freely and the sampler crawls along that ridge. Division offsets are scaled
        # from unit draws because their spread may be near zero (see hazard.py).
        unit = numpyro.sample(
            "division_unit", dist.ZeroSumNormal(1.0, event_shape=(len(divisions),))
        )
        of_division = numpyro.deterministic("division", sd["division"] * unit)
        of_year = numpyro.sample("year", dist.ZeroSumNormal(sd["year"], event_shape=(len(years),)))
        # The company effects are integrated out exactly. Given the other terms, a company's
        # n filings are jointly normal with variance residual^2 on the diagonal and company^2
        # everywhere, and that density needs only each company's sum and sum of squares.
        rest = y_j - level - of_division[division_of_row] - of_year[year_j]
        total = jax.ops.segment_sum(rest, company_j, n_companies)
        squares = jax.ops.segment_sum(rest**2, company_j, n_companies)
        within, between = sd["residual"] ** 2, sd["company"] ** 2
        mixed = within + size * between
        log_det = (size - 1) * jnp.log(within) + jnp.log(mixed)
        quadratic = (squares - between / mixed * total**2) / within
        numpyro.factor("filings", -0.5 * jnp.sum(log_det + quadratic + size * jnp.log(2 * jnp.pi)))

    mcmc = MCMC(
        NUTS(model, dense_mass=True),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    grouped = mcmc.get_samples(group_by_chain=True)
    stats = summary({k: grouped[k] for k in ["level", *[f"sd_{c}" for c in COMPONENTS]]}, prob=0.95)
    max_r_hat = float(max(np.max(s["r_hat"]) for s in stats.values()))
    min_effective = float(min(np.min(s["n_eff"]) for s in stats.values()))
    divergences = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    if check and (max_r_hat > R_HAT_LIMIT or min_effective < MIN_EFFECTIVE or divergences):
        raise ConvergenceError(
            f"R-hat {max_r_hat:.3f}, effective draws {min_effective:.0f}, divergences {divergences}"
        )

    samples = mcmc.get_samples()
    sds = np.column_stack([np.asarray(samples[f"sd_{c}"]) for c in COMPONENTS])
    shares = sds**2 / (sds**2).sum(axis=1, keepdims=True)

    def interval(values: np.ndarray) -> dict:
        low, mid, high = np.quantile(values, [0.025, 0.5, 0.975])
        return {"value": float(mid), "low": float(low), "high": float(high)}

    year_effects = np.asarray(samples["year"])
    return {
        "outcome": outcome,
        "scale": scale,
        "n": len(y),
        "levels": {"company": len(companies), "division": len(divisions), "year": len(years)},
        "level": interval(np.asarray(samples["level"])),
        "sd": {c: interval(sds[:, i]) for i, c in enumerate(COMPONENTS)},
        "share": {c: interval(shares[:, i]) for i, c in enumerate(COMPONENTS)},
        "mean_share": {c: {"value": float(shares[:, i].mean())} for i, c in enumerate(COMPONENTS)},
        "year_effects": {str(int(y_)): interval(year_effects[:, i]) for i, y_ in enumerate(years)},
        "division_effects": {
            str(d): interval(np.asarray(samples["division"])[:, i]) for i, d in enumerate(divisions)
        },
        "diagnostics": {
            "max_r_hat": max_r_hat,
            "min_effective_draws": min_effective,
            "divergences": divergences,
            "draws": draws * chains,
        },
    }
