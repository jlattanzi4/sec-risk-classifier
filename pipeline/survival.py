"""Survival curves for added sentences, straight from counts.

Time is counted in filings. A sentence that appears in t filings and is gone
from the next was removed at age t. One that appears in t filings and has no
next filing to be checked against (the company's last, a gap, the end of the
corpus) is censored: it counts as surviving ages 1 to t - 1 and says nothing
about age t.

`survival` at age t is the Kaplan-Meier estimate of the share of added
sentences that appear in more than t filings.
"""

import numpy as np
import pandas as pd

Z = 1.96


def _table(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    oldest = int(durations.max())
    ages = np.arange(1, oldest + 1)
    removed = np.bincount(durations[events], minlength=oldest + 1)[1:]
    censored = np.bincount(durations[~events], minlength=oldest + 1)[1:]
    reached = len(durations) - np.concatenate([[0], np.cumsum(removed + censored)[:-1]])
    at_risk = reached - censored
    watched = at_risk > 0
    ages, removed, at_risk = ages[watched], removed[watched], at_risk[watched]
    hazard = removed / at_risk
    survival = np.cumprod(1 - hazard)
    # Greenwood's variance, with bounds on the log(-log) scale so they stay inside 0 and 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.cumsum(removed / (at_risk * (at_risk - removed)))
        spread = Z * np.sqrt(variance) / np.abs(np.log(survival))
        low, high = survival ** np.exp(spread), survival ** np.exp(-spread)
    undefined = ~np.isfinite(spread)  # survival of exactly 0 or 1 has no interval this way
    low[undefined], high[undefined] = survival[undefined], survival[undefined]
    return pd.DataFrame(
        {
            "age": ages, "at_risk": at_risk, "removed": removed,
            "hazard": hazard, "survival": survival, "low": low, "high": high,
        }
    )  # fmt: skip


def life_table(chains: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """At each age: chains at risk, removed, the hazard, and survival with 95% bounds.

    `chains` has `duration` and `event`. With `by`, one table per value of
    that column, stacked, with the column kept. An age at which no chain is
    at risk has no row.
    """
    if by is None:
        return _table(chains["duration"].to_numpy(), chains["event"].to_numpy(dtype=bool))
    parts = [
        _table(part["duration"].to_numpy(), part["event"].to_numpy(dtype=bool)).assign(
            **{by: value}
        )
        for value, part in chains.groupby(by)
    ]
    table = pd.concat(parts, ignore_index=True)
    return table[[by, *[c for c in table.columns if c != by]]]


def median_survival(table: pd.DataFrame) -> int | None:
    """The first age at which half or fewer are left; None if the data end before that."""
    below = table[table["survival"] <= 0.5]
    return int(below["age"].iloc[0]) if len(below) else None
