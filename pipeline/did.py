"""Did companies rewrite their risk sections when COVID-19 arrived, or add to them?

Annual reports for fiscal 2019 were filed across the weeks in which the
pandemic began. A company that filed on 2 March 2020 wrote its risk section
before; one that filed on 25 March wrote it after. That is not an experiment:
companies that file late differ from those that file early. But they differ
every year, and in earlier years nothing happened in mid-March, so the earlier
years measure the standing difference and the change in that difference in
2020 is the estimate (a difference-in-differences).

Exposure is set from the calendar: filed on or after 11 March of the year
after the fiscal year. In 2020 that is the day the WHO declared a pandemic.
"""

import re

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences

CUTOFF_MONTH, CUTOFF_DAY = 3, 11
EVENT_YEAR, COMPARISON = 2019, (2016, 2017, 2018)
OUTCOMES = ("share_new", "share_removed", "share_near", "growth", "share_new_ex_covid")
COVID = re.compile(r"covid|coronavirus|sars-cov|2019-ncov", re.IGNORECASE)
# Weeks from the cutoff: (first, last), None for open-ended.
WEEK_BINS = [(None, -4), (-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 6), (7, None)]
REFERENCE_WEEK = -2
MIN_EVENT_FILINGS = 20


def _cutoff(years: pd.Series) -> pd.Series:
    """11 March of the year after each fiscal year."""
    return pd.to_datetime({"year": years + 1, "month": CUTOFF_MONTH, "day": CUTOFF_DAY})


def covid_shares(cfg: Config, years: tuple[int, ...] = (2017, 2018, 2019, 2020)) -> pd.DataFrame:
    """Per paired filing: whether it mentions COVID, the share of its sentences that do,
    and the share of its sentences that are both new and mention it."""
    parts = []
    for year in years:
        path = cfg.interim_dir / "labels" / f"pair={year}.parquet"
        if not path.exists():
            continue
        labels = pd.read_parquet(path)
        labels = labels[labels["side"] == "cur"]
        text = read_sentences(cfg, year, ["cik", "sent_idx", "text"])
        text = text[text["cik"].isin(labels["cik"].unique())]
        mentions = text.assign(covid=text["text"].str.contains(COVID).to_numpy())
        merged = labels.merge(mentions[["cik", "sent_idx", "covid"]], on=["cik", "sent_idx"])
        merged["new_covid"] = merged["covid"] & (merged["label"] == "new")
        shares = merged.groupby("cik").agg(
            mentions_covid=("covid", "any"),
            share_covid=("covid", "mean"),
            share_new_covid=("new_covid", "mean"),
        )
        parts.append(shares.reset_index().assign(filing_year=year))
    if not parts:
        return pd.DataFrame(
            columns=["cik", "filing_year", "mentions_covid", "share_covid", "share_new_covid"]
        )
    return pd.concat(parts, ignore_index=True)


def analysis_table(cfg: Config) -> pd.DataFrame:
    """One row per paired filing, with its exposure and outcomes.

    `days_from_cutoff` counts from 11 March of the year after the fiscal
    year, `week` is that in whole weeks, and `late` is filed on or after the
    cutoff. `late_usual` applies the same test to the company's previous
    filing, so it was settled before the event year began. `december`,
    `same_fye` and `unique` are the three conditions of the main sample.
    """
    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet",
        columns=["cik", "filing_year", "report_date", "date_filed", "match_status", "sic_division"],
    )
    previous = filings[["cik", "filing_year", "report_date", "date_filed"]].assign(
        filing_year=filings["filing_year"] + 1
    )
    previous = previous.rename(columns={"report_date": "prev_report", "date_filed": "prev_filed"})
    table = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet")
    table = table.merge(filings, on=["cik", "filing_year"]).merge(
        previous, on=["cik", "filing_year"]
    )

    table["days_from_cutoff"] = (table["date_filed"] - _cutoff(table["filing_year"])).dt.days
    table["week"] = table["days_from_cutoff"] // 7
    table["late"] = table["days_from_cutoff"] >= 0
    table["late_usual"] = table["prev_filed"] >= _cutoff(table["filing_year"] - 1)
    table["december"] = table["report_date"].dt.month == 12
    table["same_fye"] = table["report_date"].dt.month == table["prev_report"].dt.month
    table["unique"] = table["match_status"] == "unique"
    table["growth"] = np.log(table["n_sentences"] / table["n_prev"].where(table["n_prev"] > 0))

    table = table.merge(covid_shares(cfg), on=["cik", "filing_year"], how="left")
    table["mentions_covid"] = table["mentions_covid"].fillna(False).astype(bool)
    table[["share_covid", "share_new_covid"]] = table[["share_covid", "share_new_covid"]].fillna(
        0.0
    )
    table["share_new_ex_covid"] = table["share_new"] - table["share_new_covid"]
    return table.drop(columns=["prev_report"])


def main_sample(
    table: pd.DataFrame,
    years: tuple[int, ...] = (*COMPARISON, EVENT_YEAR),
    require_unique: bool = True,
    require_same_fye: bool = True,
) -> pd.DataFrame:
    """December fiscal years, unchanged from the year before, with a single matching 10-K."""
    keep = table["filing_year"].isin(years) & table["december"]
    if require_unique:
        keep &= table["unique"]
    if require_same_fye:
        keep &= table["same_fye"]
    return table[keep]


def _estimate(row: pd.Series) -> dict:
    return {
        "estimate": float(row["Estimate"]),
        "se": float(row["Std. Error"]),
        "low": float(row["2.5%"]),
        "high": float(row["97.5%"]),
        "p": float(row["Pr(>|t|)"]),
    }


def did(
    table: pd.DataFrame,
    outcome: str,
    event_year: int = EVENT_YEAR,
    comparison: tuple[int, ...] = COMPARISON,
    exposure: str = "late",
    fixed_effects: bool = True,
) -> dict:
    """The difference-in-differences for one outcome.

        outcome ~ exposed + exposed x event year | company + year

    with standard errors clustered by company. The coefficient on the
    interaction is how much more the outcome changed for exposed filers in
    the event year than the standing early-late gap in the comparison years.
    Without `fixed_effects`, companies are pooled and only year effects kept.
    """
    import pyfixest as pf

    data = table[table["filing_year"].isin([*comparison, event_year])].dropna(subset=[outcome])
    data = data.assign(
        exposed=data[exposure].astype(int), event=(data["filing_year"] == event_year).astype(int)
    )
    data["treated"] = data["exposed"] * data["event"]
    absorbed = "cik + filing_year" if fixed_effects else "filing_year"
    fit = pf.feols(f"{outcome} ~ exposed + treated | {absorbed}", data=data, vcov={"CRV1": "cik"})
    means = data.groupby(["exposed", "event"])[outcome].mean()
    return {
        "outcome": outcome,
        "event_year": event_year,
        "comparison": list(comparison),
        "exposure": exposure,
        "fixed_effects": fixed_effects,
        **_estimate(fit.tidy().loc["treated"]),
        "filings": len(data),
        "companies": int(data["cik"].nunique()),
        "exposed_event_filings": int(data["treated"].sum()),
        "means": {
            name: {
                period: float(means.get((exposed, event), np.nan))
                for period, event in (("comparison", 0), ("event", 1))
            }
            for name, exposed in (("early", 0), ("late", 1))
        },
    }


def placebos(table: pd.DataFrame, outcome: str, exposure: str = "late") -> list[dict]:
    """The same estimate with the shock moved to a year in which nothing happened.

    Fiscal 2017 against 2016 and 2018, and fiscal 2018 against 2016 and 2017.
    Fiscal 2019 is in neither.
    """
    quiet = table[table["filing_year"].isin(COMPARISON)]
    return [
        did(
            quiet,
            outcome,
            event_year=year,
            comparison=tuple(y for y in COMPARISON if y != year),
            exposure=exposure,
        )
        for year in COMPARISON[1:]
    ]


def _bin_label(first: int | None, last: int | None) -> str:
    if first is None:
        return f"≤ {last}"
    if last is None:
        return f"{first} and later"
    return str(first) if first == last else f"{first} to {last}"


def _bins(weeks: pd.Series, in_event: pd.Series) -> list[tuple[int | None, int | None]]:
    """WEEK_BINS, with any bin holding too few event-year filings merged into its neighbor
    on the side away from the cutoff (or toward it, for the outermost bins)."""
    bins = list(WEEK_BINS)

    def count(b: tuple) -> int:
        low = -np.inf if b[0] is None else b[0]
        high = np.inf if b[1] is None else b[1]
        return int((in_event & weeks.between(low, high)).sum())

    while len(bins) > 2:
        thin = [i for i, b in enumerate(bins) if count(b) < MIN_EVENT_FILINGS]
        if not thin:
            break
        i = thin[0]
        before_cutoff = bins[i][1] is not None and bins[i][1] < 0
        j = i - 1 if (before_cutoff and i > 0) or i == len(bins) - 1 else i + 1
        low, high = min(i, j), max(i, j)
        bins[low : high + 1] = [(bins[low][0], bins[high][1])]
    return bins


def event_study(
    table: pd.DataFrame,
    outcome: str,
    event_year: int = EVENT_YEAR,
    comparison: tuple[int, ...] = COMPARISON,
    reference: int = REFERENCE_WEEK,
) -> list[dict]:
    """The event-year difference by filing week, relative to a week before the cutoff.

        outcome ~ week bin + week bin x event year | company + year

    One row per bin: the estimate and interval for the interaction (zero and
    no interval for the reference bin, the busiest week before the cutoff),
    and how many event-year and comparison filings it holds. Flat estimates
    before week 0 are what the design needs; what happens from week 0 on is
    the result.
    """
    import pyfixest as pf

    data = table[table["filing_year"].isin([*comparison, event_year])].dropna(subset=[outcome])
    data = data.assign(event=(data["filing_year"] == event_year).astype(int))
    bins = _bins(data["week"], data["event"] == 1)
    terms, rows = [], []
    for i, (first, last) in enumerate(bins):
        low = -np.inf if first is None else first
        high = np.inf if last is None else last
        inside = data["week"].between(low, high)
        row = {
            "bin": _bin_label(first, last),
            "weeks": [first, last],
            "n_event": int((inside & (data["event"] == 1)).sum()),
            "n_comparison": int((inside & (data["event"] == 0)).sum()),
        }
        if low <= reference <= high:
            rows.append(row | {"estimate": 0.0, "se": None, "low": None, "high": None, "p": None})
            continue
        data[f"bin{i}"] = inside.astype(int)
        data[f"event{i}"] = data[f"bin{i}"] * data["event"]
        terms += [f"bin{i}", f"event{i}"]
        rows.append(row | {"term": f"event{i}"})
    fit = pf.feols(
        f"{outcome} ~ {' + '.join(terms)} | cik + filing_year", data=data, vcov={"CRV1": "cik"}
    )
    tidy = fit.tidy()
    return [row | _estimate(tidy.loc[row.pop("term")]) if "term" in row else row for row in rows]


def has_heading(texts: pd.Series, kind: str) -> pd.Series:
    """Whether each text is a heading of the kind the SEC's 2020 rules brought in.

    `summary`: a short line with both "summary" and "risk" (a summary of risk
    factors). `general`: a short line beginning "General risk" (the caption
    the rules ask generic risks to be gathered under).
    """
    short = texts.str.split().str.len() <= 12
    lower = texts.str.lower()
    if kind == "summary":
        return short & lower.str.contains("summary") & lower.str.contains("risk")
    if kind == "general":
        return short & lower.str.match(r"\W*general risks?\b")
    raise ValueError(kind)


RULES_EFFECTIVE = pd.Timestamp("2020-11-09")  # the SEC's amended risk-factor rules


def covid_by_month(table: pd.DataFrame, years: tuple[int, ...] = (2018, 2019, 2020)) -> list[dict]:
    """The exposure check: by month filed, how many filings mention COVID at all.

    Before February 2020 the word did not exist, so any share above zero
    there is an error in the dates; from April 2020 on nearly every filing
    should have it.
    """
    part = table[table["filing_year"].isin(years)]
    month = part["date_filed"].dt.to_period("M").astype(str)
    grouped = part.groupby(month).agg(
        filings=("cik", "size"),
        mention=("mentions_covid", "mean"),
        share_of_sentences=("share_covid", "mean"),
    )
    return [
        {
            "month": m,
            "filings": int(r.filings),
            "mention": float(r.mention),
            "share_of_sentences": float(r.share_of_sentences),
        }
        for m, r in grouped.iterrows()
        if r.filings >= 20
    ]


def survived_one_year(chains: pd.DataFrame) -> dict:
    """Of added sentences that could be checked a year later, the share still there."""
    watched = chains[chains["event"] | (chains["duration"] > 1)]
    return {
        "watched": len(watched),
        "still_there": float((watched["duration"] > 1).mean()) if len(watched) else None,
    }


def year_after(cfg: Config, table: pd.DataFrame) -> dict:
    """Fiscal 2020, described and not explained.

    Reports on fiscal 2020 were written during the pandemic and, for most,
    under the SEC's amended rules, so nothing here separates the two. It
    gives: how fiscal 2020 compares with 2019 on each outcome; the few
    fiscal-2020 reports filed before the rules took effect against those
    filed after; how many sections have the headings the rules ask for; and
    whether what was added for fiscal 2019 was still there a year later.
    """
    shares = [
        "share_new",
        "share_removed",
        "share_near",
        "growth",
        "share_new_covid",
        "share_new_ex_covid",
    ]
    paired = table[table["unique"] & table["filing_year"].isin([2018, 2019, 2020])]
    by_year = {
        str(year): {"filings": len(part)} | {s: float(part[s].mean()) for s in shares}
        for year, part in paired.groupby("filing_year")
    }
    last = paired[paired["filing_year"] == 2020]
    after = last["date_filed"] >= RULES_EFFECTIVE
    rules = {
        name: {"filings": len(part)} | {s: float(part[s].mean()) for s in shares}
        for name, part in (("before", last[~after]), ("after", last[after]))
    }

    headings = {}
    for year in (2018, 2019, 2020):
        text = read_sentences(cfg, year, ["cik", "text"])
        text = text[text["cik"].isin(paired.loc[paired["filing_year"] == year, "cik"])]
        filings = text["cik"].nunique()
        headings[str(year)] = {"filings": int(filings)} | {
            kind: float(text.loc[has_heading(text["text"], kind), "cik"].nunique() / filings)
            for kind in ("summary", "general")
        }

    chains = pd.read_parquet(cfg.interim_dir / "chains.parquet")
    born = chains[chains["birth_year"] == 2019]
    text = read_sentences(cfg, 2019, ["cik", "sent_idx", "text"])
    born = born.merge(text, on=["cik", "sent_idx"])
    covid = born["text"].str.contains(COVID).to_numpy()
    earlier = chains[chains["birth_year"].isin(COMPARISON)]
    return {
        "by_year": by_year,
        "fiscal_2020_by_rules": rules,
        "share_filed_after_rules": float(after.mean()),
        "headings": headings,
        "added_in_2019": {
            "mention_covid": survived_one_year(born[covid]) | {"sentences": int(covid.sum())},
            "others": survived_one_year(born[~covid]) | {"sentences": int((~covid).sum())},
        },
        "added_in_2016_2018": survived_one_year(earlier) | {"sentences": len(earlier)},
    }
