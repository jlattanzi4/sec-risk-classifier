"""Gather milestone 4's results into `results/test2020.json`."""

import json
from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.did import (
    COMPARISON,
    EVENT_YEAR,
    OUTCOMES,
    analysis_table,
    covid_by_month,
    did,
    event_study,
    main_sample,
    placebos,
    year_after,
)

YEARS = (*COMPARISON, EVENT_YEAR)
KEY_OUTCOMES = ("share_new", "share_removed")


def sample_counts(table: pd.DataFrame) -> dict:
    """How many filings are left after each condition of the main sample."""
    paired = table[table["filing_year"].isin(YEARS)]
    december = paired[paired["december"]]
    unchanged = december[december["same_fye"]]
    unique = unchanged[unchanged["unique"]]
    return {
        "paired_filings": len(paired),
        "december_year_end": len(december),
        "year_end_unchanged": len(unchanged),
        "single_matching_filing": len(unique),
        "companies": int(unique["cik"].nunique()),
        "by_year": {
            str(year): {
                "filings": len(part),
                "late": float(part["late"].mean()),
                "late_usual": float(part["late_usual"].mean()),
            }
            for year, part in unique.groupby("filing_year")
        },
    }


def exposure_check(sample: pd.DataFrame) -> dict:
    """Does filing late in 2020 actually mean writing about COVID?"""
    out = {}
    for year, part in sample[sample["filing_year"].isin([2018, 2019])].groupby("filing_year"):
        out[str(year)] = {
            name: {
                "filings": len(group),
                "mention": float(group["mentions_covid"].mean()),
                "share_of_sentences": float(group["share_covid"].mean()),
                "share_new_covid": float(group["share_new_covid"].mean()),
            }
            for name, group in (("early", part[~part["late"]]), ("late", part[part["late"]]))
        }
    event = sample[sample["filing_year"] == EVENT_YEAR]
    by_week = event.groupby("week").agg(filings=("cik", "size"), mention=("mentions_covid", "mean"))
    out["by_week"] = [
        {"week": int(week), "filings": int(row.filings), "mention": float(row.mention)}
        for week, row in by_week.iterrows()
        if -8 <= week <= 12
    ]
    return out


def robustness(table: pd.DataFrame, outcome: str) -> dict:
    """The main estimate with one thing changed at a time."""
    sample = main_sample(table)
    near_cutoff = sample[sample["week"].between(-3, 2)]
    return {
        "main": did(sample, outcome),
        "exposure by last year's filing date": did(sample, outcome, exposure="late_usual"),
        "no company fixed effects": did(sample, outcome, fixed_effects=False),
        "with filings that have several candidate 10-Ks": did(
            main_sample(table, require_unique=False), outcome
        ),
        "with companies whose fiscal year end changed": did(
            main_sample(table, require_same_fye=False), outcome
        ),
        "within three weeks of the cutoff": did(near_cutoff, outcome),
    }


def build_did_results(cfg: Config) -> Path:
    """Run the 2020 analysis and save it."""
    table = analysis_table(cfg)
    sample = main_sample(table)
    out = {
        "sample": sample_counts(table),
        "exposure": exposure_check(sample),
        "covid_by_month": covid_by_month(table[table["december"] & table["unique"]]),
        "did": {outcome: did(sample, outcome) for outcome in OUTCOMES},
        "robustness": {outcome: robustness(table, outcome) for outcome in KEY_OUTCOMES},
        "event_study": {
            outcome: event_study(sample, outcome)
            for outcome in ("share_new", "share_removed", "share_new_ex_covid")
        },
        "placebos": {outcome: placebos(sample, outcome) for outcome in OUTCOMES},
        "year_after": year_after(cfg, table),
    }
    cfg.results_dir.mkdir(exist_ok=True)
    path = cfg.results_dir / "test2020.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path
