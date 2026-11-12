"""Gather milestone 3's results into `results/explain.json`.

Everything `docs/explain.md` quotes: how many sentences were followed and how
many could not be, the survival curves, the proportional-hazards check, the
hazard model, and the variance decomposition on both scales.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import Config
from pipeline.hazard import company_sample, cox_check, fit_hazard, person_periods
from pipeline.survival import life_table, median_survival
from pipeline.variance import fit_variance

HAZARD_COMPANIES = 2000


def _curve(table: pd.DataFrame) -> list[dict]:
    columns = ["age", "at_risk", "removed", "hazard", "survival", "low", "high"]
    return [
        {
            k: (int(v) if k in ("age", "at_risk", "removed") else round(float(v), 5))
            for k, v in row.items()
        }
        for row in table[columns].to_dict("records")
    ]


def _curves(chains: pd.DataFrame, by: str) -> dict:
    table = life_table(chains, by=by)
    return {
        str(value): {
            "chains": int((chains[by] == value).sum()),
            "median": median_survival(part),
            "curve": _curve(part),
        }
        for value, part in table.groupby(by)
    }


def survival_results(chains: pd.DataFrame) -> dict:
    """Survival overall and by the splits the write-up shows."""
    overall = life_table(chains)
    thirds = chains["n_words"].quantile([1 / 3, 2 / 3]).round().astype(int).tolist()
    length = pd.cut(
        chains["n_words"],
        [0, thirds[0], thirds[1], 10**6],
        labels=[
            f"up to {thirds[0]} words",
            f"{thirds[0] + 1} to {thirds[1]} words",
            f"over {thirds[1]} words",
        ],
    ).astype(str)
    born = pd.cut(
        chains["birth_year"],
        [2006, 2010, 2014, 2017, 2020],
        labels=["2007-10", "2011-14", "2015-17", "2018-20"],
    )
    return {
        "overall": {
            "chains": len(chains),
            "median": median_survival(overall),
            "curve": _curve(overall),
        },
        "by_peer_shared": _curves(
            chains.assign(group=chains["peer_shared"].map({True: "shared", False: "own"})), "group"
        ),
        "by_length": _curves(chains.assign(group=length), "group"),
        "by_birth": _curves(chains.assign(group=born.astype(str)), "group"),
    }


def chain_counts(cfg: Config, chains: pd.DataFrame) -> dict:
    """What was followed, and what could not be."""
    files = sorted((cfg.interim_dir / "sentences").glob("year=*.parquet"))
    all_sentences = sum(pq.read_metadata(f).num_rows for f in files)
    metrics = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet", columns=["n_sentences"])
    never_watched = (~chains["event"]) & (chains["duration"] == 1)
    return {
        "chains": len(chains),
        "companies": int(chains["cik"].nunique()),
        "removed": int(chains["event"].sum()),
        "still_there_when_the_data_end": int((~chains["event"]).sum()),
        "never_watched": int(never_watched.sum()),
        "peer_shared": int(chains["peer_shared"].sum()),
        "sentences_in_first_filings": int(all_sentences - metrics["n_sentences"].sum()),
        "by_birth_year": {
            str(int(y)): int(n) for y, n in chains.groupby("birth_year").size().items()
        },
    }


def build_explain_results(cfg: Config, hazard_companies: int = HAZARD_COMPANIES) -> Path:
    """Run the survival, hazard and variance analyses and save them."""
    chains = pd.read_parquet(cfg.interim_dir / "chains.parquet")
    sample = company_sample(chains, n=hazard_companies)
    cox = cox_check(sample)
    periods = person_periods(sample)
    varies = cox["proportional_hazards_rejected"]
    hazard = {"constant": fit_hazard(periods, vary_by_age=False)}
    if varies:
        hazard["by_age"] = fit_hazard(periods, vary_by_age=True)

    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year", "sic_division"]
    )
    metrics = pd.read_parquet(cfg.interim_dir / "filing_metrics.parquet").merge(
        filings, on=["cik", "filing_year"]
    )
    out = {
        "counts": chain_counts(cfg, chains),
        "survival": survival_results(chains),
        "hazard_sample": {
            "companies": int(sample["cik"].nunique()),
            "chains": len(sample),
            "person_period_rows": len(periods),
        },
        "cox": cox,
        "effect_varies_by_age": bool(varies),
        "hazard": hazard,
        "variance": {scale: fit_variance(metrics, scale=scale) for scale in ("share", "logit")},
    }
    cfg.results_dir.mkdir(exist_ok=True)
    path = cfg.results_dir / "explain.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path
