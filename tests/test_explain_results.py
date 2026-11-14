import json

import numpy as np
import pandas as pd
import pytest

from pipeline.explain_results import build_explain_results, chain_counts, survival_results
from pipeline.io import write_parquet
from tests.conftest import write_tables
from tests.test_hazard import simulate_chains
from tests.test_variance import simulate


def test_survival_is_reported_overall_and_by_each_split():
    chains = simulate_chains(n_companies=30, per_company=40, sigma=0.3, seed=5)
    result = survival_results(chains)
    assert set(result) == {"overall", "by_peer_shared", "by_length", "by_birth"}
    assert set(result["by_peer_shared"]) == {"shared", "own"}
    assert result["overall"]["chains"] == len(chains)
    for split in ("by_peer_shared", "by_length", "by_birth"):
        assert sum(group["chains"] for group in result[split].values()) == len(chains)
    survival = [row["survival"] for row in result["overall"]["curve"]]
    assert survival == sorted(survival, reverse=True) and 0 < survival[-1] < survival[0] < 1


def test_counts_say_what_was_followed_and_what_could_not_be(cfg):
    # The 2017 filing is the company's first: its two sentences have no known age.
    write_tables(
        cfg,
        {
            (1, 2017): ["We may lose customers.", "Costs may rise."],
            (1, 2018): ["We may lose customers.", "A pandemic may come.", "Rates may rise."],
        },
    )
    write_parquet(
        pd.DataFrame({"cik": [1], "filing_year": [2018], "n_sentences": [3]}),
        cfg.interim_dir / "filing_metrics.parquet",
    )
    chains = pd.DataFrame(
        {
            "cik": [1, 1],
            "birth_year": [2018, 2018],
            "duration": [1, 1],
            "event": [False, False],
            "peer_shared": [True, False],
        }
    )
    counts = chain_counts(cfg, chains)
    assert counts["chains"] == 2 and counts["companies"] == 1
    assert counts["removed"] == 0 and counts["never_watched"] == 2
    assert counts["sentences_in_first_filings"] == 2
    assert counts["peer_shared"] == 1 and counts["by_birth_year"] == {"2018": 2}


def test_the_results_file_has_every_section(cfg):
    chains = simulate_chains(n_companies=60, per_company=40, sigma=0.4, seed=6)
    write_parquet(
        chains.assign(sent_idx=np.arange(len(chains))), cfg.interim_dir / "chains.parquet"
    )
    filings = simulate(n_companies=60)
    write_parquet(
        filings[["cik", "filing_year", "sic_division"]], cfg.interim_dir / "filings.parquet"
    )
    write_parquet(
        filings[["cik", "filing_year", "share_new"]].assign(n_sentences=200),
        cfg.interim_dir / "filing_metrics.parquet",
    )
    write_parquet(
        pd.DataFrame({"cik": [1] * 250}), cfg.interim_dir / "sentences" / "year=2010.parquet"
    )

    # A few draws are enough to see that every part is computed and saved; whether a fit has
    # converged is the fits' own tests' business.
    quick = {"warmup": 30, "draws": 30, "chains": 2, "check": False}
    path = build_explain_results(cfg, hazard_companies=40, fits=quick)
    saved = json.loads(path.read_text())

    assert path == cfg.results_dir / "explain.json"
    assert set(saved) == {
        "counts",
        "survival",
        "hazard_sample",
        "cox",
        "effect_varies_by_age",
        "hazard",
        "variance",
    }
    assert saved["hazard_sample"]["companies"] == 40
    assert "constant" in saved["hazard"]
    assert ("by_age" in saved["hazard"]) == saved["effect_varies_by_age"]
    assert set(saved["variance"]) == {"share", "logit"}
    for fit in saved["variance"].values():
        assert sum(part["value"] for part in fit["mean_share"].values()) == pytest.approx(1.0)
