import json

import pandas as pd
import pytest

from pipeline.candidates import build_candidates
from pipeline.embed import Embedder, build_embeddings, build_similarity
from pipeline.io import write_parquet
from pipeline.labels import build_filing_metrics, build_labels
from pipeline.peers import build_peers
from pipeline.report_data import build_measure_results, examples, peer_reach, shares
from tests.conftest import write_tables
from tests.test_cosine import Letters

KEPT = "We depend on a small number of large customers for most of our revenue each year."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
EDITED = "Our business could be harmed if we lose many of our key personnel or senior managers."
GONE = "We have a long history of operating losses and may never be profitable again at all."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for the products that we sell."
COMMON = "Our stock price may be volatile and you could lose all or part of your investment."


@pytest.fixture
def built(cfg):
    filings = {(1, 2017): [KEPT, OLD, GONE, COMMON], (1, 2018): [KEPT, EDITED, FRESH, COMMON]}
    # Seven other companies file COMMON in 2018, so it reaches at least 5 other groups.
    for cik, name in zip(
        range(2, 9), "alpha bravo charlie delta echo foxtrot golf".split(), strict=True
    ):
        filings[(cik, 2017)] = [
            f"Company {name} wrote this sentence for itself alone last year.",
            COMMON,
        ]
        filings[(cik, 2018)] = [
            f"Company {name} wrote this sentence for itself alone this year.",
            COMMON,
        ]
    write_tables(cfg, filings)
    listing = pd.read_parquet(cfg.interim_dir / "filings.parquet").assign(
        accession="0000000000-18-000001"
    )
    write_parquet(listing, cfg.interim_dir / "filings.parquet")
    build_candidates(cfg, workers=1)
    build_embeddings(cfg, embedder=Embedder(Letters(), "letters", dim=26))
    build_similarity(cfg)
    build_labels(cfg, tau_v=0.3, tau_p=0.995)
    build_peers(cfg)
    build_filing_metrics(cfg)
    cfg.results_dir.mkdir()
    (cfg.results_dir / "thresholds.json").write_text(json.dumps({"tau_v": 0.3, "tau_p": 0.995}))
    for name in ("embedders.json", "baselines.json"):
        (cfg.results_dir / name).write_text("{}")
    return cfg


def test_shares_are_shares_and_add_up(built):
    result = shares(built)
    assert result["filings"] == 8 and result["sentences"] == 4 + 7 * 2
    pooled = result["pooled"]
    assert all(0 <= value <= 1 for value in pooled.values())
    assert pooled["exact"] + pooled["near"] + pooled["paraphrase"] + pooled["new"] == pytest.approx(
        1
    )
    by_chars = result["pooled_chars"]
    assert by_chars["exact"] + by_chars["near"] + by_chars["paraphrase"] + by_chars[
        "new"
    ] == pytest.approx(1)
    assert [row["year"] for row in result["by_year"]] == [2018]
    assert sum(result["share_new_histogram"]["counts"]) == 8


def test_peer_reach_counts_other_groups(built):
    reach = peer_reach(built)
    in_2018 = next(row for row in reach["by_year"] if row["year"] == 2018)
    # COMMON is 8 of the 18 sentences filed in 2018, and the only one anyone shares.
    assert in_2018["all"]["k5"] == pytest.approx(8 / 18) and in_2018["all"]["k10"] == 0
    assert reach["most_shared_last_year"][0] == {"text": COMMON, "other_groups": 7}


def test_examples_follow_the_rule_and_carry_their_partner(built):
    rows = examples(built, n_filings=8, per_label=1)
    by_label = {row["label"]: row for row in rows}
    assert set(by_label) == {"exact", "near", "new"}  # nothing in the fixture is reworded
    assert by_label["near"]["text"] == EDITED and by_label["near"]["previous_text"] == OLD
    assert by_label["near"]["jaccard"] == pytest.approx(7 / 17, abs=1e-3)
    assert by_label["new"]["previous_text"] is None
    assert by_label["exact"]["accession"] == "0000000000-18-000001"


def test_the_results_file_has_every_section(built):
    saved = json.loads(build_measure_results(built).read_text())
    assert {
        "model",
        "tau_v",
        "tau_p",
        "shares",
        "partner_agreement",
        "peers",
        "thresholds",
        "embedders",
        "baselines",
        "examples",
    } <= set(saved)
