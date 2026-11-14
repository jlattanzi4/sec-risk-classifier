import json
import zipfile

import pandas as pd
import pytest

from pipeline.candidates import build_candidates
from pipeline.embed import Embedder, build_embeddings, build_similarity
from pipeline.io import write_parquet
from pipeline.labels import build_filing_metrics, build_labels
from pipeline.peers import build_peers
from pipeline.report_json import (
    build_report_json,
    closest_to_median,
    company_name,
    example_redline,
    select,
)
from tests.conftest import write_tables
from tests.test_cosine import Letters

KEPT = "We depend on a small number of large customers for most of our revenue each year."
OLD = "Our business could be harmed if we lose any of our key personnel or senior managers."
EDITED = "Our business could be harmed if we lose many of our key personnel or senior managers."
GONE = "We have a long history of operating losses and may never be profitable again at all."
FRESH = "A pandemic could disrupt our supply chain and reduce demand for the products that we sell."
FIRST = "Zinc quotas vexed jumpy buyers of quartz for five weeks."

SHARES = ["share_exact", "share_near", "share_paraphrase", "share_new", "share_removed"]


def metrics(rows):
    """Filing metrics from (cik, year, sentences, exact, near, reworded, new, removed)."""
    columns = ["cik", "filing_year", "n_sentences", *SHARES]
    return pd.DataFrame(rows, columns=columns)


def test_the_example_is_the_filing_closest_to_the_median_filing():
    table = metrics(
        [
            (1, 2019, 200, 0.90, 0.05, 0.00, 0.05, 0.02),
            (2, 2019, 200, 0.79, 0.10, 0.01, 0.10, 0.06),  # nearest the medians below
            (3, 2019, 200, 0.50, 0.10, 0.05, 0.35, 0.30),
            (4, 2018, 200, 0.80, 0.10, 0.01, 0.09, 0.05),
            (5, 2018, 200, 0.80, 0.10, 0.01, 0.09, 0.05),
        ]
    )
    assert closest_to_median(table, year=2019, sentences=(150, 400)) == (2, 2019)


def test_only_filings_of_the_year_and_the_size_asked_for_are_candidates():
    table = metrics(
        [
            (1, 2019, 900, 0.80, 0.10, 0.01, 0.09, 0.05),  # the median filing, but too long
            (2, 2018, 200, 0.80, 0.10, 0.01, 0.09, 0.05),  # and in another year
            (3, 2019, 150, 0.70, 0.15, 0.02, 0.13, 0.08),
            (4, 2019, 400, 0.99, 0.01, 0.00, 0.00, 0.00),
        ]
    )
    assert closest_to_median(table, year=2019, sentences=(150, 400)) == (3, 2019)


def test_a_tie_goes_to_the_lowest_cik():
    table = metrics(
        [
            (9, 2019, 200, 0.80, 0.10, 0.01, 0.09, 0.05),
            (7, 2019, 200, 0.80, 0.10, 0.01, 0.09, 0.05),
            (8, 2019, 200, 0.70, 0.10, 0.01, 0.19, 0.05),
        ]
    )
    assert closest_to_median(table, year=2019, sentences=(150, 400)) == (7, 2019)


@pytest.fixture
def submissions(tmp_path):
    """SEC's bulk submissions archive, with the one company the fixtures use."""
    path = tmp_path / "submissions.zip"
    record = {
        "name": "KELLANOVA",
        "formerNames": [
            {
                "name": "KELLOGG CO",
                "from": "1994-04-29T00:00:00.000Z",
                "to": "2023-09-28T00:00:00.000Z",
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("CIK0000000001.json", json.dumps(record))
    return path


def test_a_company_is_named_as_it_was_when_it_filed(submissions):
    assert company_name(submissions, 1, pd.Timestamp("2020-02-24")) == "KELLOGG CO"
    assert company_name(submissions, 1, pd.Timestamp("2024-02-20")) == "KELLANOVA"


def test_a_company_missing_from_the_archive_has_no_name(submissions):
    assert company_name(submissions, 2, pd.Timestamp("2020-02-24")) is None


@pytest.fixture
def built(cfg):
    write_tables(cfg, {(1, 2017): [FIRST, KEPT, OLD, GONE], (1, 2018): [KEPT, EDITED, FRESH]})
    listing = pd.read_parquet(cfg.interim_dir / "filings.parquet").assign(
        accession=["0000055067-18-000004", "0000055067-19-000007"],
        date_filed=pd.to_datetime(["2018-02-20", "2019-02-25"]),
    )
    write_parquet(listing, cfg.interim_dir / "filings.parquet")
    build_candidates(cfg, workers=1)
    build_embeddings(cfg, embedder=Embedder(Letters(), "letters", dim=26))
    build_similarity(cfg)
    build_labels(cfg, tau_v=0.3, tau_p=0.995)
    build_peers(cfg)
    build_filing_metrics(cfg)
    return cfg


def test_the_redline_gives_this_year_s_sentences_in_order_with_what_each_matched(
    built, submissions
):
    redline = example_redline(built, 1, 2018, submissions=submissions)
    assert [s["label"] for s in redline["sentences"]] == ["exact", "near", "new"]
    assert [s["text"] for s in redline["sentences"]] == [KEPT, EDITED, FRESH]
    assert [s["chars"] for s in redline["sentences"]] == [len(KEPT), len(EDITED), len(FRESH)]
    # Only an edited sentence needs last year's wording shown beside it.
    assert [s.get("previous") for s in redline["sentences"]] == [None, OLD, None]


def test_removed_sentences_sit_where_they_were_cut_from(built, submissions):
    redline = example_redline(built, 1, 2018, submissions=submissions)
    # FIRST came before anything that was kept: it sits above the first sentence (-1).
    # GONE followed OLD, which lives on as sentence 1 (EDITED).
    assert [(r["text"], r["after"]) for r in redline["removed"]] == [(FIRST, -1), (GONE, 1)]


def test_the_redline_says_which_filing_it_is(built, submissions):
    redline = example_redline(built, 1, 2018, submissions=submissions)
    assert redline["cik"] == 1 and redline["year"] == 2018
    assert redline["name"] == "KELLOGG CO" and redline["date_filed"] == "2019-02-25"
    assert redline["url"] == (
        "https://www.sec.gov/Archives/edgar/data/1/000005506719000007/"
        "0000055067-19-000007-index.htm"
    )
    assert redline["shares"]["exact"] == pytest.approx(1 / 3)
    assert redline["shares"]["removed"] == pytest.approx(2 / 4)


def test_select_copies_the_paths_asked_for_and_nothing_else():
    source = {"a": {"b": 1.23456789, "c": 2}, "d": [1, 2], "unused": "x"}
    assert select(source, ["a.b", "d"], "measure.json") == {"a": {"b": 1.234568}, "d": [1, 2]}


def test_a_field_the_page_needs_but_a_results_file_lacks_is_named():
    with pytest.raises(KeyError, match=r"measure\.json has no shares\.by_year"):
        select({"shares": {}}, ["shares.by_year"], "measure.json")


def test_the_report_json_holds_each_results_file_s_fields_and_the_example(built, submissions):
    built.results_dir.mkdir()
    for name in ("measure.json", "explain.json", "test2020.json"):
        (built.results_dir / name).write_text(json.dumps({"kept": {"value": 1}, "unused": 2}))
    fields = {"measure": ["kept.value"], "explain": ["kept"], "test2020": ["kept"]}

    path = build_report_json(
        built, fields=fields, example_year=2018, example_sentences=(1, 10), submissions=submissions
    )
    report = json.loads(path.read_text())

    assert path == built.root / "docs" / "data" / "report.json"
    assert set(report) == {"measure", "explain", "test2020", "example"}
    assert report["measure"] == {"kept": {"value": 1}}
    assert report["example"]["cik"] == 1 and len(report["example"]["sentences"]) == 3
    # The rule that chose the example travels with it, for the page to state.
    assert report["example"]["rule"] == {"year": 2018, "sentences": [1, 10]}


def test_building_twice_writes_the_same_bytes(built, submissions):
    built.results_dir.mkdir()
    for name in ("measure.json", "explain.json", "test2020.json"):
        (built.results_dir / name).write_text(json.dumps({"kept": 1}))
    kwargs = {
        "fields": {"measure": ["kept"], "explain": ["kept"], "test2020": ["kept"]},
        "example_year": 2018,
        "example_sentences": (1, 10),
        "submissions": submissions,
    }
    first = build_report_json(built, **kwargs).read_bytes()
    assert build_report_json(built, **kwargs).read_bytes() == first
