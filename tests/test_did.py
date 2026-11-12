import numpy as np
import pandas as pd
import pytest

from pipeline.did import (
    analysis_table,
    did,
    event_study,
    has_heading,
    main_sample,
    placebos,
)
from pipeline.io import write_parquet
from tests.conftest import write_tables

# In the simulated panels a company files in the same week every year, so the week dummies are
# absorbed by the company effects; pyfixest says so each time.
pytestmark = pytest.mark.filterwarnings("ignore:(?s).*multicollinearity.*")

KEPT = "We depend on a small number of large customers for most of our revenue each year."
GONE = "We have a long history of operating losses and may never be profitable again at all."
FRESH = "Demand for the products that we sell could fall if our customers lose confidence in us."
VIRUS = "The COVID-19 outbreak could disrupt our supply chain and reduce demand for our products."


def filing_rows(*rows):
    columns = ["cik", "filing_year", "report_date", "date_filed", "match_status", "sic_division"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["report_date"], frame["date_filed"] = (
        pd.to_datetime(frame["report_date"]),
        pd.to_datetime(frame["date_filed"]),
    )
    return frame


@pytest.fixture
def table(cfg):
    """Four companies with fiscal 2018 and 2019 filings; what differs is when and how they file."""
    sentences, dates = {}, []
    spec = {
        1: ("12-31", "2019-03-10", "2020-03-10", "unique"),  # the day before the cutoff, both years
        2: ("12-31", "2019-03-11", "2020-03-11", "unique"),  # on the cutoff
        3: ("06-30", "2018-09-01", "2019-09-01", "unique"),  # fiscal year ends in June
        4: ("12-31", "2019-02-20", "2020-03-20", "multiple"),  # early last year, late this year
    }
    for cik, (fye, filed_2018, filed_2019, status) in spec.items():
        sentences[(cik, 2018)] = [KEPT, GONE]
        sentences[(cik, 2019)] = [KEPT, FRESH, VIRUS, VIRUS.replace("supply chain", "factories")]
        dates.append((cik, 2018, f"2018-{fye}", filed_2018, status, "D"))
        dates.append((cik, 2019, f"2019-{fye}", filed_2019, status, "D"))
    # Company 5 moved its fiscal year end from September to December.
    sentences[(5, 2018)], sentences[(5, 2019)] = [KEPT], [KEPT]
    dates += [
        (5, 2018, "2018-09-30", "2018-12-01", "unique", "D"),
        (5, 2019, "2019-12-31", "2020-03-01", "unique", "D"),
    ]
    write_tables(cfg, sentences)
    write_parquet(filing_rows(*dates), cfg.interim_dir / "filings.parquet")
    labels = []
    for cik in spec:
        labels += [
            (cik, "cur", 0, "exact", 0),
            (cik, "cur", 1, "new", -1),
            (cik, "cur", 2, "new", -1),
            (cik, "cur", 3, "new", -1),
            (cik, "prev", 0, "exact", 0),
            (cik, "prev", 1, "removed", -1),
        ]
    labels += [(5, "cur", 0, "exact", 0), (5, "prev", 0, "exact", 0)]
    write_parquet(
        pd.DataFrame(labels, columns=["cik", "side", "sent_idx", "label", "link_idx"]),
        cfg.interim_dir / "labels" / "pair=2019.parquet",
    )
    metrics = pd.DataFrame(
        {
            "cik": [1, 2, 3, 4, 5],
            "filing_year": 2019,
            "n_sentences": [4, 4, 4, 4, 1],
            "n_prev": [2, 2, 2, 2, 1],
            "share_exact": [0.25] * 4 + [1.0],
            "share_near": 0.0,
            "share_paraphrase": 0.0,
            "share_new": [0.75] * 4 + [0.0],
            "share_removed": [0.5] * 4 + [0.0],
        }
    )
    write_parquet(metrics, cfg.interim_dir / "filing_metrics.parquet")
    return analysis_table(cfg).set_index("cik")


def test_exposure_follows_the_calendar_date_in_a_leap_year(table):
    assert not table.loc[1, "late"] and table.loc[2, "late"]
    assert table.loc[1, "days_from_cutoff"] == -1 and table.loc[2, "days_from_cutoff"] == 0
    assert table.loc[1, "week"] == -1 and table.loc[2, "week"] == 0


def test_last_year_s_filing_date_gives_the_usual_exposure(table):
    assert (
        not table.loc[1, "late_usual"] and table.loc[2, "late_usual"]
    )  # 2019-03-10 and 2019-03-11
    assert table.loc[4, "late"] and not table.loc[4, "late_usual"]


def test_the_main_sample_is_december_unchanged_and_unique(table):
    kept = main_sample(table.reset_index(), years=(2019,))["cik"].tolist()
    assert kept == [1, 2]  # 3: June; 4: several candidate filings; 5: moved


def test_growth_and_covid_shares(table):
    assert table.loc[1, "growth"] == pytest.approx(np.log(4 / 2))
    assert table.loc[1, "mentions_covid"]
    assert table.loc[1, "share_new_covid"] == pytest.approx(2 / 4)
    assert table.loc[1, "share_new_ex_covid"] == pytest.approx(1 / 4)
    assert not table.loc[5, "mentions_covid"] and table.loc[5, "share_new_covid"] == 0


def panel(effect, n=1500, seed=0, post_weeks_only=True):
    """Companies observed for fiscal 2016-2019; late filers get `effect` in 2019."""
    rng = np.random.default_rng(seed)
    week = rng.choice(
        [-5, -4, -3, -2, -1, 0, 1, 2, 4, 8],
        n,
        p=[0.03, 0.05, 0.15, 0.27, 0.1, 0.17, 0.05, 0.1, 0.05, 0.03],
    )
    rows = []
    company = rng.normal(0, 0.05, n)
    for year in (2016, 2017, 2018, 2019):
        late = week >= 0
        bump = effect * late * (year == 2019)
        y = 0.15 + 0.04 * late + company + 0.01 * (year - 2016) + bump + rng.normal(0, 0.08, n)
        rows.append(
            pd.DataFrame(
                {
                    "cik": np.arange(n),
                    "filing_year": year,
                    "week": week,
                    "late": late,
                    "share_new": y,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_a_true_effect_is_recovered():
    result = did(panel(0.03), "share_new")
    assert result["low"] < 0.03 < result["high"] and result["low"] > 0
    assert result["filings"] == 6000 and result["companies"] == 1500


def test_no_effect_gives_an_interval_around_zero():
    result = did(panel(0.0, seed=1), "share_new")
    assert result["low"] < 0 < result["high"]


def test_cell_means_match_a_direct_count():
    data = panel(0.03)
    result = did(data, "share_new")
    direct = data[(data["filing_year"] == 2019) & data["late"]]["share_new"].mean()
    assert result["means"]["late"]["event"] == pytest.approx(direct)
    assert set(result["means"]) == {"early", "late"} and set(result["means"]["early"]) == {
        "comparison",
        "event",
    }


def test_a_company_seen_once_does_not_move_the_estimate():
    data = panel(0.03)
    once = pd.DataFrame(
        {"cik": [99999], "filing_year": [2019], "week": [1], "late": [True], "share_new": [0.9]}
    )
    assert did(pd.concat([data, once]), "share_new")["estimate"] == pytest.approx(
        did(data, "share_new")["estimate"], abs=1e-6
    )


def test_placebos_leave_the_event_year_out():
    data = panel(0.2)  # a huge 2019 effect that must not leak in
    results = placebos(data, "share_new")
    assert [r["event_year"] for r in results] == [2017, 2018]
    assert all(r["low"] < 0 < r["high"] for r in results)
    assert all(r["filings"] == 4500 for r in results)


def test_the_event_study_is_flat_before_and_shifted_after():
    study = event_study(panel(0.05, n=4000, seed=2), "share_new")
    by_bin = {row["bin"]: row for row in study}
    assert by_bin["-2"]["estimate"] == 0 and by_bin["-2"]["low"] is None  # the reference week
    for name in ("-3", "-1"):
        assert by_bin[name]["low"] < 0 < by_bin[name]["high"], name
    for name in ("0", "2"):
        assert by_bin[name]["low"] < 0.05 < by_bin[name]["high"] and by_bin[name]["low"] > 0, name
    assert sum(row["n_event"] for row in study) == 4000


def test_a_thin_week_is_merged_into_its_neighbor():
    data = panel(0.05, n=4000, seed=2)
    data = data[~((data["week"] == 1) & (data["cik"] % 20 != 0))]  # leave week 1 nearly empty
    bins = [row["bin"] for row in event_study(data, "share_new")]
    assert "1" not in bins and "1 to 2" in bins


def test_summary_headings_are_found_and_lookalikes_are_not():
    assert has_heading(
        pd.Series(["Summary of Risk Factors", "Risk Factor Summary", "SUMMARY RISK FACTORS"]),
        "summary",
    ).all()
    assert not has_heading(
        pd.Series(
            [
                "Summary of Significant Accounting Policies",
                "Risk Factors",
                "The following summary of the risks we face is not complete and you should read all of it carefully.",
            ]
        ),
        "summary",
    ).any()
    assert has_heading(pd.Series(["General Risk Factors", "General Risks"]), "general").all()
    assert not has_heading(
        pd.Series(["Risks Related to General Economic Conditions and Other Matters Affecting Us"]),
        "general",
    ).any()
