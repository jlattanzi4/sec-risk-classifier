import json
import shutil
import zipfile

import pandas as pd
import pytest

from pipeline import stages
from pipeline.config import Config
from pipeline.industry import SUBMISSIONS_URL
from pipeline.run import summarize_filings
from pipeline.stages import build_filings

FILINGS_COLUMNS = [
    "cik",
    "filing_year",
    "item_1a_length",
    "report_date",
    "date_filed",
    "form_type",
    "accession",
    "n_candidates",
    "match_status",
    "sic",
    "sic_division",
]


@pytest.fixture
def cfg(tmp_path):
    return Config(
        root=tmp_path,
        raw_dir=tmp_path / "data/raw/edgar",
        interim_dir=tmp_path / "data/interim",
        sec_user_agent="Jane Doe jane@example.com",
    )


def write_corpus(cfg: Config, rows: list[tuple[str, int, str]]) -> None:
    """The processed corpus, with the columns and dtypes of the real parquet."""
    path = cfg.root / "data/processed/risk_factors_2006_2020.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "filename": [f"{cik}_{year}.htm" for cik, year, _ in rows],
            "cik": [cik for cik, _, _ in rows],
            "filing_year": [year for _, year, _ in rows],
            "section_1A": [text for _, _, text in rows],
            "item_1a_length": [len(text) for _, _, text in rows],
        }
    ).to_parquet(path)


def submission(sic: str, filings: list[tuple[str, str, str, str]]) -> str:
    """One company's record in SEC's submissions zip."""
    return json.dumps(
        {
            "sic": sic,
            "filings": {
                "recent": {
                    "form": [f[0] for f in filings],
                    "filingDate": [f[1] for f in filings],
                    "reportDate": [f[2] for f in filings],
                    "accessionNumber": [f[3] for f in filings],
                },
                "files": [],
            },
        }
    )


@pytest.fixture
def fake_fetch(monkeypatch, tmp_path):
    """Serve SEC's submissions zip from a fixture with two companies."""
    submissions = tmp_path / "fixture-submissions.zip"
    with zipfile.ZipFile(submissions, "w") as zf:
        zf.writestr(
            "CIK0001453122.json",
            submission(
                "3714",
                [
                    ("10-K/A", "2016-02-18", "2015-10-31", "0001554795-16-000063"),
                    ("10-K", "2016-02-13", "2015-10-31", "0001554795-16-000045"),
                ],
            ),
        )
        zf.writestr(
            "CIK0001157377.json",
            submission("", [("10-KT", "2015-09-30", "2015-06-30", "0001157377-15-000010")]),
        )
    calls = []

    def fetch(url, dest, user_agent, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(submissions, dest)
        return dest

    monkeypatch.setattr(stages, "fetch", fetch)
    return calls


def test_build_filings_on_fixtures(cfg, fake_fetch):
    write_corpus(
        cfg,
        [
            ("1453122", 2015, "A report on fiscal 2015, filed (and amended) in 2016."),
            ("0001157377", 2015, "A transition report, 10-KT."),
            ("1606163", 2014, "A company SEC's submissions data doesn't have."),
        ],
    )
    path = build_filings(cfg)

    assert path == cfg.interim_dir / "filings.parquet"
    out = pd.read_parquet(path)
    assert list(out.columns) == FILINGS_COLUMNS
    assert len(out) == 3
    rows = out.set_index(["cik", "filing_year"])
    assert rows.loc[(1453122, 2015), "date_filed"] == pd.Timestamp("2016-02-13")
    assert rows.loc[(1453122, 2015), "report_date"] == pd.Timestamp("2015-10-31")
    assert rows.loc[(1453122, 2015), "accession"] == "0001554795-16-000045"
    assert rows.loc[(1453122, 2015), "match_status"] == "unique"
    assert rows.loc[(1453122, 2015), "sic"] == 3714
    assert rows.loc[(1453122, 2015), "sic_division"] == "D"
    assert rows.loc[(1157377, 2015), "form_type"] == "10-KT"
    assert pd.isna(rows.loc[(1157377, 2015), "sic"])
    assert rows.loc[(1606163, 2014), "match_status"] == "none"
    assert pd.isna(rows.loc[(1606163, 2014), "sic_division"])
    assert str(out["sic"].dtype) == "Int64"
    # One download: the submissions zip holds both the filing dates and the SIC codes.
    assert fake_fetch == [SUBMISSIONS_URL]

    fake_fetch.clear()
    assert build_filings(cfg) == path
    assert fake_fetch == []


def test_summary_reports_the_gate_share():
    df = pd.DataFrame(
        {
            "match_status": ["unique", "multiple", "amendment_only", "none"],
            "sic_division": ["D", None, "D", "I"],
        }
    )
    text = summarize_filings(df)
    assert "dated by an original 10-K (unique or multiple): 50.0%" in text
    assert "with a SIC division: 75.0%" in text
    assert "  D  Manufacturing" in text
