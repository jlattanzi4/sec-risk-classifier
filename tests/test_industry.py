import json
import zipfile

import pytest

from pipeline.industry import read_sic_codes, sic_division


@pytest.mark.parametrize(
    ("sic", "division"),
    [
        (999, "A"),
        (1000, "B"),
        (3999, "D"),
        (4000, "E"),
        (6799, "H"),
        (6800, None),  # a gap between divisions
        (9999, "K"),
        (None, None),
    ],
)
def test_sic_division_boundaries(sic, division):
    assert sic_division(sic) == division


def test_reads_sic_codes_from_submissions_zip(tmp_path):
    zip_path = tmp_path / "submissions.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CIK0000320193.json", json.dumps({"cik": "320193", "sic": "7372"}))
        zf.writestr("CIK0001018724.json", json.dumps({"cik": "1018724", "sic": ""}))
        # A paged history file for the same CIK must not be mistaken for the main record.
        zf.writestr("CIK0000320193-submissions-001.json", json.dumps({"filings": []}))
    assert read_sic_codes(zip_path, [320193, 1018724, 999]) == {
        320193: 7372,
        1018724: None,
        999: None,
    }
