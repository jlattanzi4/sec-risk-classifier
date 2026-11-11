"""Industry for each company: its SIC code from SEC submissions data, grouped into divisions."""

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"

# The SIC system's divisions: letter -> (first code, last code, name).
SIC_DIVISIONS: dict[str, tuple[int, int, str]] = {
    "A": (100, 999, "Agriculture, Forestry, Fishing"),
    "B": (1000, 1499, "Mining"),
    "C": (1500, 1799, "Construction"),
    "D": (2000, 3999, "Manufacturing"),
    "E": (4000, 4999, "Transportation, Communications, Utilities"),
    "F": (5000, 5199, "Wholesale Trade"),
    "G": (5200, 5999, "Retail Trade"),
    "H": (6000, 6799, "Finance, Insurance, Real Estate"),
    "I": (7000, 8999, "Services"),
    "J": (9100, 9729, "Public Administration"),
    "K": (9900, 9999, "Nonclassifiable"),
}


def sic_division(sic: int | None) -> str | None:
    """The SIC division letter for a code, or None if the code is missing or in no division."""
    if sic is None:
        return None
    for letter, (first, last, _name) in SIC_DIVISIONS.items():
        if first <= sic <= last:
            return letter
    return None


def read_sic_codes(zip_path: Path, ciks: Iterable[int]) -> dict[int, int | None]:
    """Read each CIK's SIC code from SEC's bulk submissions zip.

    Reads only the `CIK##########.json` members asked for; the archive is never
    extracted. A missing member or an empty `sic` maps to None.
    """
    codes: dict[int, int | None] = {}
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        for cik in ciks:
            name = f"CIK{cik:010d}.json"
            sic = json.loads(zf.read(name)).get("sic") if name in members else None
            codes[cik] = int(sic) if sic else None
    return codes
