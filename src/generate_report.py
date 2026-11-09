"""
Build the project report, docs/report.html, from the processed filings.

Every number on the page is computed here from the data (or read from a saved
model result) and filled into src/report_template.html, so the page can't
drift from the data. Run from the repo root:

    python src/generate_report.py
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).parent.parent
FILINGS = PROJECT_ROOT / "data" / "processed" / "risk_factors_2006_2020.parquet"
MODEL_RESULTS = PROJECT_ROOT / "models" / "model_comparison_optimized.csv"
TEMPLATE = PROJECT_ROOT / "src" / "report_template.html"
OUTPUT = PROJECT_ROOT / "docs" / "report.html"

# all-MiniLM-L6-v2 truncates at 256 tokens: about the first 1,370 characters
# of these sections (measured in the 2026 review, see issue #1).
SBERT_CHARS = 1_400
# boilerplate_detection.py cut each company's text to 50,000 characters for TF-IDF.
TFIDF_CAP = 50_000
# A dense 10-K page holds roughly 3,000 characters; used only to draw Figure 1.
CHARS_PER_PAGE = 3_000


def yearly_lengths(filings: pd.DataFrame) -> pd.DataFrame:
    by_year = filings.groupby("filing_year")["item_1a_length"]
    table = pd.DataFrame(
        {
            "count": by_year.size(),
            "median": by_year.median(),
            "p25": by_year.quantile(0.25),
            "p75": by_year.quantile(0.75),
        }
    )
    return (
        table.round().astype(int).reset_index().rename(columns={"filing_year": "year"})
    )


def best_f1() -> str:
    results = pd.read_csv(MODEL_RESULTS)
    return f"{results.loc[results['is_best'], 'f1_macro'].iloc[0]:.1%}"


def build() -> str:
    filings = pq.read_table(
        FILINGS, columns=["cik", "filing_year", "item_1a_length"]
    ).to_pandas()
    years = yearly_lengths(filings)
    first, last = years.iloc[0], years.iloc[-1]
    median_all = int(filings["item_1a_length"].median())

    def page_row(r):
        return {
            "year": int(r["year"]),
            "chars": int(r["median"]),
            "pages": r["median"] / CHARS_PER_PAGE,
        }

    data = {
        "years": years.to_dict(orient="records"),
        "pageRows": [page_row(first), page_row(last)],
        "charsPerPage": CHARS_PER_PAGE,
        "sbertChars": SBERT_CHARS,
        "tfidfCap": TFIDF_CAP,
    }
    rows = "\n".join(
        f'        <tr><td>{r.year}</td><td class="num">{r.count:,}</td><td class="num">{r.median:,}</td>'
        f'<td class="num">{r.p25:,}</td><td class="num">{r.p75:,}</td></tr>'
        for r in years.itertuples()
    )
    fields = {
        "filings": f"{len(filings):,}",
        "companies": f"{filings['cik'].nunique():,}",
        "year_min": str(first["year"]),
        "year_max": str(last["year"]),
        "median_y0": f"{first['median']:,}",
        "median_y1": f"{last['median']:,}",
        "pages_y0": f"{first['median'] / CHARS_PER_PAGE:.0f}",
        "pages_y1": f"{last['median'] / CHARS_PER_PAGE:.0f}",
        "growth": f"{last['median'] / first['median']:.1f}",
        "median_all": f"{median_all:,}",
        "share_over_cap": f"{(filings['item_1a_length'] > TFIDF_CAP).mean():.0%}",
        "share_read": f"{SBERT_CHARS / median_all:.1%}",
        "f1": best_f1(),
        "generated": datetime.now(UTC).strftime("%B %-d, %Y"),
        "table_rows": rows,
        "data_json": json.dumps(data, separators=(",", ":")),
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fields.items():
        html = html.replace(f"%%{key}%%", value)
    leftover = re.findall(r"%%\w+%%", html)
    if leftover:
        raise ValueError(f"Unfilled template fields: {sorted(set(leftover))}")
    return html


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
