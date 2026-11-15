"""Build the report page, docs/report.html, from docs/data/report.json.

The page shows nothing that is not in that one committed file. Every number
is formatted here from a value in it and filled into src/report_template.html,
which holds no numbers of its own; the tables under each chart are written
here too, so the page reads without JavaScript. Run from the repo root:

    uv run python -m pipeline.run report-data   # writes docs/data/report.json
    uv run python src/generate_report.py
"""

import difflib
import html
import json
import math
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data" / "report.json"
TEMPLATE = ROOT / "src" / "report_template.html"
OUTPUT = ROOT / "docs" / "report.html"

LABELS = {
    "exact": "Kept word for word",
    "near": "Edited",
    "paraphrase": "Reworded",
    "new": "New",
    "removed": "Removed",
}
MINUS = "−"

# The redline figure: a page holds about 3,000 characters, as 30 lines of 100.
LINE_CHARS, PAGE_LINES = 100, 30
PAGE_W, PAGE_H, PAD_X, PAD_Y = 120, 160, 10, 11
SENTENCE_GAP = 1.5  # characters of white between two sentences on a line


def pct(share: float, digits: int = 1) -> str:
    return f"{100 * share:.{digits}f}%"


def count(value: float) -> str:
    return f"{value:,.0f}"


def points(share: float, digits: int = 1) -> str:
    """A change in percentage points, signed, with a real minus sign."""
    text = f"{100 * share:+.{digits}f}"
    return text.replace("-", MINUS)


def span(low: float, high: float, fmt=pct) -> str:
    return f"{fmt(low)} to {fmt(high)}"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def display_name(name: str) -> str:
    """SEC writes company names in capitals; the page doesn't shout."""
    return name.title() if name.isupper() else name


def word_diff(before: str, after: str) -> str:
    """`after` marked up against `before`: cut words struck, added words underlined."""
    old, new = before.split(), after.split()
    parts = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            parts.append(f"<del>{esc(' '.join(old[a0:a1]))}</del>")
        if op in ("insert", "replace"):
            parts.append(f"<ins>{esc(' '.join(new[b0:b1]))}</ins>")
        if op == "equal":
            parts.append(esc(" ".join(new[b0:b1])))
    return " ".join(parts)


def reading_order(example: dict) -> list[dict]:
    """This year's sentences in order, with last year's removed ones where they were cut.

    Each item is a sentence dict plus `label` (removed ones get "removed") and
    `id`, its position in this list.
    """
    cut_after: dict[int, list[dict]] = {}
    for removed in example["removed"]:
        cut_after.setdefault(removed["after"], []).append(removed | {"label": "removed"})
    items = list(cut_after.get(-1, []))
    for position, sentence in enumerate(example["sentences"]):
        items.append(sentence)
        items.extend(cut_after.get(position, []))
    return [item | {"id": i} for i, item in enumerate(items)]


def lay_out(items: list[dict]) -> list[list[tuple[int, int, float, float]]]:
    """Flow sentences onto pages as line segments: per page, (id, line, start, end).

    A sentence takes as many characters of line as it has, wrapping like
    text. A new paragraph starts a new line. `start` and `end` are in
    characters from the left margin.
    """
    pages: list[list[tuple[int, int, float, float]]] = [[]]
    line, at, paragraph = 0, 0.0, None
    for item in items:
        if "paragraph" in item:
            if paragraph is not None and item["paragraph"] != paragraph and at > 0:
                line, at = line + 1, 0.0
            paragraph = item["paragraph"]
        left = float(item["chars"])
        while left > 0:
            if at >= LINE_CHARS - 2:
                line, at = line + 1, 0.0
            if line >= PAGE_LINES:
                pages.append([])
                line = 0
            take = min(left, LINE_CHARS - at)
            pages[-1].append((item["id"], line, at, at + take))
            at, left = at + take, left - take
        at += SENTENCE_GAP
    return pages


def page_svg(number: int, segments: list[tuple[int, int, float, float]], items: list[dict]) -> str:
    """One page of the redline as an inline SVG: a visible bar and a taller hit area a segment."""
    pitch = (PAGE_H - 2 * PAD_Y) / PAGE_LINES
    bars, hits = [], []
    for sid, line, start, end in segments:
        item = items[sid]
        thick = 2.6 if item.get("heading") else 1.9
        y = PAD_Y + line * pitch + (pitch - thick) / 2
        x, width = PAD_X + start, max(end - start, 1.5)
        bars.append(
            f'<rect class="s {item["label"]}" data-s="{sid}" x="{x:.1f}" y="{y:.2f}" '
            f'width="{width:.1f}" height="{thick}" rx="0.9"/>'
        )
        hits.append(
            f'<rect class="h" data-s="{sid}" x="{x - 0.75:.1f}" y="{PAD_Y + line * pitch:.2f}" '
            f'width="{width + 1.5:.1f}" height="{pitch:.2f}"/>'
        )
    return (
        f'<svg class="sheet" viewBox="0 0 {PAGE_W} {PAGE_H}" aria-hidden="true" '
        f'data-page="{number}"><rect class="paper" x="0.5" y="0.5" width="{PAGE_W - 1}" '
        f'height="{PAGE_H - 1}" rx="2"/>{"".join(bars)}{"".join(hits)}</svg>'
    )


def sentence_html(item: dict, clip: int | None = None) -> str:
    """A sentence as the redline shows it: edits marked against last year's wording."""

    def cut(text: str) -> str:
        return text if clip is None or len(text) <= clip else text[: clip - 1].rstrip() + "…"

    if item["label"] == "near":
        return word_diff(cut(item["previous"]), cut(item["text"]))
    if item["label"] == "paraphrase":
        return (
            f"<ins>{esc(cut(item['text']))}</ins>"
            f'<span class="was">Last year: {esc(cut(item["previous"]))}</span>'
        )
    if item["label"] == "removed":
        return f"<del>{esc(cut(item['text']))}</del>"
    if item["label"] == "new":
        return f"<ins>{esc(cut(item['text']))}</ins>"
    return esc(cut(item["text"]))


def changes_list(items: list[dict]) -> str:
    """Every sentence that isn't a word-for-word copy, in reading order, as a text redline."""
    rows = [
        f'<li><span class="tag">{LABELS[item["label"]]}</span> {sentence_html(item)}</li>'
        for item in items
        if item["label"] != "exact"
    ]
    return "\n".join(rows)


def table(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    """An HTML table; columns from `numeric_from` on are right-aligned numbers."""

    def cells(tag: str, values: list[str]) -> str:
        return "".join(
            f'<{tag}{" class=\"num\"" if i >= numeric_from else ""}>{value}</{tag}>'
            for i, value in enumerate(values)
        )

    body = "\n".join(f"<tr>{cells('td', row)}</tr>" for row in rows)
    return (
        f'<table class="numbers"><thead><tr>{cells("th", head)}</tr></thead>'
        f"<tbody>\n{body}\n</tbody></table>"
    )


def example_parts(example: dict) -> tuple[dict, dict]:
    """Values, tables and chart data for the redline figure."""
    items = reading_order(example)
    pages = lay_out(items)
    counts = {label: sum(item["label"] == label for item in items) for label in LABELS}
    filed = date.fromisoformat(example["date_filed"])
    name = display_name(example["name"])
    values = {
        "ex_name": esc(name),
        "ex_year": str(example["year"]),
        "ex_prev_year": str(example["year"] - 1),
        "ex_filed": f"{filed.day} {filed:%B %Y}",
        "ex_url": esc(example["url"]),
        "ex_n": count(len(example["sentences"])),
        "ex_exact_n": count(counts["exact"]),
        "ex_near_n": count(counts["near"]),
        "ex_para_n": count(counts["paraphrase"]),
        "ex_new_n": count(counts["new"]),
        "ex_removed_n": count(counts["removed"]),
        "ex_changed_n": count(len(items) - counts["exact"]),
        "ex_pages_n": count(len(pages)),
        "ex_min": count(example["rule"]["sentences"][0]),
        "ex_max": count(example["rule"]["sentences"][1]),
        "ex_rule_year": str(example["rule"]["year"]),
        "ex_exact": pct(example["shares"]["exact"]),
        "ex_new": pct(example["shares"]["new"]),
        "ex_removed": pct(example["shares"]["removed"]),
        "redline_pages": "\n".join(
            page_svg(number + 1, segments, items) for number, segments in enumerate(pages)
        ),
        "redline_changes": changes_list(items),
    }
    data = {
        "name": name,
        "sentences": [
            {"label": item["label"], "html": sentence_html(item, clip=360)} for item in items
        ],
    }
    return values, data


def measure_parts(measure: dict) -> tuple[dict, dict]:
    """Values, tables and chart data for sections 1 to 3."""
    shares, peers = measure["shares"], measure["peers"]
    thresholds, baselines = measure["thresholds"], measure["baselines"]
    years = shares["by_year"]
    first, last = years[0], years[-1]
    before_last = years[:-1]
    most_new = max(before_last, key=lambda y: y["pooled"]["new"])
    least_new = min(years, key=lambda y: y["pooled"]["new"])
    spread = shares["by_filing"]

    real = thresholds["real_pairs"]
    makeup, intervals = real["makeup"], real["intervals"]
    first_cut = real["at_known_answer_tau_p"]
    known = thresholds["known_answer"]
    hard = known["labeled"]["different_hard"]
    carried = known["carried_over"]
    stranger = baselines["stranger_filings"]
    random_pairs = baselines["random_pairs"]
    shuffled = baselines["shuffled_words"]
    judge_two = real["other_views"]["judge 2 on its subset"]
    judge_one = real["other_views"]["judge 1 on pairs judge 2 never saw"]

    peer_years = peers["by_year"]
    k = f"k{peers['primary_k']}"
    most_shared = peers["most_shared_last_year"][:5]

    tiers = [
        ("Edited", "near", 1 - makeup["near"]["different"], intervals["near"]["carried"]),
        (
            "Reworded",
            "paraphrase",
            1 - makeup["paraphrase"]["different"],
            intervals["paraphrase"]["carried"],
        ),
        ("New", "new", makeup["new"]["different"], intervals["new"]["different"]),
    ]

    values = {
        "filings": count(shares["filings"]),
        "companies": count(shares["companies"]),
        "sentences": count(shares["sentences"]),
        "sentences_millions": f"{shares['sentences'] / 1e6:.1f}",
        "previous_sentences": count(shares["previous_sentences"]),
        "year_first": str(first["year"]),
        "year_last": str(last["year"]),
        "pooled_exact": pct(shares["pooled"]["exact"]),
        "pooled_near": pct(shares["pooled"]["near"]),
        "pooled_para": pct(shares["pooled"]["paraphrase"]),
        "pooled_new": pct(shares["pooled"]["new"]),
        "pooled_removed": pct(shares["pooled"]["removed"]),
        "median_exact": pct(spread["exact"]["p50"]),
        "median_near": pct(spread["near"]["p50"]),
        "median_new": pct(spread["new"]["p50"]),
        "median_removed": pct(spread["removed"]["p50"]),
        "new_p10": pct(spread["new"]["p10"]),
        "new_p90": pct(spread["new"]["p90"]),
        "nothing_new": count(shares["filings_nothing_new"]),
        "all_new": count(shares["filings_all_new"]),
        "most_new_year": str(most_new["year"]),
        "most_new_exact": pct(most_new["pooled"]["exact"], 0),
        "least_new_year": str(least_new["year"]),
        "least_new_exact": pct(least_new["pooled"]["exact"], 0),
        "last_new": pct(last["pooled"]["new"]),
        "before_last_new": pct(years[-2]["pooled"]["new"]),
        "tau_v_percent": pct(measure["tau_v"]),
        "tau_p": f"{measure['tau_p']:.3f}",
        "tau_p_first": f"{thresholds['tau_p_known_answer']:.3f}",
        "known_precision": pct(carried["precision_without_hard"]["value"]),
        "known_recall": pct(carried["recall"]["value"]),
        "hard_called": count(hard["near"] + hard["paraphrase"]),
        "hard_n": count(sum(hard.values())),
        "kappa_first": f"{real['rounds']['v1']['kappa']:.2f}",
        "kappa_second": f"{real['rounds']['v2']['kappa']:.2f}",
        "kappa_bar": f"{real['good_kappa']:.2f}",
        "judged_n": count(judge_one["n_pairs"] + judge_two["n_pairs"]),
        "judge_two_n": count(judge_two["n_pairs"]),
        "agreed_n": count(real["n"]),
        "first_cut_wrong": pct(first_cut["makeup"]["paraphrase"]["different"], 0),
        "near_right": pct(tiers[0][2]),
        "near_right_span": span(*tiers[0][3]),
        "para_right": pct(tiers[1][2]),
        "para_right_span": span(*tiers[1][3]),
        "para_pairs": count(makeup["paraphrase"]["n"]),
        "new_right": pct(tiers[2][2]),
        "judge_two_para": pct(1 - judge_two["paraphrase"]["different"]),
        "judge_one_para": pct(1 - judge_one["paraphrase"]["different"]),
        "near_different": pct(makeup["near"]["different"], 0),
        "shuffled_clear": pct(shuffled["share_cosine_at_or_above_tau_p"]),
        "shuffled_n": count(shuffled["n"]),
        "random_matched": count(random_pairs["n_near"] + random_pairs["n_paraphrase"]),
        "random_n": count(random_pairs["n"]),
        "stranger_filings": count(stranger["n_filings"]),
        "stranger_sentences": count(stranger["own"]["n"]),
        "stranger_reworded": pct(stranger["stranger"]["paraphrase"]),
        "stranger_copied": pct(stranger["stranger"]["exact"] + stranger["stranger"]["near"]),
        "own_copied": pct(stranger["own"]["exact"] + stranger["own"]["near"]),
        "peer_k": str(peers["primary_k"]),
        "peer_words": str(peers["full_sentence_words"]),
        "peer_full": pct(peers["overall"]["full"][k]),
        "peer_first": pct(peer_years[0]["full"][k]),
        "peer_first_year": str(peer_years[0]["year"]),
        "peer_last": pct(peer_years[-1]["full"][k]),
        "peer_before_last": pct(peer_years[-2]["full"][k]),
        "peer_before_last_year": str(peer_years[-2]["year"]),
        "peer_times": f"{peer_years[-1]['full'][k] / peer_years[0]['full'][k]:.0f}",
        "peer_grouped": count(peers["filings_in_groups_of_two_or_more"]),
        "most_shared": "\n".join(
            f'<li>“{esc(row["text"])}” <span class="tally">{count(row["other_groups"])} '
            "other companies</span></li>"
            for row in most_shared
        ),
        "table_years": table(
            ["Fiscal year", "Filings", "Kept", "Edited", "Reworded", "New", "Removed"],
            [
                [str(y["year"]), count(y["filings"])]
                + [
                    pct(y["pooled"][s])
                    for s in ("exact", "near", "paraphrase", "new", "removed")
                ]
                for y in years
            ],
        ),
        "table_spread": table(
            ["Share of a filing's sentences", "10th", "25th", "Median", "75th", "90th"],
            [
                [LABELS[s]] + [pct(spread[s][q]) for q in ("p10", "p25", "p50", "p75", "p90")]
                for s in ("exact", "near", "paraphrase", "new", "removed")
            ],
        ),
        "table_tiers": table(
            ["The matcher said", "Judged right", "95% interval", "Pairs", "Sentences they stand for"],
            [
                [name, pct(right), span(*interval), count(makeup[key]["n"]), count(makeup[key]["stands_for"])]
                for name, key, right, interval in tiers
            ],
        ),
        "table_cosine": table(
            ["Cosine with the closest of last year's sentences", "Pairs", "Judged carried over"],
            [
                [
                    f"{band['low']:.2f} to {band['high']:.2f}",
                    count(band["n"]),
                    pct(band["same"] + band["paraphrase"]),
                ]
                for band in real["by_cosine"]
            ],
        ),
        "table_stranger": table(
            ["Matched against", "Kept", "Edited", "Reworded", "New"],
            [
                [name] + [pct(stranger[key][s]) for s in ("exact", "near", "paraphrase", "new")]
                for name, key in (
                    ("The company's own previous filing", "own"),
                    ("An unrelated company's", "stranger"),
                    ("An unrelated company's, same industry", "stranger_same_division"),
                )
            ],
        ),
        "table_peers": table(
            ["Fiscal year", "Full sentences", f"Filed by {peers['primary_k']} or more other companies"],
            [[str(y["year"]), count(y["full_sentences"]), pct(y["full"][k])] for y in peer_years],
        ),
    }
    data = {
        "years": [
            {"year": y["year"], "filings": y["filings"]}
            | {s: y["pooled"][s] for s in ("exact", "near", "paraphrase", "new", "removed")}
            for y in years
        ],
        "histogram": shares["share_new_histogram"],
        "median_new": spread["new"]["p50"],
        "tiers": [
            {"name": name, "value": right, "low": interval[0], "high": interval[1], "n": makeup[key]["n"]}
            for name, key, right, interval in tiers
        ],
        "cosine": [
            {"low": band["low"], "high": band["high"], "n": band["n"], "carried": band["same"] + band["paraphrase"]}
            for band in real["by_cosine"]
        ],
        "tau_p": measure["tau_p"],
        "tau_p_first": thresholds["tau_p_known_answer"],
        "stranger": [
            {"name": name} | {s: stranger[key][s] for s in ("exact", "near", "paraphrase", "new")}
            for name, key in (
                ("Its own previous filing", "own"),
                ("An unrelated company's", "stranger"),
            )
        ],
        "peers": [{"year": y["year"], "share": y["full"][k]} for y in peer_years],
    }
    return values, data


HAZARD_TERMS = {
    "peer_shared": "Shared with other companies",
    "peer_shared, age 1": "Shared, at its first filing",
    "peer_shared, ages 2-3": "Shared, at its second and third",
    "peer_shared, ages 4+": "Shared, from its fourth on",
    "position: middle": "In the middle of the section",
    "position: bottom": "At the bottom of the section",
    "section: section_mid": "In a mid-length section",
    "section: section_high": "In a long section",
    "words: words_mid": "A mid-length sentence",
    "words: words_high": "A long sentence",
}
CHARTED_TERMS = [name for name in HAZARD_TERMS if not name.startswith("words")]
VARIANCE_PARTS = {
    "residual": "The same company, year to year",
    "company": "The company",
    "year": "The fiscal year",
    "division": "The industry",
}


def ratio(value: float) -> str:
    return f"{value:.2f}"


def explain_parts(explain: dict) -> tuple[dict, dict]:
    """Values, tables and chart data for the section on how long sentences last."""
    counts, survival = explain["counts"], explain["survival"]
    overall = survival["overall"]["curve"]
    shared, own = survival["by_peer_shared"]["shared"], survival["by_peer_shared"]["own"]
    fit = explain["hazard"]["by_age" if explain["effect_varies_by_age"] else "constant"]
    ratios = fit["hazard_ratios"]
    peer_terms = [name for name in ratios if name.startswith("peer_shared")]
    divisions = [ratios[name]["value"] for name in ratios if name.startswith("division")]
    born = [name for name in ratios if name.startswith("born")]
    primary, logit = explain["variance"]["share"]["share"], explain["variance"]["logit"]["share"]

    def named(term: str) -> str:
        if term.startswith("born: "):
            return f"Added for fiscal {term[6:].replace('-', '–')}"
        return HAZARD_TERMS[term]

    rows = [name for name in CHARTED_TERMS if name in ratios]
    by_age = {row["age"]: row for row in overall}
    own_by_age = {row["age"]: row for row in own["curve"]}
    shared_by_age = {row["age"]: row for row in shared["curve"]}
    values = {
        "chains": count(counts["chains"]),
        "chains_removed": count(counts["removed"]),
        "first_filing_millions": f"{counts['sentences_in_first_filings'] / 1e6:.1f}",
        "gone_first_year": pct(overall[0]["hazard"], 0),
        "survive_first": pct(overall[0]["survival"]),
        "median_all": str(survival["overall"]["median"]),
        "last_age": str(overall[-1]["age"]),
        "survive_last": pct(overall[-1]["survival"], 0),
        "median_shared": str(shared["median"]),
        "median_own": str(own["median"]),
        "shared_first": pct(shared["curve"][0]["survival"]),
        "own_first": pct(own["curve"][0]["survival"]),
        "shared_chains": count(shared["chains"]),
        "less_first": pct(1 - ratios[peer_terms[0]]["value"], 0),
        "less_later": pct(1 - ratios[peer_terms[1]]["value"], 0) if len(peer_terms) > 1 else "",
        "hr_shared_old": ratio(ratios[peer_terms[-1]]["value"]),
        "hr_shared_old_span": span(
            ratios[peer_terms[-1]]["low"], ratios[peer_terms[-1]]["high"], ratio
        ),
        "long_section_more": pct(ratios["section: section_high"]["value"] - 1, 0),
        "division_low": ratio(min(divisions)),
        "division_high": ratio(max(divisions)),
        "company_times": f"{math.exp(fit['company_sd']['value']):.1f}",
        "hazard_companies": count(fit["companies"]),
        "hazard_sentences": count(fit["chains_watched_at_age_1"]),
        "var_within": pct(primary["residual"]["value"], 0),
        "var_company": pct(primary["company"]["value"], 0),
        "var_year": pct(primary["year"]["value"], 0),
        "var_industry": pct(primary["division"]["value"], 0),
        "var_company_logit": pct(logit["company"]["value"], 0),
        "var_within_logit": pct(logit["residual"]["value"], 0),
        "table_survival": table(
            ["Filings since added", "Still watched", "Cut at the next filing", "All still there", "Shared", "The company's own"],
            [
                [
                    str(age),
                    count(row["at_risk"]),
                    pct(row["hazard"]),
                    pct(row["survival"]),
                    pct(shared_by_age[age]["survival"]) if age in shared_by_age else "",
                    pct(own_by_age[age]["survival"]) if age in own_by_age else "",
                ]
                for age, row in by_age.items()
            ],
        ),
        "table_hazard": table(
            ["A sentence that is...", "Hazard ratio", "95% interval"],
            [
                [named(name), ratio(ratios[name]["value"]), span(ratios[name]["low"], ratios[name]["high"], ratio)]
                for name in [*rows, *(n for n in HAZARD_TERMS if n.startswith("words") and n in ratios), *born]
            ],
        ),
        "table_variance": table(
            ["Share of the variation that belongs to", "Estimate", "95% interval", "On a logit scale"],
            [
                [label, pct(primary[key]["value"]), span(primary[key]["low"], primary[key]["high"]), pct(logit[key]["value"])]
                for key, label in VARIANCE_PARTS.items()
            ],
        ),
    }
    data = {
        "survival": {
            name: [
                {"age": row["age"], "survival": row["survival"], "low": row["low"], "high": row["high"]}
                for row in group["curve"]
            ]
            for name, group in (("shared", shared), ("own", own))
        },
        "hazard": [
            {"name": named(name), "value": ratios[name]["value"], "low": ratios[name]["low"], "high": ratios[name]["high"]}
            for name in rows
        ],
        "variance": [
            {"name": label, "value": primary[key]["value"], "low": primary[key]["low"], "high": primary[key]["high"]}
            for key, label in VARIANCE_PARTS.items()
        ],
    }
    return values, data


def covid_parts(test: dict) -> tuple[dict, dict]:
    """Values, tables and chart data for the 2020 section."""
    did, after = test["did"], test["year_after"]
    exposure = test["exposure"]["2019"]
    sample = test["sample"]
    by_year = after["by_year"]
    year_keys = sorted(by_year)
    rules = after["fiscal_2020_by_rules"]
    outcomes = [
        ("New sentences", "share_new"),
        ("Removed", "share_removed"),
        ("Edited in place", "share_near"),
        ("New, not mentioning COVID", "share_new_ex_covid"),
    ]
    variants = test["robustness"]["share_new"]
    no_fixed = variants["no company fixed effects"]
    usual = variants["exposure by last year's filing date"]
    placebo_new = test["placebos"]["share_new"]
    placebo_all = [fit for fits in test["placebos"].values() for fit in fits]
    weeks = test["event_study"]["share_new"]

    def estimate_row(name: str, fit: dict) -> list[str]:
        return [name, points(fit["estimate"]), span(fit["low"], fit["high"], points), count(fit["filings"])]

    values = {
        "event_year": str(did["share_new"]["event_year"]),
        "did_filings": count(sample["single_matching_filing"]),
        "did_companies": count(sample["companies"]),
        "late_filings": count(exposure["late"]["filings"]),
        "early_filings": count(exposure["early"]["filings"]),
        "late_covid": pct(exposure["late"]["share_new_covid"]),
        "early_covid": pct(exposure["early"]["share_new_covid"]),
        "late_mention": pct(exposure["late"]["mention"], 0),
        "early_mention": pct(exposure["early"]["mention"], 0),
        "did_new": points(did["share_new"]["estimate"]),
        "did_new_span": span(did["share_new"]["low"], did["share_new"]["high"], points),
        "did_removed": points(did["share_removed"]["estimate"]),
        "did_removed_span": span(did["share_removed"]["low"], did["share_removed"]["high"], points),
        "no_fixed_new": points(no_fixed["estimate"]),
        "no_fixed_span": span(no_fixed["low"], no_fixed["high"], points),
        "usual_new": points(usual["estimate"]),
        "placebo_years": " and ".join(str(fit["event_year"]) for fit in placebo_new),
        "placebo_new": " and ".join(points(fit["estimate"]) for fit in placebo_new),
        "placebo_n": count(len(placebo_all)),
        "placebo_excluding_zero": count(
            sum(fit["low"] > 0 or fit["high"] < 0 for fit in placebo_all)
        ),
        "after_year": year_keys[-1],
        "after_prev_year": year_keys[-2],
        "after_new": pct(by_year[year_keys[-1]]["share_new"]),
        "after_prev_new": pct(by_year[year_keys[-2]]["share_new"]),
        "after_removed": pct(by_year[year_keys[-1]]["share_removed"]),
        "after_prev_removed": pct(by_year[year_keys[-2]]["share_removed"]),
        "after_new_covid": pct(by_year[year_keys[-1]]["share_new_covid"]),
        "after_rules_share": pct(after["share_filed_after_rules"]),
        "rules_before_n": count(rules["before"]["filings"]),
        "rules_before_removed": pct(rules["before"]["share_removed"]),
        "rules_after_n": count(rules["after"]["filings"]),
        "rules_after_removed": pct(rules["after"]["share_removed"]),
        "summary_heading": pct(after["headings"][year_keys[-1]]["summary"]),
        "summary_heading_before": pct(after["headings"][year_keys[-2]]["summary"]),
        "general_heading": pct(after["headings"][year_keys[-1]]["general"]),
        "table_did": table(
            ["Outcome", "Estimate (points)", "95% interval", "Filings"],
            [estimate_row(name, did[key]) for name, key in outcomes],
        ),
        "table_weeks": table(
            ["Weeks from 11 March", "Fiscal 2019 filings", "New sentences (points)", "95% interval"],
            [
                [
                    week["bin"].replace("-", MINUS),
                    count(week["n_event"]),
                    points(week["estimate"]) if week["low"] is not None else "reference",
                    span(week["low"], week["high"], points) if week["low"] is not None else "",
                ]
                for week in weeks
            ],
        ),
        "table_after": table(
            ["Fiscal year", "Filings", "New", "of which mention COVID", "Removed"],
            [
                [
                    year,
                    count(by_year[year]["filings"]),
                    pct(by_year[year]["share_new"]),
                    pct(by_year[year]["share_new_covid"]),
                    pct(by_year[year]["share_removed"]),
                ]
                for year in year_keys
            ]
            + [
                [
                    f"{year_keys[-1]}, filed {when} the new rules",
                    count(rules[when]["filings"]),
                    pct(rules[when]["share_new"]),
                    pct(rules[when]["share_new_covid"]),
                    pct(rules[when]["share_removed"]),
                ]
                for when in ("before", "after")
            ],
        ),
    }
    data = {
        "did": [
            {"name": name, "value": did[key]["estimate"], "low": did[key]["low"], "high": did[key]["high"]}
            for name, key in outcomes
        ],
        "weeks": [
            {"name": week["bin"].replace("-", MINUS), "value": week["estimate"], "low": week["low"], "high": week["high"], "n": week["n_event"]}
            for week in weeks
        ],
        "after": [
            {"name": year, "new": by_year[year]["share_new"], "removed": by_year[year]["share_removed"]}
            for year in year_keys
        ]
        + [
            {"name": f"{year_keys[-1]}, {when} the rules", "new": rules[when]["share_new"], "removed": rules[when]["share_removed"]}
            for when in ("before", "after")
        ],
    }
    return values, data


def build(report: dict, template: str) -> str:
    """Fill the template from the report data. Raises if a placeholder is left unfilled."""
    values: dict[str, str] = {}
    charts: dict[str, object] = {}
    for name, parts in (
        ("example", example_parts),
        ("measure", measure_parts),
        ("explain", explain_parts),
        ("test2020", covid_parts),
    ):
        section_values, section_data = parts(report[name])
        values |= section_values
        charts[name] = section_data
    values["data_json"] = json.dumps(charts, separators=(",", ":"), ensure_ascii=False).replace(
        "</", "<\\/"
    )
    page = re.sub(r"%%(\w+)%%", lambda m: values.get(m.group(1), m.group(0)), template)
    missing = sorted(set(re.findall(r"%%(\w+)%%", page)))
    if missing:
        raise KeyError(f"the template asks for values the generator doesn't have: {missing}")
    return page


def main() -> None:
    page = build(json.loads(DATA.read_text()), TEMPLATE.read_text())
    OUTPUT.write_text(page)
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(page) / 1e3:,.0f} KB)")


if __name__ == "__main__":
    main()
