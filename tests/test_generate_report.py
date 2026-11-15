import json
import re

import pytest

from src.generate_report import (
    DATA,
    LINE_CHARS,
    PAGE_LINES,
    TEMPLATE,
    build,
    explain_parts,
    lay_out,
    reading_order,
    word_diff,
)


@pytest.fixture(scope="module")
def report():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def template():
    return TEMPLATE.read_text()


@pytest.fixture(scope="module")
def page(report, template):
    return build(report, template)


def test_a_word_diff_strikes_what_was_cut_and_underlines_what_was_added():
    before = "Commodities, including corn, wheat and sugar, are our raw materials."
    after = "Commodities, including corn, wheat, rice and sugar, are our raw materials."
    assert word_diff(before, after) == (
        "Commodities, including corn, <del>wheat</del> <ins>wheat, rice</ins> "
        "and sugar, are our raw materials."
    )


def test_a_word_diff_escapes_the_filing_s_own_text():
    marked = word_diff("We own <b>patents</b>.", "We own <b>patents</b> & marks.")
    assert "<b>" not in marked and "&lt;b&gt;patents&lt;/b&gt;" in marked and "&amp;" in marked


def test_removed_sentences_are_read_where_they_were_cut():
    example = {
        "sentences": [{"label": "exact", "text": "A"}, {"label": "new", "text": "B"}],
        "removed": [
            {"after": 0, "text": "cut after A"},
            {"after": -1, "text": "cut from the top"},
            {"after": 0, "text": "also after A"},
        ],
    }
    order = reading_order(example)
    assert [item["text"] for item in order] == [
        "cut from the top",
        "A",
        "cut after A",
        "also after A",
        "B",
    ]
    assert [item["label"] for item in order] == ["removed", "exact", "removed", "removed", "new"]
    assert [item["id"] for item in order] == [0, 1, 2, 3, 4]


def test_a_sentence_takes_as_much_line_as_it_has_characters_and_wraps():
    items = [
        {"id": 0, "label": "exact", "chars": 60, "paragraph": 0},
        {"id": 1, "label": "new", "chars": 150, "paragraph": 0},
    ]
    (page,) = lay_out(items)
    assert page[0] == (0, 0, 0.0, 60.0)
    # The second starts after a gap, fills the first line, a whole second line, and ends on a third.
    assert [(sid, line) for sid, line, _, _ in page[1:]] == [(1, 0), (1, 1), (1, 2)]
    assert sum(end - start for sid, _, start, end in page if sid == 1) == pytest.approx(150)
    assert all(0 <= start < end <= LINE_CHARS for _, _, start, end in page)


def test_a_new_paragraph_starts_a_new_line_and_a_full_page_starts_another():
    items = [
        {"id": 0, "label": "exact", "chars": 30, "paragraph": 0},
        {"id": 1, "label": "exact", "chars": 30, "paragraph": 1},
    ]
    (page,) = lay_out(items)
    assert [(sid, line, start) for sid, line, start, _ in page] == [(0, 0, 0.0), (1, 1, 0.0)]

    long = [{"id": 0, "label": "exact", "chars": LINE_CHARS * (PAGE_LINES + 2), "paragraph": 0}]
    first, second = lay_out(long)
    assert len(first) == PAGE_LINES and [line for _, line, _, _ in second] == [0, 1]


def test_the_page_is_built_with_nothing_left_to_fill(page):
    assert "%%" not in page and "<!--SECTION" not in page and "/*SECTION" not in page


def test_a_value_the_template_asks_for_and_the_generator_lacks_is_named(report, template):
    with pytest.raises(KeyError, match="no_such_value"):
        build(report, template + "%%no_such_value%%")


def test_every_sentence_of_the_example_is_drawn(report, page):
    drawn = {int(i) for i in re.findall(r'class="s \w+" data-s="(\d+)"', page)}
    example = report["example"]
    assert drawn == set(range(len(example["sentences"]) + len(example["removed"])))


def test_every_chart_has_a_table_of_its_numbers(page):
    figures = re.findall(r"<figure.*?</figure>", page, flags=re.S)
    charts = [figure for figure in figures if 'class="chart"' in figure]
    assert len(charts) == page.count('class="chart"') >= 12
    for figure in charts:
        assert '<table class="numbers">' in figure, re.search(r'id="([\w-]+)"', figure).group(1)


def test_the_page_has_a_section_for_each_layer_of_the_study(page):
    for heading in ("made-h", "trust-h", "peers-h", "lasting-h", "covid-h", "wrong-h", "limits-h"):
        assert f'id="{heading}"' in page


# Digits the template may hold: names and public dates, never a result. Everything else on the
# page that is a number comes from docs/data/report.json through the generator.
TYPED = [
    "10-K",
    "Item 1A",
    "95% interval",
    "11 March",
    "November 2026",
    "51% alike",
    "21% alike",
    "71.2% F1",
    "v1-original",
    "jlattanzi4",
    "h1",
    "h2",
    "h3",
]


def test_the_template_holds_no_numbers_of_its_own(template):
    body = template.split("<body>", 1)[1]
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"%%\w+%%", "", body)
    body = re.sub(r"<[^>]+>", " ", body)  # tags and their attributes
    for allowed in TYPED:
        body = body.replace(allowed, "")
    assert re.findall(r"\S*\d\S*", body) == []


def test_every_number_in_the_survival_section_is_from_the_results(report):
    values, data = explain_parts(report["explain"])
    explain = report["explain"]
    first = explain["survival"]["overall"]["curve"][0]
    assert values["gone_first_year"] == f"{100 * first['hazard']:.0f}%"
    assert values["chains"] == f"{explain['counts']['chains']:,}"
    assert values["median_shared"] == str(explain["survival"]["by_peer_shared"]["shared"]["median"])
    assert [point["age"] for point in data["survival"]["own"]] == [
        row["age"] for row in explain["survival"]["by_peer_shared"]["own"]["curve"]
    ]
    shares = {row["name"]: row["value"] for row in data["variance"]}
    assert sum(shares.values()) == pytest.approx(
        sum(part["value"] for part in explain["variance"]["share"]["share"].values())
    )
    assert len(data["hazard"]) == len({row["name"] for row in data["hazard"]}) >= 5
