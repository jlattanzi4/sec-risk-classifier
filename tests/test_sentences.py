import pytest

from pipeline.sentences import (
    LONG_SENTENCE_CHARS,
    filing_sentences,
    is_debris,
    norm_hash,
    normalize,
    split_sentences,
)

# Also used by scripts/compare_splitters.py to score each candidate splitter.
SPLIT_CASES = [
    (
        "We are subject to U.S. federal regulation. Changes could harm us.",
        ["We are subject to U.S. federal regulation.", "Changes could harm us."],
    ),
    (
        "We borrowed $30.0 million under the facility. Interest rates may rise.",
        ["We borrowed $30.0 million under the facility.", "Interest rates may rise."],
    ),
    (
        "Competitors include Acme Corp. and Beta Inc. These firms are larger.",
        ["Competitors include Acme Corp. and Beta Inc.", "These firms are larger."],
    ),
    (
        "See Item 7, e.g., liquidity. We may need capital.",
        ["See Item 7, e.g., liquidity.", "We may need capital."],
    ),
    (
        # A short line with no end punctuation, followed by a capitalized line, is a heading.
        "Risks Related to Our Business\nWe depend on a few customers.",
        ["Risks Related to Our Business", "We depend on a few customers."],
    ),
    (
        "(1) our ability to borrow; (2) our ability to refinance.",
        ["(1) our ability to borrow; (2) our ability to refinance."],
    ),
    ("", []),
    ("Not applicable.", ["Not applicable."]),
    (
        # A line break in the middle of a sentence is layout, not a boundary.
        (
            "Forward-looking statements speak only as of the date of\nthis Annual Report. "
            "We disclaim any obligation to update them."
        ),
        [
            "Forward-looking statements speak only as of the date of this Annual Report.",
            "We disclaim any obligation to update them.",
        ],
    ),
    (
        "Our auditor, Smith & Co. LLP, is located in the U.S. We rely on its reports.",
        ["Our auditor, Smith & Co. LLP, is located in the U.S.", "We rely on its reports."],
    ),
    (
        'Customers may conclude our products are "unsafe." That would reduce sales.',
        ['Customers may conclude our products are "unsafe."', "That would reduce sales."],
    ),
    (
        "Approval by the F.D.A. is required. No. 5 fuel oil prices rose 3.5% vs. 2018 levels.",
        ["Approval by the F.D.A. is required.", "No. 5 fuel oil prices rose 3.5% vs. 2018 levels."],
    ),
    (
        # Words that look like abbreviations but end sentences all the time in a 10-K.
        "Our reports are filed with the SEC. Competitors may rely on prior art. We disagree.",
        [
            "Our reports are filed with the SEC.",
            "Competitors may rely on prior art.",
            "We disagree.",
        ],
    ),
    (
        # A cross-reference to another item of the 10-K runs straight into that item's title.
        (
            "See “Item 7. Management’s Discussion and Analysis” and Part II, Item 9A. "
            "Controls and Procedures. We face other risks."
        ),
        [
            (
                "See “Item 7. Management’s Discussion and Analysis” and Part II, Item 9A. "
                "Controls and Procedures."
            ),
            "We face other risks.",
        ],
    ),
    (
        "The Shares trade from 9:30 a.m. to 4:00 p.m. Eastern Time on Textron Inc.’s venue.",
        ["The Shares trade from 9:30 a.m. to 4:00 p.m. Eastern Time on Textron Inc.’s venue."],
    ),
]


@pytest.mark.parametrize(("text", "expected"), SPLIT_CASES)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_normalize_follows_the_global_rule():
    assert normalize("The Company’s 2019 revenue fell 12.5%.") == (
        "the company s 0000 revenue fell 00 0"
    )
    assert normalize("  Café Ⅻ — ‘quoted’  ") == "caf xii quoted"


def test_is_debris():
    assert is_debris("2019 2018 2017 $ 1,234 $ 5,678")
    assert is_debris("(a)")
    assert is_debris("")
    assert not is_debris("We may be unable to attract and retain key personnel.")


def test_norm_hash_is_a_stable_signed_int64():
    value = norm_hash("we may be unable to attract and retain key personnel")
    assert isinstance(value, int)
    assert -(2**63) <= value < 2**63
    assert value == norm_hash("we may be unable to attract and retain key personnel")
    assert value != norm_hash("we may be unable to attract or retain key personnel")


def test_filing_sentences_builds_the_table():
    long_sentence = "We " + "depend on many things and " * 70 + "more."
    assert len(long_sentence) > LONG_SENTENCE_CHARS
    text = (
        "Risks Related to Our Business\n"
        "Total revenue fell 12.5% in 2019.\n"
        "2019 2018 2017 $ 1,234 $ 5,678\n"
        "Total revenue fell 14.0% in 2020!\n" + long_sentence
    )
    df = filing_sentences(320193, 2020, text)

    assert list(df.columns) == [
        "cik",
        "filing_year",
        "sent_idx",
        "para_idx",
        "text",
        "norm",
        "norm_hash",
        "n_chars",
        "is_long",
        "is_heading",
    ]
    assert (df["cik"] == 320193).all() and (df["filing_year"] == 2020).all()
    # The table row is debris and is dropped; numbering stays contiguous, while
    # para_idx keeps each sentence's place among the section's paragraphs.
    assert not df["text"].str.contains("1,234").any()
    assert df["sent_idx"].tolist() == [0, 1, 2, 3]
    assert df["para_idx"].tolist() == [0, 1, 3, 4]
    assert df["is_heading"].tolist() == [True, False, False, False]
    fell = df[df["text"].str.startswith("Total revenue fell")]
    assert len(fell) == 2
    # Sentences that differ only in digits and punctuation share a hash.
    assert fell["norm_hash"].nunique() == 1
    assert fell["norm"].iloc[0] == "total revenue fell 00 0 in 0000"
    assert df["is_long"].tolist() == [False, False, False, True]
    assert df["n_chars"].iloc[-1] == len(long_sentence)
    assert str(df["norm_hash"].dtype) == "int64"


def test_empty_and_near_empty_sections():
    # Review focus: a near-empty section yields zero or one sentences, never an error.
    empty = filing_sentences(1, 2010, "")
    assert len(empty) == 0 and str(empty["norm_hash"].dtype) == "int64"
    assert len(filing_sentences(1, 2010, "Not applicable.")) == 0  # 13 letters: debris
    one = filing_sentences(1, 2010, "Not applicable to smaller reporting companies.")
    assert one["text"].tolist() == ["Not applicable to smaller reporting companies."]


def test_bullets_and_wrapped_lines():
    text = (
        "Any of these events could harm us, including:\n"
        "•\n"
        "regulatory authorities may withdraw approvals of the product; and\n"
        "•\n"
        "our reputation with physicians and patients may suffer.\n"
        "The agency published a final rule on civil monetary penalties for statutory "
        "violations by manufacturers, distributors and the pharmacies they supply, a rule "
        "that drew many comments from the industry and that was finalized in early\n"
        "2017, went into effect on January 1, 2019. Penalties apply as described above\n"
        "For example, inspectors visited our principal manufacturing plant.\n"
        "4. impose substantial liabilities for pollution from our operations.\n"
    )
    assert split_sentences(text) == [
        "Any of these events could harm us, including:",
        "regulatory authorities may withdraw approvals of the product; and",
        "our reputation with physicians and patients may suffer.",
        (
            "The agency published a final rule on civil monetary penalties for statutory "
            "violations by manufacturers, distributors and the pharmacies they supply, a rule "
            "that drew many comments from the industry and that was finalized in early 2017, "
            "went into effect on January 1, 2019."
        ),
        "Penalties apply as described above",
        "For example, inspectors visited our principal manufacturing plant.",
        "4. impose substantial liabilities for pollution from our operations.",
    ]
