import numpy as np
import pandas as pd

from pipeline.peers import build_peers, filer_groups, near_peer_counts, peer_counts
from pipeline.sentences import norm_hash, normalize
from tests.conftest import write_tables

# Words, not numbers: normalization turns every digit into 0, which would make these one sentence.
SHARED = [
    f"Boilerplate sentence {w} that a parent and its subsidiary both file."
    for w in ("alpha", "bravo", "charlie", "delta")
]
COMMON = "Our stock price may be volatile and you could lose all of your investment."


def table(filings: dict[int, list[str]]) -> pd.DataFrame:
    rows = [(cik, normalize(t)) for cik, texts in filings.items() for t in texts]
    df = pd.DataFrame(rows, columns=["cik", "norm"])
    df["norm_hash"] = df["norm"].map(norm_hash)
    return df


def test_companies_whose_sections_mostly_coincide_form_one_group():
    sentences = table(
        {
            10: [*SHARED, COMMON],  # parent
            20: [
                *SHARED,
                "The subsidiary has one sentence of its own here.",
            ],  # shares 4 of 6 distinct
            30: [
                COMMON,
                "An unrelated company with its own words.",
                "And a second sentence of its own.",
            ],
        }
    )
    groups = filer_groups(sentences)
    assert groups[10] == groups[20] != groups[30]
    assert groups[10] == 10  # a group is named for its lowest CIK


def test_sharing_a_little_does_not_make_a_group():
    sentences = table(
        {
            1: [COMMON, "a b c d e f g", "h i j k l m n"],
            2: [COMMON, "o p q r s t u", "v w x y z a b"],
        }
    )
    groups = filer_groups(sentences)
    assert groups[1] != groups[2]


def test_a_sentence_is_counted_once_per_group_not_once_per_company():
    sentences = table(
        {
            10: [*SHARED, COMMON],
            20: [*SHARED, COMMON],
            30: [
                COMMON,
                "An unrelated company with its own words.",
                "And a second sentence of its own.",
            ],
        }
    )
    counts = peer_counts(sentences, filer_groups(sentences)).set_index("norm_hash")
    common = counts.loc[norm_hash(normalize(COMMON))]
    assert (common["n_companies"], common["n_groups"]) == (3, 2)
    shared = counts.loc[norm_hash(normalize(SHARED[0]))]
    assert (shared["n_companies"], shared["n_groups"]) == (2, 1)


def test_near_verbatim_copies_in_other_groups_are_counted():
    base = "we may not be able to protect our intellectual property rights in every country where we sell"
    filings = {
        1: [base, "company one has a sentence that nobody else has written down"],
        # 13 of 14 and 12 of 14 shingles shared: well over the threshold.
        2: [base + " products", "company two wrote this other sentence just for itself"],
        3: [
            base.replace("we sell", "we operate"),
            "company three also has a private sentence of its own",
        ],
        4: ["a completely different sentence about weather and crops and rainfall this year", base],
        5: ["nothing in common with the rest at all except being a sentence of words"],
    }
    sentences = table(filings)
    groups = filer_groups(sentences)
    queries = sentences.index[(sentences["cik"] == 1) & (sentences["norm"] == base)]
    out = near_peer_counts(sentences, groups, queries.to_numpy(), thresholds=(0.5, 0.9))
    assert out["exact_others"].tolist() == [1]  # company 4 has it word for word
    assert out["near_others_050"].tolist() == [3]  # companies 2 and 3 have it lightly edited
    assert out["near_others_090"].tolist() == [2]  # only company 2's copy is that close (13 of 14)


def test_the_stage_writes_counts_and_groups_for_every_year(cfg):
    write_tables(
        cfg,
        {
            (10, 2018): [*SHARED, COMMON],
            (20, 2018): [*SHARED, COMMON],
            (30, 2018): [COMMON, "Its own sentence."],
            (10, 2019): [COMMON, "Something new this year."],
            (30, 2019): [COMMON],
        },
    )
    out = build_peers(cfg)
    peers = pd.read_parquet(out / "year=2018.parquet").set_index("norm_hash")
    assert peers.loc[norm_hash(normalize(COMMON)), "n_groups"] == 2
    groups = pd.read_parquet(cfg.interim_dir / "filer_groups.parquet")
    assert len(groups) == 5 and set(groups.columns) == {"cik", "filing_year", "group"}
    in_2018 = groups[groups["filing_year"] == 2018].set_index("cik")["group"]
    assert in_2018[10] == in_2018[20] and np.unique(in_2018).size == 2
