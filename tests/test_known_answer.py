import pandas as pd

from pipeline.known_answer import assemble_pairs, sample_bases, survives
from tests.conftest import write_tables

LONG = "We depend on a small number of large customers for most of our revenue each year."
SHORT = "Our stock may fall."
HEADING = "Risks Related to Our Business and the Industry in Which We Operate Today"


def test_bases_are_full_sentences_drawn_evenly_by_year(cfg):
    filings = {
        (cik, year): [f"{LONG[:-1]} in market {cik} and region {i}." for i in range(30)] + [SHORT, HEADING]
        for cik in (1, 2, 3) for year in (2017, 2018)
    }
    write_tables(cfg, filings)
    bases = sample_bases(cfg, n_per_year=10, seed=0)
    assert bases.groupby("filing_year").size().to_dict() == {2017: 10, 2018: 10}
    assert not bases["text"].isin([SHORT, HEADING]).any()
    assert bases["id"].is_unique and list(bases.columns) == ["id", "cik", "filing_year", "sent_idx", "text"]
    pd.testing.assert_frame_equal(bases, sample_bases(cfg, n_per_year=10, seed=0))
    assert not bases.equals(sample_bases(cfg, n_per_year=10, seed=1))


def test_a_light_edit_keeps_most_words_in_order():
    assert survives(LONG, LONG.replace("small", "limited"))
    assert survives(LONG, LONG.replace("each year", "in 2019"))
    assert not survives(LONG, "Most of what we earn every year comes from just a few big buyers.")


def bases_frame(n=4):
    return pd.DataFrame({
        "id": [f"ka{i:04d}" for i in range(n)], "cik": range(n), "filing_year": [2010 + i for i in range(n)],
        "sent_idx": range(n), "text": [f"{LONG[:-1]} in region {i}." for i in range(n)],
    })


def test_pairs_are_assembled_by_kind_with_their_confirmation():
    bases = bases_frame()
    generated = [
        {"id": b.id, "verbatim": b.text.replace("small", "limited"), "paraphrase": f"reworded {i}",
         "different_hard": f"changed {i}"}
        for i, b in enumerate(bases.itertuples())
    ]
    generated[3]["verbatim"] = "Nothing like the original sentence at all, completely new words here."
    verdicts = {("ka0000", "paraphrase"): "same", ("ka0000", "different_hard"): "different",
                ("ka0001", "paraphrase"): "different", ("ka0001", "different_hard"): "same",
                ("ka0002", "paraphrase"): "same", ("ka0002", "different_hard"): "different",
                ("ka0003", "paraphrase"): "same", ("ka0003", "different_hard"): "different"}
    partners = bases.assign(text=[f"A wholly unrelated sentence number {i} about weather." for i in range(4)])
    pairs = assemble_pairs(bases, generated, verdicts, partners, seed=0)

    assert len(pairs) == 16 and pairs["id"].is_unique
    assert set(pairs["kind"]) == {"verbatim", "paraphrase", "different_hard", "different_easy"}
    by = pairs.set_index(["base", "kind"])
    assert by.loc[("ka0000", "paraphrase"), "confirmed"]
    assert not by.loc[("ka0001", "paraphrase"), "confirmed"]          # the second model said it changed
    assert not by.loc[("ka0001", "different_hard"), "confirmed"]      # the second model saw no change
    assert not by.loc[("ka0003", "verbatim"), "confirmed"]            # too little of the sentence survived
    assert by.loc[("ka0002", "different_easy"), "confirmed"]
    assert by.loc[("ka0000", "paraphrase"), "a"] == bases["text"][0]
    assert by.loc[("ka0000", "paraphrase"), "b"] == "reworded 0"


def test_all_pairs_from_one_base_are_in_the_same_split():
    bases = bases_frame(40)
    generated = [{"id": i, "verbatim": t, "paraphrase": "p", "different_hard": "h"}
                 for i, t in zip(bases["id"], bases["text"], strict=True)]
    verdicts = {(i, k): v for i in bases["id"] for k, v in (("paraphrase", "same"), ("different_hard", "different"))}
    pairs = assemble_pairs(bases, generated, verdicts, bases, seed=0)
    assert (pairs.groupby("base")["split"].nunique() == 1).all()
    assert pairs.drop_duplicates("base")["split"].value_counts().to_dict() == {"tune": 20, "test": 20}
