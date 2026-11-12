import pandas as pd

from pipeline.chains import follow_chains


def births(*rows):
    return pd.DataFrame(rows, columns=["cik", "birth_year", "sent_idx"])


def side(*rows):
    """The previous-filing side of a pair: what became of each of last year's sentences."""
    return pd.DataFrame(rows, columns=["cik", "sent_idx", "label", "link_idx"])


def result(chains, cik, birth_year, sent_idx):
    row = chains[
        (chains["cik"] == cik)
        & (chains["birth_year"] == birth_year)
        & (chains["sent_idx"] == sent_idx)
    ]
    return int(row["duration"].iloc[0]), bool(row["event"].iloc[0])


def test_a_sentence_carried_once_and_then_removed():
    chains = follow_chains(
        births((1, 2017, 4)),
        {2018: side((1, 4, "exact", 6)), 2019: side((1, 6, "removed", -1))},
        last_year=2019,
    )
    assert result(chains, 1, 2017, 4) == (2, True)


def test_a_sentence_still_there_when_the_data_ends_is_censored():
    chains = follow_chains(
        births((1, 2017, 4)),
        {2018: side((1, 4, "near", 6)), 2019: side((1, 6, "paraphrase", 2))},
        last_year=2019,
    )
    assert result(chains, 1, 2017, 4) == (3, False)


def test_a_sentence_removed_at_once_lasted_one_filing():
    chains = follow_chains(
        births((1, 2017, 4)), {2018: side((1, 4, "removed", -1))}, last_year=2019
    )
    assert result(chains, 1, 2017, 4) == (1, True)


def test_a_sentence_born_in_the_last_year_is_censored_at_one():
    chains = follow_chains(births((1, 2019, 0)), {}, last_year=2019)
    assert result(chains, 1, 2019, 0) == (1, False)


def test_a_missing_year_censors_and_never_counts_as_removal():
    # The company filed for 2017 and 2019 but not 2018: there is no pair for 2018 or 2019.
    chains = follow_chains(
        births((1, 2017, 4)), {2018: side((2, 0, "removed", -1))}, last_year=2019
    )
    assert result(chains, 1, 2017, 4) == (1, False)


def test_two_sentences_merged_into_one_both_go_on():
    chains = follow_chains(
        births((1, 2017, 4), (1, 2017, 5)),
        {2018: side((1, 4, "near", 9), (1, 5, "paraphrase", 9)), 2019: side((1, 9, "removed", -1))},
        last_year=2019,
    )
    assert result(chains, 1, 2017, 4) == (2, True) and result(chains, 1, 2017, 5) == (2, True)


def test_chains_of_different_ages_and_companies_do_not_interfere():
    chains = follow_chains(
        births((1, 2017, 0), (1, 2018, 3), (2, 2018, 0)),
        {
            2018: side((1, 0, "exact", 1)),
            2019: side((1, 1, "exact", 2), (1, 3, "removed", -1), (2, 0, "exact", 0)),
        },
        last_year=2019,
    )
    assert result(chains, 1, 2017, 0) == (3, False)
    assert result(chains, 1, 2018, 3) == (1, True)
    assert result(chains, 2, 2018, 0) == (2, False)
    assert len(chains) == 3
