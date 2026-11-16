"""Follow each sentence a company adds until the company removes it.

Milestone 2 links every sentence to its match in the next filing. Chained
together, those links give each added sentence a life: the number of
consecutive filings it appears in, and whether it ended (the next filing has
no match for it) or was still there when the data stopped (the company's last
filing, a missing year, or the last year of the corpus).

Only sentences whose start was seen are followed: labeled `new` in a filing
whose previous-year filing exists. A sentence already in a company's first
filing has no known age.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences, write_parquet
from pipeline.labels import PEER_GROUPS

CHAIN_COLUMNS = [
    "cik", "birth_year", "sent_idx", "duration", "event",
    "peer_shared", "n_words", "position", "section_sentences", "sic_division",
]  # fmt: skip


def follow_chains(
    births: pd.DataFrame, next_year: dict[int, pd.DataFrame], last_year: int
) -> pd.DataFrame:
    """Give each chain its `duration` and whether it ended in removal (`event`).

    `births` has one row per chain: `cik`, `birth_year`, `sent_idx`.
    `next_year[y]` says what became, in fiscal year y, of each sentence of the
    same company's filing for y - 1: `cik`, `sent_idx`, `label` (`removed`, or
    how it was matched) and `link_idx` (where it continues). A company absent
    from `next_year[y]` did not file for both years, so its chains are
    censored there.
    """
    out = births.reset_index(drop=True).copy()
    out["duration"], out["event"] = 0, False
    alive = pd.DataFrame(columns=["chain", "cik", "sent_idx"]).astype("int64")
    first = int(out["birth_year"].min()) if len(out) else last_year
    for year in range(first, last_year + 1):
        born = out.index[out["birth_year"] == year]
        alive = pd.concat(
            [
                alive,
                pd.DataFrame(
                    {
                        "chain": born,
                        "cik": out.loc[born, "cik"],
                        "sent_idx": out.loc[born, "sent_idx"],
                    }
                ),
            ],
            ignore_index=True,
        )
        out.loc[alive["chain"], "duration"] += 1
        fate = next_year.get(year + 1)
        if fate is None or alive.empty:
            alive = alive.iloc[0:0]
            continue
        met = alive.merge(fate, on=["cik", "sent_idx"], how="left")
        removed = met["label"] == "removed"
        out.loc[met.loc[removed, "chain"], "event"] = True
        carried = met["label"].notna() & ~removed  # no row: the company has no pair this year
        alive = pd.DataFrame(
            {
                "chain": met.loc[carried, "chain"].to_numpy(),
                "cik": met.loc[carried, "cik"].to_numpy(),
                "sent_idx": met.loc[carried, "link_idx"].to_numpy(dtype=np.int64),
            }
        )
    return out


def build_chains(cfg: Config) -> Path:
    """One row per sentence added in a filing whose previous year exists: CHAIN_COLUMNS.

    The covariates are fixed at birth: whether the sentence is shared with
    peers that year, its length in words, where it sits in the section (0 at
    the top, 1 at the bottom), the section's length in sentences, and the
    company's SIC division.
    """
    births, next_year = [], {}
    paths = sorted((cfg.interim_dir / "labels").glob("pair=*.parquet"))
    for path in paths:
        year = int(path.stem.split("=")[1])
        labels = pd.read_parquet(path)
        previous = labels[labels["side"] == "prev"]
        next_year[year] = previous[["cik", "sent_idx", "label", "link_idx"]]

        sentences = read_sentences(cfg, year, ["cik", "sent_idx", "norm", "norm_hash"])
        section = sentences.groupby("cik")["sent_idx"].transform("size")
        sentences = sentences.assign(
            section_sentences=section,
            position=sentences["sent_idx"] / np.maximum(section - 1, 1),
            n_words=sentences["norm"].str.count(" ") + 1,
        ).drop(columns="norm")
        new = labels[(labels["side"] == "cur") & (labels["label"] == "new")][["cik", "sent_idx"]]
        new = new.merge(sentences, on=["cik", "sent_idx"])
        # A sentence added twice in one filing is one addition.
        new = new.sort_values(["cik", "sent_idx"], kind="stable").drop_duplicates(
            ["cik", "norm_hash"]
        )
        peers = pd.read_parquet(cfg.interim_dir / "peers" / f"year={year}.parquet")
        others = new["norm_hash"].map(peers.set_index("norm_hash")["n_groups"]) - 1
        births.append(new.assign(birth_year=year, peer_shared=(others >= PEER_GROUPS).to_numpy()))

    last_year = max(int(p.stem.split("=")[1]) for p in paths)
    chains = follow_chains(pd.concat(births, ignore_index=True), next_year, last_year)
    filings = pd.read_parquet(
        cfg.interim_dir / "filings.parquet", columns=["cik", "filing_year", "sic_division"]
    )
    division = filings.rename(columns={"filing_year": "birth_year"})
    chains = chains.merge(division, on=["cik", "birth_year"], how="left")
    out = cfg.interim_dir / "chains.parquet"
    write_parquet(chains[CHAIN_COLUMNS], out)
    return out
