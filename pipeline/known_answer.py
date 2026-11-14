"""A test set whose right answers are known by how each pair was made.

Each of 600 real sentences gets four partners:

- `verbatim`: the same sentence with names, dates and numbers changed;
- `paraphrase`: the same claim reworded;
- `different_hard`: most of the same words, a different claim;
- `different_easy`: a random sentence from another company and year.

The first three are written by Claude Haiku. A second model (Claude Sonnet)
then reads every paraphrase and every hard edit without being told which is
which and says whether the claim survived; a pair is kept only if that
verdict matches what it was meant to be. Prompts and raw replies are saved
beside the result.
"""

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.io import read_sentences, sentence_years
from pipeline.llm import claude_cli, run_batches
from pipeline.sentences import normalize

MIN_WORDS = 8
GENERATED = ("verbatim", "paraphrase", "different_hard")
EXPECTED_VERDICT = {"paraphrase": "same", "different_hard": "different"}
SUFFIX = {"verbatim": "v", "paraphrase": "p", "different_hard": "h", "different_easy": "e"}
PAIR_COLUMNS = [
    "id", "base", "kind", "split", "a", "b", "cik", "filing_year", "sent_idx", "confirmed",
]  # fmt: skip


def _eligible(cfg: Config, year: int) -> pd.DataFrame:
    """Full sentences: not a heading, not flagged long, at least MIN_WORDS words."""
    df = read_sentences(
        cfg, year, ["cik", "filing_year", "sent_idx", "text", "norm", "is_heading", "is_long"]
    )
    words = df["norm"].str.count(" ") + 1
    keep = ~df["is_heading"] & ~df["is_long"] & (words >= MIN_WORDS)
    return df.loc[keep, ["cik", "filing_year", "sent_idx", "text"]]


def sample_bases(cfg: Config, n_per_year: int = 40, seed: int = 0) -> pd.DataFrame:
    """`n_per_year` full sentences from each fiscal year, with ids `ka0000`, `ka0001`, ..."""
    parts = [
        _eligible(cfg, year).sample(n_per_year, random_state=seed) for year in sentence_years(cfg)
    ]
    bases = pd.concat(parts, ignore_index=True)
    bases.insert(0, "id", [f"ka{i:04d}" for i in range(len(bases))])
    return bases


def easy_partners(cfg: Config, bases: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """For each base, a random sentence from a different company and a different year."""
    years = sentence_years(cfg)
    per_year = bases.groupby("filing_year").size()
    parts = []
    for i, year in enumerate(years):
        # Sentences for year i's bases come from the year halfway round the corpus.
        other = years[(i + len(years) // 2) % len(years)]
        pool = _eligible(cfg, other)
        pool = pool[~pool["cik"].isin(bases.loc[bases["filing_year"] == year, "cik"])]
        parts.append(pool.sample(int(per_year[year]), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def survives(a: str, b: str, share: float = 0.6) -> bool:
    """Whether `b` is still mostly `a`: at least `share` of the words match in order."""
    left, right = normalize(a).split(), normalize(b).split()
    return SequenceMatcher(None, left, right, autojunk=False).ratio() >= share


def assemble_pairs(
    bases: pd.DataFrame,
    generated: list[dict],
    verdicts: dict[tuple[str, str], str],
    partners: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """The test set: one row per base and kind.

    `generated` holds the model's three rewrites per base id; `verdicts` maps
    (base id, kind) to the second model's "same" or "different"; `partners`
    is row-aligned with `bases`. `confirmed` is False for a paraphrase judged
    different, a hard edit judged the same, or a `verbatim` edit that kept
    under 60% of the words. Bases are split in half, `tune` and `test`, and a
    base's pairs all share its split.
    """
    rewrites = {g["id"]: g for g in generated}
    order = np.random.default_rng(seed).permutation(len(bases))
    split = np.where(np.argsort(order) < len(bases) // 2, "tune", "test")
    rows = []
    for position, base in enumerate(bases.itertuples(index=False)):
        common = {
            "base": base.id, "split": split[position], "a": base.text,
            "cik": base.cik, "filing_year": base.filing_year, "sent_idx": base.sent_idx,
        }  # fmt: skip
        easy = {"kind": "different_easy", "b": partners["text"].iloc[position], "confirmed": True}
        made = []
        for kind in GENERATED if base.id in rewrites else ():
            text = rewrites[base.id][kind].strip()
            if kind == "verbatim":
                confirmed = survives(base.text, text)
            else:
                confirmed = verdicts.get((base.id, kind)) == EXPECTED_VERDICT[kind]
            made.append({"kind": kind, "b": text, "confirmed": confirmed})
        for pair in [*made, easy]:
            rows.append({"id": f"{base.id}-{SUFFIX[pair['kind']]}", **common, **pair})
    return pd.DataFrame(rows)[PAIR_COLUMNS]


def build_known_answer(cfg: Config, call=claude_cli, n_per_year: int = 40) -> Path:
    """Sample the bases, have the models write and check the rewrites, and save the set."""
    root = cfg.validation_dir / "known_answer"
    root.mkdir(parents=True, exist_ok=True)
    bases = sample_bases(cfg, n_per_year)
    bases.to_csv(root / "bases.csv", index=False)

    generated = run_batches(
        bases[["id", "text"]].to_dict("records"),
        prompt_file=root / "prompts" / "generate_v1.md",
        out_dir=root / "raw" / "generate",
        model="haiku",
        batch_size=25,
        required=GENERATED,
        call=call,
    )

    # The second model sees paraphrases and hard edits mixed together, under ids
    # that say nothing about which is which.
    text = bases.set_index("id")["text"]
    to_check = pd.DataFrame(
        [
            {"base": g["id"], "kind": kind, "a": text[g["id"]], "b": g[kind].strip()}
            for g in generated
            for kind in EXPECTED_VERDICT
        ]
    )
    to_check = to_check.sample(frac=1, random_state=0).reset_index(drop=True)
    to_check.insert(0, "id", [f"c{i:04d}" for i in range(len(to_check))])
    to_check[["id", "base", "kind"]].to_csv(root / "confirm_key.csv", index=False)
    checked = run_batches(
        to_check[["id", "a", "b"]].to_dict("records"),
        prompt_file=root / "prompts" / "confirm_v1.md",
        out_dir=root / "raw" / "confirm",
        model="sonnet",
        batch_size=50,
        required=("label",),
        call=call,
    )
    label = {c["id"]: c["label"].strip().lower() for c in checked}
    verdicts = {
        (row.base, row.kind): label[row.id] for row in to_check.itertuples() if row.id in label
    }

    pairs = assemble_pairs(bases, generated, verdicts, easy_partners(cfg, bases))
    pairs.to_csv(root / "pairs.csv", index=False)
    return root / "pairs.csv"


def summarize_known_answer(pairs: pd.DataFrame) -> str:
    """Kept and rejected pairs by kind and split."""
    table = pairs.groupby(["kind", "confirmed"]).size().unstack(fill_value=0)
    table = table.reindex(columns=[True, False], fill_value=0)
    lines = [f"{'kind':<16}{'kept':>6}{'rejected':>10}{'tune':>6}{'test':>6}"]
    for kind in SUFFIX:
        if kind not in table.index:
            continue
        kept = pairs[(pairs["kind"] == kind) & pairs["confirmed"]]
        by_split = kept["split"].value_counts()
        lines.append(
            f"{kind:<16}{table.loc[kind, True]:>6}{table.loc[kind, False]:>10}"
            f"{by_split.get('tune', 0):>6}{by_split.get('test', 0):>6}"
        )
    lines.append(f"bases: {pairs['base'].nunique()}")
    return "\n".join(lines)
