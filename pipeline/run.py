"""Entry point: `python -m pipeline.run STAGE [STAGE ...] [--force] [--workers N]`.

Stages run in the order given. `all` runs every stage in pipeline order.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.candidates import build_candidates, summarize_candidates
from pipeline.config import load_config
from pipeline.embed import MODEL, build_embeddings, build_similarity
from pipeline.industry import SIC_DIVISIONS
from pipeline.judging import judge_with_haiku, sample_real_pairs, write_judge_inputs
from pipeline.known_answer import build_known_answer, summarize_known_answer
from pipeline.labels import build_filing_metrics, build_labels
from pipeline.peers import build_peers
from pipeline.stages import CORPUS, build_filings, build_sentences, sentence_stats
from pipeline.thresholds import build_thresholds, read_thresholds


def summarize_filings(df: pd.DataFrame) -> str:
    """The checks recorded in docs/data-foundation.md, as printable text."""
    status = df["match_status"].value_counts()
    divisions = df["sic_division"].value_counts(dropna=False).sort_index(na_position="last")
    lines = [f"filings: {len(df):,}", "match_status:"]
    lines += [f"  {s:<15}{n:>7,}  {n / len(df):6.1%}" for s, n in status.items()]
    dated = df["match_status"].isin(["unique", "multiple"]).mean()
    lines.append(f"dated by an original 10-K (unique or multiple): {dated:.1%}")
    lines.append(f"with a SIC division: {df['sic_division'].notna().mean():.1%}")
    lines.append("filings by SIC division:")
    for letter, n in divisions.items():
        name = SIC_DIVISIONS[letter][2] if isinstance(letter, str) else "(none)"
        lines.append(f"  {letter if isinstance(letter, str) else '-'}  {name:<42}{n:>7,}")
    return "\n".join(lines)


def summarize_sentences(sentences_dir: Path, n_filings: int) -> str:
    """The sentence-table checks recorded in docs/data-foundation.md, as printable text."""
    files = sorted(sentences_dir.glob("year=*.parquet"))
    stats = sentence_stats(sentences_dir).sum()
    df = pd.concat(
        pd.read_parquet(f, columns=["cik", "filing_year", "n_chars", "is_long", "is_heading"])
        for f in files
    )
    per_filing = df.groupby(["cik", "filing_year"]).size()
    megabytes = sum(f.stat().st_size for f in files) / 1e6
    n = len(df)
    return "\n".join(
        [
            f"sentences: {n:,} in {len(files)} year files ({megabytes:,.0f} MB on disk)",
            f"filings with no sentences: {n_filings - len(per_filing):,} of {n_filings:,}",
            "sentences per filing: median {:,.0f}, 10th percentile {:,.0f}, 90th {:,.0f}".format(
                *per_filing.quantile([0.5, 0.1, 0.9])
            ),
            "characters per sentence: median {:,.0f}, 10th percentile {:,.0f}, 90th {:,.0f}".format(
                *df["n_chars"].quantile([0.5, 0.1, 0.9])
            ),
            (
                f"dropped as debris: {stats['debris']:,} of {stats['segments']:,} segments "
                f"({stats['debris'] / stats['segments']:.1%})"
            ),
            f"flagged long: {df['is_long'].sum():,} ({df['is_long'].mean():.2%})",
            f"headings: {df['is_heading'].sum():,} ({df['is_heading'].mean():.1%})",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description=__doc__)
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], help="what to build")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--workers", type=int, default=8, help="processes for the parallel stages")
    args = parser.parse_args(argv)

    cfg = load_config()
    for stage in STAGES if "all" in args.stages else args.stages:
        start = time.perf_counter()
        STAGES[stage](cfg, args)
        print(f"[{stage}] done in {time.perf_counter() - start:,.0f} s\n")


def _filings(cfg, args) -> None:
    path = build_filings(cfg, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}")
    print(summarize_filings(pd.read_parquet(path)))


def _sentences(cfg, args) -> None:
    path = build_sentences(cfg, workers=args.workers, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}/")
    n_filings = pq.read_metadata(cfg.root / CORPUS).num_rows
    print(summarize_sentences(path, n_filings))


def _candidates(cfg, args) -> None:
    path = build_candidates(cfg, workers=args.workers, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}/")
    print(summarize_candidates(path))


def _known_answer(cfg, args) -> None:
    # Batches with a saved reply are read back, so this calls a model only for what is missing.
    path = build_known_answer(cfg)
    print(f"wrote {path.relative_to(cfg.root)}")
    print(summarize_known_answer(pd.read_csv(path)))


def _embeddings(cfg, args) -> None:
    print(f"embedding with {MODEL} (finished years are skipped)")
    path = build_embeddings(cfg, force=args.force)
    files = sorted(path.glob("year=*.npy"))
    megabytes = sum(f.stat().st_size for f in files) / 1e6
    print(f"wrote {path.relative_to(cfg.root)}/: {len(files)} years, {megabytes:,.0f} MB")


def _similarity(cfg, args) -> None:
    path = build_similarity(cfg, force=args.force)
    print(f"wrote {path.relative_to(cfg.root)}/")


def _judge_sample(cfg, args) -> None:
    folder = cfg.validation_dir / "real_pairs"
    if (folder / "pairs.csv").exists() and not args.force:
        # Labels are tied to this sample; drawing again would orphan them.
        print(f"using {(folder / 'pairs.csv').relative_to(cfg.root)} (pass --force to draw again)")
    else:
        write_judge_inputs(sample_real_pairs(cfg), folder)
        print(f"wrote {folder.relative_to(cfg.root)}/")
    pairs = pd.read_csv(folder / "pairs.csv")
    print(f"{len(pairs):,} pairs in {pairs['stratum'].nunique()} cells of cosine by Jaccard")
    print(pairs.groupby("stratum").agg(sampled=("id", "size"), stands_for=("weight", "sum")))


def _judge(cfg, args) -> None:
    labels = judge_with_haiku(cfg)
    print(f"judge 1 (Haiku): {len(labels):,} labels")
    print(labels["label"].value_counts().to_string())


def _thresholds(cfg, args) -> None:
    path = build_thresholds(cfg)
    saved = json.loads(path.read_text())
    print(f"wrote {path.relative_to(cfg.root)}")
    print(f"tau_v = {saved['tau_v']:.3f} (F1 {saved['f1_v']:.3f} on the tuning half)")
    print(f"tau_p = {saved['tau_p']:.3f} (F1 {saved['f1_p']:.3f} on the tuning half)")
    known = saved["known_answer"]
    print(f"held-out half ({known['n']} pairs), how each kind was labeled:")
    for kind, counts in known["labeled"].items():
        print(f"  {kind:<16}" + "".join(f"{tier} {n:>4}   " for tier, n in counts.items()))
    carried = known["carried_over"]
    for name in ("precision", "precision_without_hard", "recall"):
        share = carried[name]
        print(
            f"  carried over, {name}: {share['value']:.3f} ({share['low']:.3f}-{share['high']:.3f})"
        )
    if "real_pairs" in saved:
        real = saved["real_pairs"]
        agree = real["agreement"]
        print(
            f"judges on {agree['n']} shared pairs: kappa {agree['kappa']:.3f} "
            f"(carried over or not: {agree['kappa_carried']:.3f}), agreement {agree['agreement']:.1%}"
        )
        print(f"real pairs, by {real['basis']} ({real['n']} pairs):")
        for tier in ("near", "paraphrase", "new"):
            row = real["makeup"][tier]
            print(
                f"  {tier:<11}same {row['same']:.1%}  paraphrase {row['paraphrase']:.1%}  "
                f"different {row['different']:.1%}  ({row['n']} pairs for {row['stands_for']:,.0f})"
            )
        print(f"  recall of carried-over pairs: {real['makeup']['recall_carried']:.1%}")


def _labels(cfg, args) -> None:
    tau_v, tau_p = read_thresholds(cfg)
    build_labels(cfg, tau_v, tau_p)
    metrics = pd.read_parquet(build_filing_metrics(cfg))
    print(f"labeled with tau_v = {tau_v:.3f}, tau_p = {tau_p:.3f}; {len(metrics):,} paired filings")
    shares = [
        "share_exact",
        "share_near",
        "share_paraphrase",
        "share_new",
        "share_removed",
        "share_peer",
    ]
    weighted = {
        s: np.average(
            metrics[s], weights=metrics["n_prev" if s == "share_removed" else "n_sentences"]
        )
        for s in shares
    }
    print("pooled over sentences: " + "  ".join(f"{s[6:]} {v:.1%}" for s, v in weighted.items()))
    print(
        "median filing:         " + "  ".join(f"{s[6:]} {metrics[s].median():.1%}" for s in shares)
    )


def _peers(cfg, args) -> None:
    path = build_peers(cfg, force=args.force)
    groups = pd.read_parquet(cfg.interim_dir / "filer_groups.parquet")
    sizes = groups.groupby(["filing_year", "group"]).size()
    print(f"wrote {path.relative_to(cfg.root)}/ and filer_groups.parquet")
    print(
        f"{len(groups):,} filings in {len(sizes):,} groups; "
        f"{int(sizes[sizes > 1].sum()):,} filings are in a group of two or more; "
        f"largest group {int(sizes.max())}"
    )


# In pipeline order: each stage reads what the ones before it wrote.
STAGES = {
    "filings": _filings,
    "sentences": _sentences,
    "candidates": _candidates,
    "known-answer": _known_answer,
    "embeddings": _embeddings,
    "similarity": _similarity,
    "judge-sample": _judge_sample,
    "judge": _judge,
    "peers": _peers,
    "thresholds": _thresholds,
    "labels": _labels,
}


if __name__ == "__main__":
    main()
