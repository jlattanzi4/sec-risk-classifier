"""Choose the embedding model on the tuning half of the known-answer set.

    uv run python -m scripts.compare_embedders

A paraphrase should score higher than a sentence that says something else. For
each candidate this prints the average precision of cosine similarity at
ranking paraphrases above the two kinds of different sentence, its speed on
10,000 corpus sentences, and the share of them it would cut off. The rule,
fixed in the plan before this was run: the highest average precision wins, and
within 0.005 the faster model does.

Only pairs with `split == "tune"` are used, so the other half stays unseen for
the threshold report. Results go to results/embedders.json.
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pipeline.config import load_config
from pipeline.embed import CANDIDATES, load_embedder
from pipeline.io import read_sentences

SPEED_SAMPLE = 10_000


def main() -> None:
    cfg = load_config()
    pairs = pd.read_csv(cfg.validation_dir / "known_answer" / "pairs.csv")
    pairs = pairs[pairs["confirmed"] & (pairs["split"] == "tune") & (pairs["kind"] != "verbatim")]
    corpus = read_sentences(cfg, 2019, ["text"])["text"].sample(SPEED_SAMPLE, random_state=0)
    corpus = corpus.tolist()

    rows = []
    for name in CANDIDATES:
        embedder = load_embedder(name)
        embedder.encode(corpus[:256])                       # load weights and warm up
        start = time.perf_counter()
        embedder.encode(corpus)
        per_second = SPEED_SAMPLE / (time.perf_counter() - start)

        a = embedder.encode(pairs["a"].tolist())
        b = embedder.encode(pairs["b"].tolist())
        cosine = (a * b).sum(axis=1)
        is_paraphrase = (pairs["kind"] == "paraphrase").to_numpy()

        def precision(negative: str | None, cosine=cosine, is_paraphrase=is_paraphrase) -> float:
            keep = is_paraphrase | (pairs["kind"] == negative).to_numpy() if negative else slice(None)
            return float(average_precision_score(is_paraphrase[keep], cosine[keep]))

        rows.append(
            {
                "model": name,
                "ap": precision(None),
                "ap_vs_easy": precision("different_easy"),
                "ap_vs_hard": precision("different_hard"),
                "median_cos": {
                    kind: float(np.median(cosine[(pairs["kind"] == kind).to_numpy()]))
                    for kind in ("paraphrase", "different_hard", "different_easy")
                },
                "sentences_per_second": per_second,
                "over_token_limit": embedder.over_limit(corpus),
                "max_tokens": embedder.max_tokens,
                "dim": embedder.dim,
            }
        )
        del embedder

    best = max(r["ap"] for r in rows)
    chosen = max((r for r in rows if r["ap"] >= best - 0.005), key=lambda r: r["sentences_per_second"])
    out = {
        "n_pairs": {k: int(v) for k, v in pairs["kind"].value_counts().items()},
        "speed_sample": SPEED_SAMPLE,
        "candidates": rows,
        "chosen": chosen["model"],
    }
    cfg.results_dir.mkdir(exist_ok=True)
    (cfg.results_dir / "embedders.json").write_text(json.dumps(out, indent=1) + "\n")

    print(f"tuning pairs: {out['n_pairs']}")
    print(f"{'model':<42}{'AP':>7}{'vs easy':>9}{'vs hard':>9}{'para':>7}{'hard':>7}{'easy':>7}{'sent/s':>9}{'cut off':>9}")
    for r in rows:
        m = r["median_cos"]
        print(
            f"{r['model']:<42}{r['ap']:>7.3f}{r['ap_vs_easy']:>9.3f}{r['ap_vs_hard']:>9.3f}"
            f"{m['paraphrase']:>7.2f}{m['different_hard']:>7.2f}{m['different_easy']:>7.2f}"
            f"{r['sentences_per_second']:>9,.0f}{r['over_token_limit']:>9.2%}"
        )
    print(f"chosen: {chosen['model']}")


if __name__ == "__main__":
    main()
