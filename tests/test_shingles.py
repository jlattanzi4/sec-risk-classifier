import random
import subprocess
import sys

import numpy as np
import pytest

from pipeline.shingles import (
    best_matches,
    jaccard,
    lsh_candidates,
    minhash,
    shingle_hashes,
    shingles,
)

VOCAB = [f"w{i}" for i in range(30)]


def random_sentences(n: int, seed: int, words=(4, 30)) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(VOCAB, k=rng.randint(*words))) for _ in range(n)]


def test_a_sentence_is_its_runs_of_five_words():
    assert shingles("a b c d e f") == {"a b c d e", "b c d e f"}


def test_a_sentence_under_five_words_is_one_shingle():
    assert shingles("a b c") == {"a b c"}
    assert shingles("") == frozenset()


def test_jaccard_of_identical_and_disjoint_sets():
    s = shingles("a b c d e f g")
    assert jaccard(s, s) == 1.0
    assert jaccard(s, shingles("t u v w x y z")) == 0.0


def test_jaccard_counts_shared_over_all():
    # Three shingles each; changing the last word leaves two of them shared, four in all.
    assert jaccard(shingles("a b c d e f g"), shingles("a b c d e f x")) == pytest.approx(2 / 4)


def test_jaccard_of_two_empty_sets_is_zero():
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_short_sentences_match_only_when_identical():
    assert jaccard(shingles("we may fail"), shingles("we may fail")) == 1.0
    assert jaccard(shingles("we may fail"), shingles("we may not fail")) == 0.0


def test_best_matches_agrees_with_brute_force():
    left = [shingles(s) for s in random_sentences(200, seed=1)]
    right = [shingles(s) for s in random_sentences(200, seed=2)]
    # Plant some real overlaps so the test isn't all zeros.
    right[5], right[17] = left[40], left[3] | {"extra shingle of five words"}
    scores, where = best_matches(left, right)
    for i, a in enumerate(left):
        brute = [jaccard(a, b) for b in right]
        assert scores[i] == pytest.approx(max(brute))
        if max(brute) > 0:
            assert brute[where[i]] == pytest.approx(max(brute))
        else:
            assert where[i] == -1
    assert scores[40] == 1.0 and where[40] == 5


def test_best_matches_with_nothing_to_match():
    scores, where = best_matches([shingles("a b c d e f")], [])
    assert scores.tolist() == [0.0] and where.tolist() == [-1]
    scores, where = best_matches([], [shingles("a b c d e f")])
    assert len(scores) == 0 and len(where) == 0


def test_shingle_hashes_are_equal_for_equal_windows():
    hashes, starts = shingle_hashes(["a b c d e f", "x a b c d e", "a b"])
    assert starts.tolist() == [0, 2, 4]  # 2 shingles, 2 shingles, then 1
    assert len(hashes) == 5
    assert hashes[0] == hashes[3]  # "a b c d e" in both
    assert len(set(hashes.tolist())) == 4


def test_shingle_hashes_do_not_depend_on_the_process():
    code = (
        "from pipeline.shingles import shingle_hashes;"
        "print(shingle_hashes(['we may not be able to protect our rights', 'we may fail'])[0].tolist())"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outs) == 1


def test_minhash_estimates_jaccard():
    rng = random.Random(0)
    base = random_sentences(2000, seed=3, words=(15, 40))
    edited = []
    for s in base:  # change a few words so Jaccard spreads over 0–1
        words = s.split()
        for _ in range(rng.randint(0, 6)):
            words[rng.randrange(len(words))] = rng.choice(VOCAB)
        edited.append(" ".join(words))
    sig = minhash(*shingle_hashes(base + edited))
    estimate = (sig[:2000] == sig[2000:]).mean(axis=1)
    exact = np.array([jaccard(shingles(a), shingles(b)) for a, b in zip(base, edited, strict=True)])
    assert exact.std() > 0.2  # the pairs really do cover the range
    error = estimate - exact
    assert abs(error.mean()) < 0.01  # no bias
    # 126 positions give a standard error of sqrt(J(1-J)/126), about 0.045 at J = 0.5.
    assert abs(error).mean() < 0.04
    assert abs(error).max() < 0.2


def test_lsh_finds_planted_near_duplicates():
    sentences = random_sentences(5000, seed=4, words=(15, 40))
    rng = random.Random(1)
    planted = []
    for q in range(0, 400, 2):  # 200 planted pairs: q+1 is q with one word changed
        words = sentences[q].split()
        words[rng.randrange(len(words))] = "changed"
        sentences[q + 1] = " ".join(words)
        if jaccard(shingles(sentences[q]), shingles(sentences[q + 1])) >= 0.5:
            planted.append(q)
    assert len(planted) > 100
    sig = minhash(*shingle_hashes(sentences))
    found = lsh_candidates(sig, np.array(planted))
    hits = sum((q + 1) in set(c.tolist()) for q, c in zip(planted, found, strict=True))
    assert hits / len(planted) >= 0.95
    assert all(
        q not in set(c.tolist()) for q, c in zip(planted, found, strict=True)
    )  # never itself
