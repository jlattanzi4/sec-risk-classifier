# Validation data

Everything the measurement's validation rests on that a rerun can't recreate by itself: the test sets, the prompts, the models' raw replies, and the judges' labels. The pipeline reads these files; it calls a model only for a batch whose reply isn't saved here.

No person labeled anything in this directory. Every label came from a model, and each section below says which one.

## `known_answer/`: pairs whose right answer is known by how they were made

```
uv run python -m pipeline.run known-answer
```

600 real sentences (40 from each fiscal year, 2006 to 2020; full sentences of at least 8 words, drawn with seed 0) each get four partners:

| Kind | How it was made | Right answer |
|---|---|---|
| `verbatim` | Claude Haiku 5.5 changed only names, dates and numbers, or a word or two where there were none | the same sentence |
| `paraphrase` | Claude Haiku 5.5 reworded it, keeping the claim | same claim, new words |
| `different_hard` | Claude Haiku 5.5 kept the topic and most of the words and changed the claim | a different claim |
| `different_easy` | a random sentence from another company and another year | a different claim |

A second model, Claude Sonnet 5.5, then read every paraphrase and every hard edit, shuffled together under ids that don't say which is which, and answered one question: does B make the same claim as A? A paraphrase is kept only if it said "same", a hard edit only if it said "different". A `verbatim` edit is kept only if at least 60% of its words survive in order.

| Kind | Kept | Rejected |
|---|---|---|
| `verbatim` | 600 | 0 |
| `paraphrase` | 597 | 3 |
| `different_hard` | 600 | 0 |
| `different_easy` | 600 | 0 |

63 of the 600 `verbatim` pairs are identical once normalized (only numbers or punctuation changed).

Files:

- `bases.csv`: the 600 sentences, with the filing each came from.
- `pairs.csv`: every pair, with `kind`, `split` (`tune` or `test`; all four pairs of a sentence share a split), and `confirmed`.
- `prompts/generate_v1.md`, `prompts/confirm_v1.md`: the instructions, word for word.
- `raw/generate/`, `raw/confirm/`: for each batch, the full prompt sent (`.request.json`) and the reply (`.response.json`: the text, the models used, token counts and duration). 24 batches of 25 sentences to Haiku, 24 batches of 50 pairs to Sonnet.
- `confirm_key.csv`: which pair each anonymous id in the confirmation step stood for.

**Read before use (2026-11-11, by Claude in the build session; Joe did not hand-check):** 12 random pairs of each generated kind, and all 3 rejections.

- Every pair read was what its kind says it is.
- All 3 rejections were right: one paraphrase moved a future event into the past, one added "rather than acquisitions", one changed who does what.
- **The hard edits are mostly one-word reversals** ("adverse" to "beneficial", "encourage" to "discourage"). They keep nearly every run of five words, so by word overlap they look the same as a `verbatim` edit that swapped a synonym. That is the point of the kind, and it means no threshold on word overlap can separate the two. `docs/measurement.md` reports how the matcher labels them instead of hiding it.
- The hard edits often read as implausible for a risk section ("could have a material beneficial effect"). They test the matcher's mechanics, not how often companies really make such edits. The real pairs below answer that.

## `real_pairs/`: sentences from real consecutive filings, labeled by two judges

```
uv run python -m pipeline.run judge-sample judge thresholds
```

The known-answer set shows the matcher works on cases built to have an answer. This checks it on what it meets in practice. Each pair is a sentence that exact matching left over (B, this year) and the previous-year sentence the matcher found closest to it by cosine (A). 2,000 pairs were drawn evenly across 34 bands of cosine by Jaccard (seed 0), so every band a threshold might fall in has pairs. They stand for 3,795,151 such sentences, and each carries the weight that turns the sample back into that population.

Two judges, same rubric, three labels (same, paraphrase, different):

- **Judge 1: Claude Haiku 5.5**, all 2,000 pairs, 40 to a call.
- **Judge 2: Claude, in the build session (2026-11-11)**, a 400-pair subset spread evenly over the bands. It was shown only an id and the two sentences, in its own shuffled order: no scores, no bands, no judge-1 labels. Its labels were written before judge 1's were opened.

### Round 1 (rubric v1): the judges did not agree

| | Judge 2: same | paraphrase | different |
|---|---|---|---|
| **Judge 1: same** | 90 | 0 | 1 |
| **paraphrase** | 19 | 14 | 1 |
| **different** | 135 | 12 | 128 |

Cohen's kappa 0.336, agreement 58.0%. The plan had fixed 0.70 as good.

Nearly all of the disagreement is one cell: 135 pairs judge 1 called different and judge 2 called same. Reading them, judge 1 was calling a sentence different whenever a detail in it had been updated ("Procurestaff is a vendor management company." against "Beeline is a vendor management company."), which rubric v1 lists as a light edit. Judge 2, for its part, had made three kinds of call the rubric did not spell out, and marked them as it went:

- **41 cut-off pieces.** The sentence splitter sometimes cuts a sentence at a page break, so B is only the tail of A, with a page number in front. Judge 2 labeled these same (the words were carried over; the cut is the splitter's doing). Judge 1 called 37 of them different.
- **12 status updates** ("plans to issue final rules" becoming "issued final rules"). Judge 2 labeled these same, like an updated date. Judge 1 called 11 different.
- Small additions. Judge 2 called a general qualifier or one more item of the same kind a light edit, and a specific new risk, cause or consequence different.

### Round 2 (rubric v2): better, still short of the bar

Rubric v2 writes those cases down, with new examples (none taken from the judged pairs). Judge 1 re-labeled all 2,000 pairs under it. Judge 2's labels were not changed.

| | Judge 2: same | paraphrase | different |
|---|---|---|---|
| **Judge 1: same** | 164 | 0 | 1 |
| **paraphrase** | 21 | 17 | 3 |
| **different** | 59 | 9 | 126 |

Kappa 0.602, agreement 76.7%. On the two-way question that matters most, carried over or not, kappa is 0.636 and agreement 82.0%.

Two things to keep in mind when reading round 2:

- **Rubric v2 was written by judge 2 after seeing the disagreement**, and it writes down judge 2's reading. So round 2 measures whether Haiku can follow that reading, not whether the reading is right.
- The disagreement that remains is again mostly one cell (59 pairs), and it is a real difference of standard: judge 1 calls an edited sentence different when the edit changes its content at all (one more trademark in a list of ten), judge 2 when it adds a specific new risk. "Carried over" does not have a sharp edge for edited sentences, and two careful readers put it in different places.

### What is used

Kappa stayed under 0.70, so by the rule fixed in the plan the real-data check uses **the 307 pairs both judges agree on** (164 same, 17 paraphrase, 126 different), re-weighted to the population. `results/thresholds.json` also holds every other view: each judge alone, and judge 1 on the 1,600 pairs judge 2 never saw.

Files:

- `pairs.csv`: the 2,000 pairs with their band (`stratum`), scores and weight.
- `judge1_input.csv`, `judge2_input.csv`: exactly what each judge was shown.
- `rubric_v1.md`, `rubric_v2.md`: the instructions, word for word.
- `raw/judge1_v1/`, `raw/judge1_v2/`: each batch's prompt and reply, 50 batches a round.
- `judge1_labels_v1.csv`, `judge1_labels_v2.csv`, `judge2_labels.csv` (with judge 2's `note` on the calls above).
