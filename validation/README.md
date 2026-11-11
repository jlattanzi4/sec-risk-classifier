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
