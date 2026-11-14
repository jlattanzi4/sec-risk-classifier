# Measurement

What milestone 2 builds: a label for every sentence of every risk section that has a previous-year section to compare with, and the evidence for how far those labels can be trusted. Every number here was printed by the command named next to it and is stored in `results/`; none is typed in by hand.

```
uv run python -m pipeline.run candidates known-answer embeddings similarity \
    judge-sample judge peers thresholds labels baselines measure-report
```

The stages that call a language model (`known-answer`, `judge`) read their saved replies from `validation/` and call nothing unless a batch is missing. `embeddings` is the long one: 2 hours 10 minutes on an M3 laptop.

## The result

47,643 filings from 10,779 companies have the same company's filing for the year before. Their 14,272,386 sentences split like this:

| A sentence in this year's risk section is... | Share of sentences | Median filing |
|---|---|---|
| **exact**: in last year's section word for word | 73.4% | 78.6% |
| **near**: a light edit of a sentence there | 11.6% | 9.5% |
| **reworded**: the same claim in new words | 1.6% | 0.9% |
| **new** | 13.3% | 8.4% |

Of last year's 13,711,656 sentences, 9.9% have no match this year (**removed**; median filing 5.1%).

So the typical risk section is about 88% last year's text, kept or lightly edited, and 8% new. The spread is wide: a tenth of filings are under 0.5% new, and a tenth are over 29% new. 4,266 filings added nothing at all, and 8 were entirely new.

Weighting by characters instead of sentences moves little: 70.3% exact, 14.5% near, 1.5% reworded, 13.7% new. Long sentences are a little more likely to be edited than left alone.

### By year

| Fiscal year | Filings | Exact | Near | Reworded | New | Removed | Median filing: new |
|---|---|---|---|---|---|---|---|
| 2007 | 3,419 | 66.8% | 14.1% | 2.0% | 17.1% | 11.8% | 9.8% |
| 2008 | 3,414 | 64.3% | 13.4% | 2.2% | 20.1% | 12.7% | 14.9% |
| 2009 | 3,792 | 67.4% | 13.3% | 2.1% | 17.2% | 12.8% | 10.6% |
| 2010 | 3,763 | 68.9% | 12.9% | 2.0% | 16.2% | 12.5% | 9.3% |
| 2011 | 3,669 | 71.4% | 12.5% | 1.8% | 14.3% | 10.2% | 8.1% |
| 2012 | 3,616 | 72.6% | 12.0% | 1.7% | 13.8% | 9.7% | 7.8% |
| 2013 | 3,560 | 74.1% | 11.9% | 1.6% | 12.4% | 9.8% | 6.8% |
| 2014 | 3,453 | 74.9% | 11.6% | 1.5% | 12.0% | 9.0% | 6.9% |
| 2015 | 3,332 | 76.5% | 11.2% | 1.5% | 10.8% | 8.4% | 6.7% |
| 2016 | 3,284 | 77.4% | 10.7% | 1.4% | 10.5% | 7.9% | 6.4% |
| 2017 | 3,174 | 76.8% | 10.8% | 1.4% | 11.0% | 8.4% | 7.2% |
| 2018 | 3,145 | 77.0% | 10.7% | 1.4% | 11.0% | 8.3% | 7.5% |
| 2019 | 3,045 | 77.0% | 10.5% | 1.4% | 11.1% | 8.4% | 7.5% |
| 2020 | 2,977 | 73.0% | 10.3% | 1.7% | 15.0% | 11.9% | 11.9% |

Two things stand out before any modeling. Sections were rewritten less every year from 2008 to 2016: the share carried over word for word rose from 64% to 77%. And two years break the trend: fiscal 2008, written during the financial crisis, and fiscal 2020, written during the pandemic and mostly under the SEC's new risk-factor rules. Milestone 4 takes up 2020.

## How a sentence gets its label

Each filing is compared with the same company's filing for the previous fiscal year, and with nothing else. A gap year breaks the link.

1. **Exact.** The sentence's normalized form (lower case, digits replaced by `0`, punctuation dropped) is in last year's filing. "Revenue fell 12.5% in 2019." and "Revenue fell 14.0% in 2020." are the same sentence here.
2. **Near.** Among the sentences step 1 left over on each side, the sentence shares at least 19.8% of its runs of five words with one from last year (Jaccard of word 5-shingles at least τ_v = 0.198).
3. **Reworded.** Failing that, its embedding has a cosine of at least τ_p = 0.847 with one of last year's leftover sentences.
4. **New** otherwise. A leftover sentence of last year's that nothing matched is **removed**.

A sentence may be matched by several on the other side. It is carried over if it is anywhere in last year's filing, however many times it appears in this one.

Steps 2 and 3 store scores, not labels: every leftover sentence's best Jaccard and best cosine are kept in `data/interim/similarity/`, and labels are a function of those and the two thresholds. Changing a threshold costs nine seconds, not two hours.

Two departures from the design, both made in the plan before any result:

- **Jaccard is computed exactly, not estimated by MinHash**, inside a company's pair. Step 1 leaves 80 sentences per filing on average, so every candidate can be compared outright. MinHash is used where it is needed, across a whole year's sentences (see "Shared with peers").
- **Leftovers are compared with leftovers.** A new sentence that restates one which was also kept word for word (a summary of an existing risk, say) finds nothing to match. "Baselines" below measures how often.

## Can the labels be trusted?

Three checks, in the order they were run. The second one failed, and what was done about it is part of the record.

### 1. Pairs whose right answer is known

`validation/known_answer/`: 600 real sentences, each with four partners made so that the right label is known. A light edit (names, dates, numbers, a word or two), a paraphrase, a hard edit that keeps the words and changes the claim, and an unrelated sentence. Claude Haiku wrote the first three; Claude Sonnet, not told which was which, confirmed each paraphrase kept its claim and each hard edit changed it (3 of 1,200 rejected). Half the sentences are for tuning, half are held out.

**The embedding model was chosen on the tuning half**, by how well cosine ranks a paraphrase above a sentence that says something else (`uv run python -m scripts.compare_embedders`):

| Model | Average precision | Against unrelated sentences | Against hard edits | Sentences/s | Over its token limit |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.356 | 1.000 | 0.356 | 1,767 | 0.04% |
| **bge-small-en-v1.5** (used) | 0.447 | 1.000 | 0.447 | 885 | 0.00% |
| potion-base-8M (static word vectors) | 0.318 | 0.997 | 0.318 | 43,975 | 0.00% |

Every model separates a paraphrase from an unrelated sentence perfectly. **None can tell a paraphrase from a sentence whose claim was reversed**: with as many hard edits as paraphrases, a coin flip scores 0.5, and all three score below it. For bge-small the median cosine is 0.92 for a paraphrase and 0.92 for a hard edit. Embeddings follow the words. That is the same failure the first version of this project made at the scale of whole sections, measured this time.

**Thresholds from the tuning half**, by best F1: τ_v = 0.198 (F1 0.968) and τ_p = 0.774 (F1 0.972). On the held-out half, at the thresholds finally used:

| Kind of pair (held-out half) | Labeled near | Reworded | New |
|---|---|---|---|
| Light edit (300) | 291 | 9 | 0 |
| Paraphrase (298) | 10 | 267 | 21 |
| Unrelated (300) | 0 | 1 | 299 |
| Hard edit: words kept, claim changed (300) | 285 | 7 | 8 |

Leaving the hard edits aside, a "carried over" call is right 99.8% of the time (577 of 578; 95% interval 99.0–100%) and finds 96.5% of the pairs it should (577 of 598; 94.7–97.7%).

The hard edits are the limit of the method, and it is not a small one: **292 of 300 are called carried over.** A sentence that changes "adverse" to "beneficial" shares nearly every run of five words with its source. With them counted, precision is 66.3%. Whether that matters depends on how often companies make such edits, which constructed pairs can't say. Real pairs can.

### 2. Real pairs, two judges

`validation/real_pairs/` (the full account is in [`validation/README.md`](../validation/README.md)): 2,000 real pairs, a leftover sentence and the previous-year sentence closest to it, drawn evenly across bands of cosine by Jaccard and weighted back to the 3.8 million sentences they stand for. Claude Haiku labeled all of them; Claude, working blind in the build session, labeled 400. No person labeled any.

**The judges did not agree well enough.** The plan fixed a kappa of 0.70 as good. Under the first rubric it was 0.34: Haiku called a sentence different whenever a name or a number in it had been updated. A second rubric spelling those cases out brought it to 0.60 (82% agreement on carried over or not). The disagreement that remains is a real difference of standard about edited sentences: one judge calls a sentence different when an edit changes its content at all, the other only when it adds a specific new risk. So, by the rule fixed in advance, the check below uses only the **307 pairs both judges agree on**, and each judge's own view is shown beside it.

**The threshold tuned on constructed pairs failed.** At τ_p = 0.774, 69% of what the matcher called reworded was judged different (95% interval 58–78%). A constructed paraphrase is compared with one unrelated sentence. A real sentence is compared with every leftover sentence of the same company's previous filing, which is about the same business in the same words, and the best of dozens of such candidates clears 0.774 easily.

So τ_p was tuned again, by the same criterion (best F1), on the agreed real pairs: **τ_p = 0.847**. This departs from the plan, which tuned both thresholds on the known-answer set. The Jaccard threshold held up and was left alone.

| What the matcher said | Judged same | Paraphrase | Different | Pairs | Stand for |
|---|---|---|---|---|---|
| Near | 94.5% | 0.3% | 5.2% | 166 | 1,648,113 |
| Reworded | 35.3% | 44.8% | 19.9% | 34 | 259,610 |
| New | 0.2% | 0.3% | 99.6% | 107 | 1,887,428 |

- **Near is right 94.8% of the time** (95% interval 91.6–97.6%).
- **Reworded is right 80.1% of the time**, on 34 pairs (65.5–92.7%). It is the weakest tier and also the smallest: 1.6% of sentences.
- **New is right 99.6% of the time** (99.0–100%), and the matcher finds 99.5% of the pairs judged carried over (98.9–100%).

The reworded tier was tuned and scored on the same 141 pairs, so its 80% is on the generous side. Two views that don't share that problem: judge 2 alone puts it at 78.3% on 52 pairs, and judge 1 alone, on the 1,600 pairs judge 2 never saw, at 51.6% on 230 pairs. Judge 1 is the stricter reader throughout: it puts near at 81.1% right where judge 2 says 96.8%.

Where the judged share changes, by score (agreed pairs):

| Cosine (pairs under τ_v) | Pairs | Judged carried over |
|---|---|---|
| under 0.75 | 78 | 0.1–0.2% |
| 0.75–0.80 | 17 | 0.4% |
| 0.80–0.85 | 13 | 11.4% |
| 0.85–0.90 | 13 | 71.2% |
| 0.90–0.95 | 11 | 79.8% |
| 0.95–1.00 | 9 | 100% |

| Jaccard (near tier) | Pairs | Judged same | Different |
|---|---|---|---|
| 0.198–0.30 | 30 | 66.6% | 30.3% |
| 0.30–0.45 | 46 | 90.6% | 9.4% |
| 0.45–0.60 | 19 | 100% | 0% |
| 0.60–0.80 | 65 | 98.8% | 1.2% |
| 0.80–1.00 | 6 | 100% | 0% |

The weak spot of the near tier is its bottom band, where a sentence shares a fifth to a third of its phrasing with its match: a third of those were judged different. That band holds about 158,000 sentences, 1.1% of all.

Edits that change what a sentence says are far rarer in real filings than in the constructed set, where they are half of the near matches. Among the 166 agreed pairs the matcher called near, 5.2% by weight were judged different: a sentence with a clause of new content added, or a figure whose change reverses the point (no borrowings outstanding one year, $200 million the next).

### 3. Baselines

What the same method reports when there is nothing to find (`uv run python -m pipeline.run baselines`, half an hour).

**Two unrelated sentences almost never match.** Of 50,000 pairs of sentences from different companies and different years, 2 were labeled near and 4 reworded. The median cosine between two unrelated risk sentences is 0.54 and the 99th percentile is 0.74, well under τ_p.

**A filing's worth of candidates is a harder test.** The pipeline does not compare a sentence with one other sentence. It compares it with every leftover sentence of a filing and keeps the best. So 488 filings were put through the same three steps against the previous-year filing of an unrelated company, in place of their own:

| 146,247 sentences, matched against... | Exact | Near | Reworded | New |
|---|---|---|---|---|
| the company's own previous filing | 70.6% | 12.1% | 1.9% | 15.4% |
| an unrelated company's | 0.3% | 0.4% | 3.8% | 95.5% |
| an unrelated company's, same industry division | 0.3% | 0.6% | 4.8% | 94.4% |

The exact and near steps find almost nothing in a stranger's filing: 0.6% of sentences together (0.9% within an industry division), against 82.7% in the company's own. And what they find is real. An exact match with a stranger is a sentence both companies filed, the stock language of the next section.

The reworded step is different. **About 4% of a filing's sentences have a cosine of 0.847 or more with some sentence in an unrelated company's filing.** Two risk sections cover the same subjects, and with a few hundred candidates to choose from, the closest is often close. This is not the error rate of the reworded label: in a real pair far fewer sentences get that far (the first two steps leave 17% of them here), and they are compared with last year's leftovers, not a whole filing. It is a floor to read the label against. A reworded sentence has a close relative in last year's filing, and about one sentence in 25 has one in a stranger's.

**The embedding reads which words are there, not their order.** 20,000 sentences were each compared with themselves, words shuffled at random. The median cosine is 0.91, and 95.5% of the shuffled sentences clear τ_p: the reworded step would call word salad a rewording of the original. By runs of five words the two have nothing in common (2 of the 20,000 clear τ_v). This is the reversed-claim result of check 1 from the other side. Cosine between these embeddings measures shared vocabulary far more than meaning.

**Comparing leftovers only with leftovers costs little.** For 456 filings, the 19,317 sentences labeled new were compared with every sentence of last year's filing, not only those exact matching left over. 882 of them (4.6%) would get another label: 288 near and 594 reworded. The 594 are 3.1% of the new sentences, no more than a stranger's filing produces, so they are weak evidence of anything missed. The 288 are the real cost: 1.5% of new sentences, about 0.2% of all sentences, are light edits of a sentence that was also kept word for word.

## Shared with peers

A sentence is **shared with peers** when its normalized form is filed, in the same year, by at least 5 other companies.

"Other companies" needs care. Some filers are one business filing several times, a parent and its subsidiaries or a family of funds, and their sections are near-copies of each other. Companies whose sets of sentences have a Jaccard of 0.5 or more in a year are therefore treated as one group, and reach is counted in groups. 7,159 of the 72,613 filings are in a group of two or more; the largest group has 53.

| Filed word for word by at least... | All sentences | Full sentences (10 words or more, not a heading) |
|---|---|---|
| 1 other group | 27.4% | 25.2% |
| 5 other groups | 9.0% | 7.3% |
| 10 other groups | 5.4% | 4.0% |
| 25 other groups | 2.5% | 1.5% |

**Sharing has grown about sixfold.** The share of full sentences filed by at least 5 other groups was 2.2% for fiscal 2006, 5.6% for 2010, 7.4% for 2015 and 13.7% for 2020. The largest single rise is the last, from 9.9% for 2019.

The most widely shared full sentences of fiscal 2020 are what a reader would guess: "Investing in our common stock involves a high degree of risk." (246 other groups), "The risks and uncertainties described below are not the only ones we face." (204), and a run of sentences about defending intellectual property (142 to 192 each). One caution: because digits are normalized away, "As of December 31, 2020, we had an accumulated deficit of $246.0 million." counts as one sentence across the 195 other groups that state an accumulated deficit in those words. That is the definition doing what it says, and it is why the full-sentence column is the better guide.

**Counting close copies as well as exact ones** raises the figure by as much as "close" is loosened. On a sample of 150,000 sentences (10,000 a year, with Jaccard estimated by MinHash), the share of full sentences filed by at least 5 other groups:

| A copy is... | Share of full sentences |
|---|---|
| word for word | 6.8% |
| or a Jaccard of at least 0.8 | 8.3% |
| or at least 0.5 | 15.4% |
| or at least τ_v (0.198) | 28.8% |

The sample takes the same number of sentences from every year, which is why its word-for-word figure is 6.8% and not the 7.3% above. The last row is not a finding. τ_v was tuned for comparing a sentence with one company's previous filing. Two unrelated sentences clear it now and then (2 of the 50,000 random pairs in the first baseline), and a year holds more than a million sentences to be compared with, so a match at τ_v somewhere in a year's filings says little about copying. The first three rows are the range: **between 7% and 15% of full sentences are filed, exactly or closely, by at least five other groups.**

## Limits

- **No person labeled anything.** Both judges are language models from the same family, and they may share blind spots. Where they disagree, this page says so and shows both.
- **The reworded tier rests on little.** Its threshold was set on 141 judged pairs and its precision is known only to within about 14 points either way. It is 1.6% of sentences, so no headline number here depends on it, and later milestones report their results with and without it.
- **Edits that change a claim count as carried over.** A sentence kept with one word reversed, or with a clause of new content added, is "near". About 5% of near matches were judged to say something different.
- **Company names stay in the text.** A sentence that differs between two companies only by a name is not an exact match, so word-for-word sharing between companies is understated.
- **Cut-off sentences.** Where the source text breaks a sentence at a page boundary, the pieces are matched as sentences in their own right. Judge 2 met 41 such pieces among 400 pairs.
- **The previous filing is the only comparison.** A sentence dropped one year and restored the next is new when it returns.

## What changed from the plan

The plan is `docs/superpowers/plans/2026-11-11-m2-measurement.md`. Four things were done differently, each recorded when it was decided:

1. **A second rubric and a second round of judging**, after the first round's kappa of 0.34 was traced to judge 1 misreading the first rubric. Both rounds are kept.
2. **τ_p re-tuned on real pairs**, after the value from constructed pairs proved two-thirds wrong on them.
3. **Near-verbatim sharing between companies is reported at Jaccard 0.5 and 0.8 as well as at τ_v**, because τ_v is far too weak a test against a whole year's sentences.
4. **The baselines use smaller samples** than planned (50,000 random pairs in place of 200,000, 500 filings against a stranger in place of 2,000, and 10,000 sentences a year for close copies between companies in place of 20,000), to keep the run to half an hour.
