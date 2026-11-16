# How long does a risk sentence last, and what drives rewriting?

Once a company adds a sentence to its risk section, how long does it stay, what makes it stay longer, and how much of a filing's rewriting belongs to the company, its industry and the year. Every result here is in `results/explain.json`, written by:

```
uv run python -m pipeline.run chains explain
```

`chains` takes a minute. `explain` takes 45 minutes on an M3 laptop, nearly all of it the two hazard models.

The decision rules (what counts as a failed proportional-hazards check and what follows, the convergence standard for the Bayesian fits, the primary scale for the variance decomposition) were fixed in the analysis plan, `docs/superpowers/plans/2026-11-11-m3-explain.md`, before any result.

## The answer

**A new sentence has a 23% chance of being gone the next year, and the risk falls every year it survives.** Half of all added sentences are gone within five filings; 30% are still there after thirteen.

**Sentences shared with other companies last twice as long.** A sentence that at least five other companies also filed, the year it was added, has a median life of ten filings against five for a sentence of the company's own. The protection is strongest in the first year and has faded by the fourth.

**How much a company rewrites in a given year is mostly not explained by which company it is.** Of the variation in the share of new sentences across filings, 11% belongs to the company, 4% to the year, 1% to the industry, and 84% is the same company differing from itself from one year to the next.

## What was followed

A chain starts where a sentence is labeled new ([measurement](measurement.md)) and follows the sentence's best match through each later filing of the same company. It ends when a filing has no match for it (removed), or when the data stop: the company's last filing, a missing year, or fiscal 2020 (censored). Time is counted in filings.

| | Sentences |
|---|---|
| Added sentences followed | 1,884,229, from 10,277 companies |
| Removed while watched | 428,241 |
| Still there when the data end | 1,455,988 |
| ...of which added in the last filing seen, so never watched | 703,350 |
| Shared with at least 5 other companies when added | 118,986 (6.3%) |
| Left out: already in a company's first filing, so of unknown age | 7,076,404 |

The last row is the largest. A sentence present in the first filing the data hold for a company could be one year old or twenty, so it starts no chain. What follows describes sentences whose arrival was seen.

## How long a sentence lasts

Kaplan–Meier survival from counts by age. Filings are annual, so the life table is exact and needs no model.

| Filings since it was added | Still at risk | Removed at the next filing | Survive past it |
|---|---|---|---|
| 1 | 1,180,879 | 22.8% | 77.2% |
| 2 | 591,206 | 14.5% | 66.0% |
| 3 | 328,845 | 11.1% | 58.6% |
| 4 | 186,758 | 9.5% | 53.1% |
| 5 | 111,410 | 7.8% | 48.9% |
| 10 | | 6.4% | 35.4% |
| 13 | | 5.1% | 29.7% |

The yearly risk of removal falls from 23% to 15% to 11% and settles at 5 to 7% from the sixth filing on. A sentence that has lasted is likely to keep lasting. The intervals are narrow at this size: survival past the first filing is 77.1% to 77.2%.

By what kind of sentence it is:

| Sentences | Chains | Survive one filing | Survive five | Median life (filings) |
|---|---|---|---|---|
| Shared with 5 or more other companies when added | 118,986 | 86.8% | 63.4% | 10 |
| The company's own | 1,765,243 | 76.5% | 48.0% | 5 |
| Up to 20 words | 631,052 | 76.7% | 49.5% | 5 |
| 21 to 32 words | 630,148 | 77.3% | 49.1% | 5 |
| Over 32 words | 623,029 | 77.4% | 48.2% | 5 |
| Added for fiscal 2007–10 | 548,296 | 73.5% | 46.5% | 5 |
| Added for fiscal 2011–14 | 499,453 | 79.1% | 51.3% | 6 |
| Added for fiscal 2015–17 | 365,371 | 78.6% | 48.0% | 5 |
| Added for fiscal 2018–20 | 471,109 | 78.7% | | not reached |

Shared sentences stand apart. Length makes no difference. Sentences added during and just after the financial crisis were cut sooner than those added later.

## What makes a sentence last

The table above compares one thing at a time, and the things overlap: shared sentences sit in particular companies and particular parts of a section. A model holds the others fixed.

### The check the design asked for

A Cox model with a test of proportional hazards, on 123,606 watched chains from 1,510 sampled companies, with errors clustered by company. It puts the hazard ratio for a shared sentence at 0.63 (0.57 to 0.70) and **rejects proportional hazards for it** (p = 0.0001): the effect of being shared is not the same at every age. By the rule fixed in the plan, the reported model therefore lets that effect differ by age.

One trap worth recording: lifelines' default for this test ranks the event times, ranks break ties by row order, and with annual data nearly every time is tied. With the rows in company order, every company-level covariate appears to change with age. The test here uses the Kaplan–Meier transform, which has no such dependence.

### The reported model

A discrete-time hazard model: the chance a sentence is removed at each filing, given that it reached it, on the complementary log-log scale (the grouped-time form of proportional hazards), with a baseline for each age and a random effect for each company. It is fit by NUTS on a sample of 2,000 companies drawn in proportion from each industry. 1,529 of them added at least one sentence that was watched, 226,769 sentences in all.

Hazard ratios, with 95% credible intervals. Below 1, the sentence is less likely to be removed.

| | Hazard ratio | 95% interval |
|---|---|---|
| **Shared with 5 or more other companies, at its first filing** | 0.58 | 0.56 to 0.61 |
| ...at its second and third | 0.70 | 0.65 to 0.74 |
| ...from its fourth on | 0.95 | 0.85 to 1.06 |
| In the middle third of the section (against the top third) | 0.80 | 0.79 to 0.81 |
| In the bottom third | 0.86 | 0.84 to 0.87 |
| In a section of middling length (against the shortest third) | 1.20 | 1.17 to 1.24 |
| In one of the longest sections | 1.77 | 1.70 to 1.85 |
| Of middling length itself (against the shortest third) | 0.98 | 0.97 to 1.00 |
| Among the longest sentences | 0.98 | 0.96 to 1.00 |
| Added for fiscal 2011–14 (against 2007–10) | 0.77 | 0.75 to 0.78 |
| Added for fiscal 2015–17 | 0.84 | 0.82 to 0.86 |
| Added for fiscal 2018–20 | 0.81 | 0.78 to 0.84 |

Read plainly:

- **A shared sentence is 42% less likely to be cut at its first filing, 30% less at its second and third, and no different from any other sentence after that.** Boilerplate is protected while it is new. Once any sentence has survived three filings, it is furniture.
- **Sentences at the top of a section turn over most.** The opening third is where a company describes what changed this year.
- **Long sections churn more**, per sentence: a sentence in one of the longest sections is 77% more likely to be cut at any filing than one in the shortest.
- **A sentence's own length does not matter.**
- **Industry does not matter.** All eleven industry groups (the ten SIC divisions, and companies with no industry code) have hazard ratios between 0.98 and 1.04, each with an interval that includes 1.
- **The company matters a great deal.** The standard deviation of the company effect is 0.91 (0.87 to 0.95) on the log-hazard scale: a company one standard deviation above the average cuts its new sentences at two and a half times the average rate.

Both fits met the convergence standard fixed in the plan (R-hat at most 1.01, at least 400 effective draws for every reported parameter, no divergent transitions): R-hat 1.008 and 539 effective draws at worst, in 6,000 draws. The model with one effect for shared sentences at all ages gives 0.65 (0.62 to 0.67) and the same answers for everything else.

## Where rewriting comes from

A different question, about filings and not sentences: each of the 47,643 paired filings has a share of new sentences. How much of the variation in that share belongs to the company, its industry and the year? A model with a random offset for each of 10,779 companies, 11 industry groups and 14 fiscal years, fit by NUTS:

| Share of the variation in a filing's share of new sentences | Estimate | 95% interval |
|---|---|---|
| The same company, from one year to another | 83.5% | 77.8% to 86.0% |
| The company | 11.0% | 9.9% to 12.2% |
| The fiscal year | 3.6% | 1.7% to 8.9% |
| The industry | 1.4% | 0.4% to 6.1% |

**Rewriting is an event, not a trait.** Knowing which company filed tells you a ninth of what there is to know about how much of its section is new. Most of the variation is a company rewriting heavily one year and hardly at all the next.

The year effects are small beside that but they are the ones the yearly shares show: fiscal 2008 is 6.5 points above the average year, 2016 is 2.8 below, and 2020 is 2.0 above.

The industry interval is wide because there are only eleven groups to estimate a spread from. That is what eleven groups can say.

**On another scale the shares move, and the order mostly holds.** The model above is linear in the share, the scale fixed in the plan as primary. Fit to the logit of the share, which stretches differences near zero, the company's part is 20.9% (16.5% to 23.0%), the industry's 8.6% (2.1% to 28.0%), the year's 3.3% (1.6% to 7.8%), and the rest 66.8% (52.6% to 72.1%). On either scale, year-to-year variation within a company is the largest part by far, and the company comes second. The industry's part is larger on the logit scale and poorly pinned down on both.

## Limits

- **Only sentences whose arrival was seen.** The 7.1 million sentences already in companies' first filings are left out, and they are probably the oldest and most durable text there is. Survival here is the survival of additions.
- **A missed match ends a chain early.** If the matcher fails to link a sentence to its next-year version, the sentence counts as removed and its successor as new. The measurement puts that at about 0.5% of carried-over sentences a year, which compounds over a long chain, so long survival is understated a little.
- **A reversed claim continues a chain.** The matcher calls a sentence carried over when its words are kept and its meaning changed (`docs/measurement.md`), so "lasting" means the wording lasted.
- **Merges.** Two of last year's sentences may continue into the same sentence this year; both chains go on.
- **The hazard model is fit on a sample** of 2,000 companies, and its sentence characteristics are cut into thirds. The survival tables use every chain.
- **Shared is fixed at birth.** A sentence that other companies adopt later is still counted as the company's own.
- **These are associations.** A shared sentence lasts longer; nothing here says that sharing is why.

## Decisions made during fitting

The analysis plan fixed the decision rules. These are the choices it left open, or that the fits forced.

1. **The variance model samples industry effects directly**, where the hazard model scales them from unit draws. Scaled, the variance fit misses the convergence standard on the logit scale: 4 divergent transitions in 4,000 draws, and still 1 in 8,000 with the step size tuned for 95% and for 99% acceptance, each time where the industry spread was drawn large. Most industries have hundreds or thousands of companies behind them, so their offsets are well measured, and a well-measured offset is better sampled directly. Sampled that way, both scales have no divergences and four to six times the effective draws, and the estimates are the same.
2. **The hazard model's step size is tuned for 95% acceptance**, not the default 80%. On simulated data the default gave one divergent transition in 4,000 draws in one seed of three, from the same cause; at 95%, none in any seed. Here the scaled form is the right one, because an industry's effect on the hazard is barely pinned down.
3. **The hazard fits use 1,500 draws a chain**, not 1,000, to clear 400 effective draws for every parameter with room to spare.
4. **The proportional-hazards test uses the Kaplan–Meier time transform**, for the reason given above.
