# Rubric v2: was this sentence carried over from last year?

Each item is a pair of sentences from the risk-factor sections of one company's annual reports (SEC Form 10-K) for two consecutive years. A is from the earlier report and B is from the later one. The question is whether the company reused A when it wrote B. Give exactly one label.

**same**: B is A, reused. The company kept the sentence and at most touched it up.

**paraphrase**: B makes the same claim as A about the same risk, but was written afresh: substantially different wording, order or structure, with nothing material added or dropped.

**different**: B is not A reused, or it is A changed so that it now says something else: another risk, the opposite, or a specific new risk, cause or consequence that A did not have.

What counts as a touch-up (label **same**):

- A changed name, product, counterparty, date, number or percentage, even though the fact stated is then a new one. "Acme Staffing is a vendor management company." → "Borealis is a vendor management company." is **same**. So is "Sales to our top five customers were 41% of revenue in 2012." → "Sales to our top five customers were 38% of revenue in 2013." The sentence was reused; only its details were updated.
- A status brought up to date. "The agency plans to issue final rules in mid-2015." → "The agency issued final rules in October 2015." is **same**.
- A word or two swapped, added or dropped; a general qualifier added ("in a timely manner", "or other countries"); a clause dropped with nothing put in its place; clauses reordered with the same words.
- **A cut-off piece.** The text was split by machine, so B is sometimes only the tail, the head or one list item of A, possibly with a stray page number or page header in front ("27 - terms acceptable to us."). If everything in B is in A, label **same**: those words were carried over.

What makes it **different**:

- B adds a specific new risk, cause or consequence. "A natural disaster could disrupt our operations." → "A natural disaster or a pandemic such as COVID-19 could disrupt our operations and reduce demand." is **different**. So is A followed by a further sentence or clause of new information.
- The update reverses the point of the sentence. "Our credit rating carries a stable outlook." → "Our credit rating carries a negative outlook." is **different**. So is "We had no borrowings outstanding." → "We had $200 million outstanding.", and net income becoming a net loss.
- B is about something else, even if the two share a stock phrase such as "could have a material adverse effect on our business".

Between **same** and **paraphrase**: would a reader say B was copied from A and touched up (**same**), or written again to say the same thing (**paraphrase**)? "The loss of any of our key executives could harm our business." → "Our business could suffer if one or more members of senior management were to leave." is **paraphrase**.

Judge what happened to the sentence, not how many words the two share.

Reply with a JSON array and nothing else: one object per pair, in the order given, with the keys "id" and "label", where "label" is "same", "paraphrase" or "different".

Pairs:
