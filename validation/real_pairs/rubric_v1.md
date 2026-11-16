# Rubric v1: was this sentence carried over from last year?

Each item is a pair of sentences from the risk-factor sections of one company's annual reports (SEC Form 10-K) for two consecutive years. A is from the earlier report and B is from the later one. Decide what B is in relation to A, and give exactly one label.

**same**: B is A, carried over. At most light edits: a changed name, date or number; a word or two swapped, added or dropped; punctuation; one more item in a long list. What A claims, B claims.

**paraphrase**: B makes the same claim as A about the same risk, in substantially different words: reworded, reordered, split or merged. Nothing material is added or dropped.

**different**: B does not make the same claim as A. It is about another risk, or it says the opposite, or it says materially more or less than A (a new cause, a new consequence, a different likelihood or scope), however many words the two share.

How to decide:

- Judge what the sentences claim, not how many words they share.
- An updated figure or date is a light edit ("same"), unless the figure is the point of the sentence and the change reverses it (a loss becomes a profit).
- If B is A plus words that add a new risk, cause or consequence, it is "different". If the added words only elaborate (an example of something already named, a synonym), it is "same".
- Headings and list fragments are judged the same way: do they say the same thing?
- Between "same" and "paraphrase": would a reader say B was copied from A and touched up ("same"), or written afresh to say the same thing ("paraphrase")?
- Between "paraphrase" and "different": if you can name something B asserts that A does not, or the reverse, and it matters, it is "different".

Examples:

- A: "We had net losses of $4.2 million in 2017 and may not become profitable." B: "We had net losses of $6.8 million in 2018 and may not become profitable." → same
- A: "We rely on a limited number of suppliers for key components." B: "We rely on a small number of suppliers for key components of our products." → same
- A: "The loss of any of our key executives could harm our business." B: "Our business could suffer if one or more members of senior management were to leave." → paraphrase
- A: "Our products may contain defects, which could result in claims against us." B: "Customers could bring claims against us if defects are found in the products we sell." → paraphrase
- A: "A natural disaster could disrupt our operations." B: "A natural disaster or a public health emergency, such as a pandemic, could disrupt our operations and reduce demand for our services." → different
- A: "We face intense competition in the markets we serve." B: "We depend on third parties to manufacture our products." → different

Reply with a JSON array and nothing else: one object per pair, in the order given, with the keys "id" and "label", where "label" is "same", "paraphrase" or "different".

Pairs:
