You are helping build a test set for a tool that compares sentences from the risk-factor sections of companies' annual reports (SEC Form 10-K). For each sentence below, write three new versions. They are used to test whether the tool can tell a copied sentence from a reworded one from a changed one, so each version must be exactly what its definition says.

1. "verbatim": the same sentence with only surface details changed, the way a company updates last year's sentence. Change names (companies, people, places, products), dates and numbers to different plausible ones. If the sentence has none of those, change one or two words to close synonyms. Do not change what the sentence claims, and leave every other word exactly as it is.

2. "paraphrase": the same claim in different words. Reword and restructure the sentence so that few runs of five or more consecutive words survive, but keep every fact, condition and qualification, and add nothing. Keep the register of an annual report.

3. "different_hard": a sentence that looks like an edit of the original but makes a different claim. Keep the topic and most of the original words, and change the substance: reverse the claim, change the cause or the consequence, or swap in a different risk. A reader comparing the two should say they do not mean the same thing.

Reply with a JSON array and nothing else: one object per sentence, in the order given, with the keys "id", "verbatim", "paraphrase" and "different_hard".

Sentences:
